from collections import defaultdict
import os
import threading
import time

import cv2
import serial
import torch

ONE_SECOND = 1
ONE_MILLISECOND = ONE_SECOND / 1000

BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)

CAMERA_ID = int(os.getenv("CAMERA_ID", "0"))
CLASS_ID = int(os.getenv("CLASS_ID", "41"))  # 0: person, 41: cup
SHOW = int(os.getenv("SHOW", "1"))
WIDTH = int(os.getenv("WIDTH", "320"))
HEIGHT = int(os.getenv("HEIGHT", "240"))
PORT = int(os.getenv("PORT", "0"))
HERTZ = int(os.getenv("HERTZ", "50"))

class PWMPinWidth:
    def __init__(self, pin, pulse_width):
        self.pin = pin
        self.pulse_width = pulse_width

def write(ser, port, value):
    ser.write(f"@00P{port}{value:0>2x}\r".encode())

def sleep_until(start, value):
    remaining = value - (time.time() - start)
    if remaining > 0: 
        time.sleep(remaining)

def pwm(port, set_points, hertz):
    # Commands:
    #  Configure port:
    #   @00Dp00 -> set port 'p' to write
    #   @00Dpff -> set port 'p' to read
    #  Write to port:
    #   @00Pp00 -> set all pins low on port 'p'
    #   @00Ppff -> set all pins high on port 'p'
    #   @00Pp01 -> set pin 0 high on port 'p'
    step = (1 / hertz)
    max_value = sum(2 ** sp.pin for sp in set_points)

    with serial.Serial("/dev/tty.usbmodem0275631", timeout=1) as ser:
        ser.write(f"@00D{port}00\r".encode())
        while True:
            value = max_value
            start = time.time()
            write(ser, port, value)

            grouped_set_points = defaultdict(list)
            for set_point in set_points:
                grouped_set_points[set_point.pulse_width].append(set_point.pin)

            for pulse_width, pins in sorted(grouped_set_points.items()):
                sleep_until(start, pulse_width)                
                value -= sum(2 ** pin for pin in pins)
                write(ser, port, value)

            sleep_until(start, step)

def init_model(class_id):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [class_id]
    return model

def init_camera(camera_id, width, height):
    camera = cv2.VideoCapture(camera_id)
    assert camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera

def init_pwm(port, hertz):
    set_points = [PWMPinWidth(0, 1 / 1000), PWMPinWidth(1, 1 / 1000)]
    thread = threading.Thread(name="pwm", target=pwm, args=(port, set_points, hertz))
    thread.start()
    return set_points

def get_frame(camera):
    return camera.read()[1] [..., ::-1] 

def get_target_position(model, frame):
    detections = model([frame])
    xywh = detections.xywh[0]
    if xywh.shape[0] == 0:
        return None
    x, y, *_ = xywh[0].tolist()
    return int(y), int(x)

def run():
    model = init_model(CLASS_ID)
    camera = init_camera(CAMERA_ID, WIDTH, HEIGHT)
    pwm_set_points = init_pwm(PORT, HERTZ)

    cy, cx = HEIGHT // 2, WIDTH // 2
    
    while True:
        start = time.time()

        frame = get_frame(camera)
        target_position = get_target_position(model, frame)

        if target_position is not None:
            print("[LOCKED ON]", end="")
            ty, tx = target_position
            dy, dx = cy - ty, cx - tx

            if dy > 0:
                pwm_set_points[0].pulse_width = 0.5 / 1000
            else:
                pwm_set_points[0].pulse_width = 6. / 1000
            if dx > 0:
                pwm_set_points[1].pulse_width = 6. / 1000
            else:
                pwm_set_points[1].pulse_width = 0.5 / 1000

            print(f" {dy=}, {dx=}, pw0={pwm_set_points[0].pulse_width}, pw1={pwm_set_points[1].pulse_width}", end="")
            
            if SHOW:
                cv2.arrowedLine(frame[..., ::-1], (cx, cy), (tx, ty), BGR_GREEN, 1, tipLength=0.1)
                cv2.arrowedLine(frame[..., ::-1], (cx, cy), (cx, ty), BGR_RED, 2, tipLength=0.1)
                cv2.arrowedLine(frame[..., ::-1], (cx, cy), (tx, cy), BGR_BLUE, 2, tipLength=0.1)

                cv2.line(frame[..., ::-1], (tx-3, ty), (tx+3, ty), BGR_GREEN, 1)
                cv2.line(frame[..., ::-1], (tx, ty-3), (tx, ty+3), BGR_GREEN, 1)

        else:
            print(f"[NO LOCK ON]", end="")

        if SHOW:
            cv2.imshow("image", frame[..., ::-1])
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        t = time.time() - start
        fps = 1 / t
        print(f" {fps=}")

def main():
    try:
        run()
    except Exception as e:
        print(e)
        with serial.Serial("/dev/tty.usbmodem0275631", timeout=1) as ser:
            for port in range(3):
                write(ser, port, 0)

if __name__ == "__main__":
    main()
