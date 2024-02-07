from collections import defaultdict
import os
from multiprocessing import Process, Value
import time

import cv2
import serial
import torch

BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)

CAMERA_ID = int(os.getenv("CAMERA_ID", "0"))
# 0: person, 39: bottle, 40: wine glass, 41: cup, 46: banana, 67: cell phone, 65: remote, 76: scissors
CLASS_ID = int(os.getenv("CLASS_ID", "41"))
MODEL_SIZE = os.getenv("MODEL_SIZE", "s")
DEBUG = int(os.getenv("DEBUG", "1"))
WIDTH = int(os.getenv("WIDTH", "640"))
HEIGHT = int(os.getenv("HEIGHT", "480"))
PORT = int(os.getenv("PORT", "0"))
HERTZ = int(os.getenv("HERTZ", "50"))

class PWMPin:
    def __init__(self, pin, pulse_width):
        self.pin = pin
        self.pulse_width = Value("d",pulse_width)

def write(ser, port, value):
    # Write to port:
    #  @00Pp00 -> set all pins low on port 'p'
    #  @00Ppff -> set all pins high on port 'p'
    #  @00Pp01 -> set pin 0 high on port 'p'
    ser.write(f"@00P{port}{value:0>2x}\r".encode())

def configure(ser, port, mode):
    # Configure port:
    #  @00Dp00 -> set port 'p' to write mode
    #  @00Dpff -> set port 'p' to read mode
    assert mode in ("w", "r")
    ser.write(f"@00D{port}{'00' if mode == 'w' else 'ff'}\r".encode())

def sleep(duration):
    # Accurate, but CPU intensive sleep function
    now = time.perf_counter()
    end = now + duration
    while now < end:
        now = time.perf_counter()

def sleep_until(start, value):
    remaining = value - (time.time() - start)
    if remaining > 0: 
        sleep(remaining)

def pwm(port, pwm_pins, hertz):
    step = (1 / hertz)
    max_value = sum(2 ** pp.pin for pp in pwm_pins)
    with serial.Serial("/dev/tty.usbmodem0275631", timeout=1) as ser:
        configure(ser, port, mode="w")
        while True:
            value = max_value
            start = time.time()
            write(ser, port, value)

            grouped = defaultdict(list)
            for pwm_pin in pwm_pins:
                grouped[pwm_pin.pulse_width.value].append(pwm_pin.pin)

            for pulse_width, pins in sorted(grouped.items()):
                sleep_until(start, pulse_width)                
                value -= sum(2 ** pin for pin in pins)
                write(ser, port, value)

            sleep_until(start, step)

def init_model(class_id, model_size):
    model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', pretrained=True)
    model.classes = [class_id]
    return model

def init_camera(camera_id, height, width):
    camera = cv2.VideoCapture(camera_id)
    assert camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    assert camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    return camera

def init_pwm(port, hertz):
    pwm_pins = [PWMPin(0, 1 / 1000), PWMPin(1, 1 / 1000)]
    process = Process(target=pwm, args=(port, pwm_pins, hertz))
    process.daemon = True
    process.start()
    return pwm_pins

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
    model = init_model(CLASS_ID, MODEL_SIZE)
    pwm_pins = init_pwm(PORT, HERTZ)
    camera = init_camera(CAMERA_ID, HEIGHT, WIDTH)
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
                pwm_pins[0].pulse_width.value = 0.5 / 1000
            else:
                pwm_pins[0].pulse_width.value = 6. / 1000
            if dx > 0:
                pwm_pins[1].pulse_width.value = 6. / 1000
            else:
                pwm_pins[1].pulse_width.value = 0.5 / 1000

            print(f" {dy=}, {dx=}, pw0={pwm_pins[0].pulse_width.value}, pw1={pwm_pins[1].pulse_width.value}", end="")
            
            if DEBUG:
                cv2.arrowedLine(frame[..., ::-1], (cx, cy), (tx, ty), BGR_GREEN, 1, tipLength=0.1)
                cv2.arrowedLine(frame[..., ::-1], (cx, cy), (cx, ty), BGR_RED, 2, tipLength=0.1)
                cv2.arrowedLine(frame[..., ::-1], (cx, cy), (tx, cy), BGR_BLUE, 2, tipLength=0.1)

                cv2.line(frame[..., ::-1], (tx-3, ty), (tx+3, ty), BGR_GREEN, 1)
                cv2.line(frame[..., ::-1], (tx, ty-3), (tx, ty+3), BGR_GREEN, 1)

        else:
            print(f"[NO LOCK ON]", end="")

        if DEBUG:
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
        with serial.Serial("/dev/tty.usbmodem0275631", timeout=1) as ser:
            write(ser, PORT, 0)

if __name__ == "__main__":
    main()
