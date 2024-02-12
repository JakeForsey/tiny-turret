#!venv/bin/python3
import os
import multiprocessing as mp
import time

import cv2
import numpy as np
from simple_pid import PID
from serial import Serial
import torch
from ultralytics.models import YOLO

BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)

CAMERA_ID = int(os.getenv("CAMERA_ID", "0"))
# 0: person, 39: bottle, 40: wine glass, 41: cup, 46: banana, 67: cell phone, 65: remote, 76: scissors
CLASS_ID = int(os.getenv("CLASS_ID", "41"))
MODEL_SIZE = os.getenv("MODEL_SIZE", "x")
DEBUG = os.getenv("DEBUG", "1") == "1"
assert DEBUG
WIDTH = int(os.getenv("WIDTH", "640"))
HEIGHT = int(os.getenv("HEIGHT", "480"))
USB_DEVICE = os.getenv("USB_DEVICE", "/dev/ttyACM0")

time_fn = time.perf_counter

class PWMPort:
    def __init__(self, port: int, usb_device: str) -> None:
        assert port in (0, 1, 2)
        self.process = None | mp.Process
        self.pulse_width: mp.Value = mp.Value("d", 1.5)
        self.port: int = port
        self.usb_device: str = usb_device

    def __enter__(self) -> 'PWMPort':
        self.process = mp.Process(target=start_pwm, args=(self.usb_device, self.port, self.pulse_width))
        self.process.daemon = True
        self.process.start()
        return self
    
    def __exit__(self, type, value, traceback) -> None:
        self.process.terminate()
    
    def increment(self, increment: float) -> None:
        new = self.pulse_width.value + increment
        new = min(2.0, new)
        new = max(1.0, new)
        self.pulse_width.value = new

    def set(self, new: float) -> None:
        new = min(2.0, new)
        new = max(1.0, new)
        self.pulse_width.value = new

    def get(self) -> float:
        return self.pulse_width.value

class Camera:
    def __init__(self, camera_id: int, height: int, width: int) -> None:
        self.camera_id = camera_id
        self.height = height
        self.width = width
        self.capture: None | cv2.VideoCapture = None
    
    def __enter__(self) -> 'Camera':
        self.capture = cv2.VideoCapture(self.camera_id)
        assert self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        assert self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        return self
    
    def __exit__(self, type, value, traceback) -> None:
        self.capture.release()

    def get_frame(self) -> np.ndarray:
        return self.capture.read()[1] [..., ::-1] 

def run_pwm(ser: Serial, port: int, pulse_width: mp.Value):
    step = milliseconds(20.)
    while True:
        t = time_fn()
        set_high(ser, port)
        sleep_until(t + milliseconds(pulse_width.value))
        set_low(ser, port)
        ser.flush()
        sleep_until(t + step)

def start_pwm(usb_device: str, port: int, pulse_width: mp.Value) -> None:
    with Serial(usb_device) as ser:
        ser.set_low_latency_mode(True)
        configure_write(ser, port)
        try:
            run_pwm(ser, port, pulse_width)
        except Exception as e:
            print(e)
            set_low(ser, port)

def set_high(ser: Serial, port: int) -> None:
    ser.write(f"@00P{port}ff\r".encode())

def set_low(ser: Serial, port: int) -> None:
    ser.write(f"@00P{port}00\r".encode())

def configure_write(ser: Serial, port: int) -> None:
    ser.write(f"@00D{port}00\r".encode())

def milliseconds(seconds: float) -> float:
    return seconds / 1000.

def sleep_until(end: float) -> None:
    now = time_fn()
    while now < end:
        now = time_fn()

def init_model(class_id: int, model_size: str) -> YOLO:
    model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', pretrained=True)
    model.classes = [class_id]
    return model

def get_target_position(model: YOLO, frame: np.ndarray) -> tuple[int, int]:
    detections = model([frame])
    xywh = detections.xywh[0]
    if xywh.shape[0] == 0:
        return None
    x, y, *_ = xywh[0].tolist()
    return int(y), int(x)

def position_to_pw(pos: float, dim: int) -> float:
    return 1 + (pos / dim)

def main() -> None:
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter('output.mp4', fourcc, 10, (WIDTH, HEIGHT)) 

    cy, cx = HEIGHT // 2, WIDTH // 2
    model = init_model(CLASS_ID, MODEL_SIZE)
    x_pid = PID(.1, 1., 0.0, sample_time=1 / 15, setpoint=cx, starting_output=cx, output_limits=(0, WIDTH))
    y_pid = PID(.1, 1., 0.0, sample_time=1 / 15, setpoint=cy, starting_output=cy, output_limits=(0, HEIGHT))
    with PWMPort(1, USB_DEVICE) as y_axis,\
            PWMPort(0, USB_DEVICE) as x_axis,\
            Camera(CAMERA_ID, HEIGHT, WIDTH) as camera:

        try:
            while True:
                frame = camera.get_frame()
                target_position = get_target_position(model, frame)

                if target_position is not None:
                    ty, tx = target_position
                    dy, dx = cy - ty, cx - tx
                    print(f"[LOCKED ON]  {dx=:>4}  {dy=:>4}")
                    x = x_pid(tx)
                    y = y_pid(ty)
                    x_axis.set(position_to_pw(x, WIDTH))
                    y_axis.set(position_to_pw(y, HEIGHT))
                    if DEBUG:
                        cv2.arrowedLine(frame[..., ::-1], (cx, cy), (tx, ty), BGR_GREEN, 1, tipLength=0.1)
                        cv2.arrowedLine(frame[..., ::-1], (cx, cy), (cx, ty), BGR_RED, 2, tipLength=0.1)
                        cv2.arrowedLine(frame[..., ::-1], (cx, cy), (tx, cy), BGR_BLUE, 2, tipLength=0.1)

                        cv2.line(frame[..., ::-1], (tx-3, ty), (tx+3, ty), BGR_GREEN, 1)
                        cv2.line(frame[..., ::-1], (tx, ty-3), (tx, ty+3), BGR_GREEN, 1)
                else:
                    print(f"[NO LOCK ON]")
                    time.sleep(0.01)

                if DEBUG:
                    print(frame.shape)
                    writer.write(frame[..., ::-1])
                    # cv2.imshow("image", frame[..., ::-1])
                    # if cv2.waitKey(10) & 0xFF == ord('q'):
                    #     break
        except:
            writer.release()


if __name__ == "__main__":
    main()
