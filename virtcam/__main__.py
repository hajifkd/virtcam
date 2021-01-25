import tensorflow as tf
from cv2 import cv2  # for pylint...
from tensorflow.keras.preprocessing import image
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.model import BodyPixModelWrapper
import pyfakewebcam
from inotify.adapters import Inotify
from threading import Thread
import numpy as np
import os
import json


def bodypix_model() -> BodyPixModelWrapper:
    return load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
    ))


def get_dimension(cap: cv2.VideoCapture) -> (int, int):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


class VirtualCameraFilter:
    def __init__(self, real_cam_index: int, virt_device_name: str, model: BodyPixModelWrapper, background: np.ndarray, threshold: float):
        self.model = model
        self.real_cam_index = real_cam_index
        self.real_cam = cv2.VideoCapture(real_cam_index)
        self.dim = get_dimension(self.real_cam)
        self.real_cam.release()
        self.virt_cam = pyfakewebcam.FakeWebcam(virt_device_name, *self.dim)
        self.running = False
        self.virt_cam.schedule_frame(
            np.zeros((self.dim[1], self.dim[0], 3), dtype=np.uint8))
        self.background = tf.image.resize(
            [background], (self.dim[1], self.dim[0]))[0]
        self.threshold = threshold

    def run(self):
        self.real_cam = cv2.VideoCapture(self.real_cam_index)
        while self.running:
            _, data = self.real_cam.read()
            data = self.process(data[..., ::-1])
            # TODO process data
            self.virt_cam.schedule_frame(data)

        self.real_cam.release()

    def process(self, raw: np.ndarray) -> np.ndarray:
        result = self.model.predict_single(raw)
        mask = result.get_mask(threshold=self.threshold, dtype=np.float32)
        bg = self.background * (1 - mask)
        fore = raw * mask
        return np.uint8(np.clip(bg + fore, 0.0, 255.0))


def main():
    config = json.load(
        open(os.environ['VIRTCAM_CONF'])) if 'VIRTCAM_CONF' in os.environ else {}
    read_cam_index = config.get("input_index") or 0
    virt_device_name = config.get("virt_device") or "/dev/video2"
    threshold = config.get("threshold") or 0.7
    background = image.img_to_array(image.load_img(config.get(
        "background"))) if "background" in config else np.full((100, 100, 3), 255, dtype=np.uint8)
    model = bodypix_model()
    filter = VirtualCameraFilter(
        read_cam_index, virt_device_name, model, background, threshold)

    # observer must after filter
    observer = Inotify()
    observer.add_watch(virt_device_name)

    thread: Thread = None

    for (_header, type_names, _path, _) in observer.event_gen(yield_nones=False):
        if 'IN_CLOSE_WRITE' in type_names:
            if not thread:
                continue
            # just close virt cam..
            filter.running = False
            thread.join()
            thread = None
        elif 'IN_OPEN' in type_names and not thread:
            filter.running = True
            thread = Thread(target=filter.run)
            thread.start()


if __name__ == '__main__':
    main()
