#!/usr/bin/env python3
"""
camera_manager.py

Manages camera acquisition for the Acoustic Imager.

Responsibilities:
- Detects and initializes an available camera backend (Picamera2, libcamera GStreamer, or OpenCV/V4L2).
- Runs capture in a background "latest frame" thread so the main loop never blocks on camera IO.
- Provides a thread-safe method to fetch the most recent frame for rendering.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Picamera2 optional
try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA2_AVAILABLE = True
except Exception:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False


@dataclass
class CameraConfig:
    use_camera: bool
    use_libcamera: bool
    camera_index: int
    width: int
    height: int


class CameraManager:
    """
    CameraManager encapsulates:
      - Camera backend detection
      - Background capture thread
      - Thread-safe access to the latest frame

    Frames returned by get_latest_frame() are:
      - RGB from Picamera2 (as delivered by picamera2.capture_array), OR
      - BGR from OpenCV capture paths.
    The caller can convert as needed (your original main converts RGB->BGR for picamera2).
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg

        self.camera_available: bool = False
        self.camera_type: Optional[str] = None  # "picamera2" | "opencv" | None

        self._picam2 = None
        self._cap = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ok: bool = False

        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # ----------------------------
    # Public API
    # ----------------------------
    def init(self) -> bool:
        """
        Detect and initialize a camera backend.
        Returns True if a backend is initialized; False otherwise.
        """
        if not self.cfg.use_camera:
            self._set_unavailable()
            return False

        # 1) Picamera2 (best for Pi)
        if self.cfg.use_libcamera and PICAMERA2_AVAILABLE and Picamera2 is not None:
            if self._try_init_picamera2():
                return True

        # 2) libcamera via GStreamer
        if self.cfg.use_libcamera:
            if self._try_init_gstreamer():
                return True

        # 3) OpenCV direct (V4L2 fallback)
        if self._try_init_opencv_direct():
            return True

        self._set_unavailable()
        return False

    def start(self) -> None:
        """
        Start the latest-frame worker thread if camera is initialized.
        Safe to call multiple times.
        """
        if not self.camera_available or self.camera_type is None:
            return

        self._stop = False
        with self._lock:
            self._latest_frame = None
            self._latest_ok = False

        if self._thread is not None and self._thread.is_alive():
            return

        if self.camera_type == "picamera2":
            self._thread = threading.Thread(target=self._worker_picam2, daemon=True)
        else:
            self._thread = threading.Thread(target=self._worker_opencv, daemon=True)

        self._thread.start()

    def stop(self) -> None:
        """
        Stop the worker thread and release camera resources.
        """
        self._stop = True
        time.sleep(0.02)

        # Release OpenCV capture
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

        # Stop/close Picamera2
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
        self._picam2 = None

        self.camera_available = False
        self.camera_type = None

        with self._lock:
            self._latest_frame = None
            self._latest_ok = False

        self._thread = None

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns (ok, frame). Frame is a reference to the last captured array.
        Caller should copy if it needs an immutable snapshot.
        """
        with self._lock:
            return self._latest_ok, self._latest_frame

    # ----------------------------
    # Backend init helpers
    # ----------------------------
    def _set_unavailable(self) -> None:
        self.camera_available = False
        self.camera_type = None
        self._cap = None
        self._picam2 = None

    def _try_init_picamera2(self) -> bool:
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"}
            )
            picam2.configure(config)

            # Optional controls (best-effort)
            try:
                picam2.set_controls({"AwbEnable": True, "AwbMode": 1})
            except Exception:
                pass

            picam2.start()
            time.sleep(1.0)

            test = picam2.capture_array()
            if test is None or getattr(test, "size", 0) <= 0:
                try:
                    picam2.stop()
                    picam2.close()
                except Exception:
                    pass
                return False

            self._picam2 = picam2
            self._cap = None
            self.camera_type = "picamera2"
            self.camera_available = True
            return True

        except Exception:
            self._picam2 = None
            return False

    def _try_init_gstreamer(self) -> bool:
        try:
            gst = (
                f"libcamerasrc ! video/x-raw,width={self.cfg.width},height={self.cfg.height},framerate=30/1 ! "
                f"videoconvert ! appsink"
            )
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                cap.release()
                return False

            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) <= 0:
                cap.release()
                return False

            self._cap = cap
            self._picam2 = None
            self.camera_type = "opencv"
            self.camera_available = True
            return True

        except Exception:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
            return False

    def _try_init_opencv_direct(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.cfg.camera_index)

            if not cap.isOpened():
                cap.release()
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm up
            for _ in range(3):
                cap.read()

            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) <= 0:
                cap.release()
                return False

            self._cap = cap
            self._picam2 = None
            self.camera_type = "opencv"
            self.camera_available = True
            return True

        except Exception:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
            return False

    # ----------------------------
    # Worker threads
    # ----------------------------
    def _worker_picam2(self) -> None:
        while not self._stop:
            try:
                if self._picam2 is None:
                    time.sleep(0.01)
                    continue

                fr = self._picam2.capture_array()
                if fr is not None and getattr(fr, "size", 0) > 0:
                    with self._lock:
                        self._latest_frame = fr
                        self._latest_ok = True
            except Exception:
                time.sleep(0.005)

    def _worker_opencv(self) -> None:
        while not self._stop:
            try:
                if self._cap is None:
                    time.sleep(0.01)
                    continue

                ok = self._cap.grab()
                if not ok:
                    time.sleep(0.005)
                    continue

                ok, fr = self._cap.retrieve()
                if ok and fr is not None and getattr(fr, "size", 0) > 0:
                    with self._lock:
                        self._latest_frame = fr
                        self._latest_ok = True
            except Exception:
                time.sleep(0.005)
