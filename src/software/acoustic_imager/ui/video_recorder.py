#!/usr/bin/env python3
"""
video_recorder.py

Provides video recording functionality for the Acoustic Imager.

This module implements the VideoRecorder class, which:
- Starts and stops video capture sessions
- Streams raw BGR frames to an ffmpeg subprocess
- Encodes the frames into an H.264 MP4 file
- Supports pausing and resuming recording
- Tracks the output file and number of recorded frames

It is used by the UI to record the rendered visualization frames to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import subprocess
import numpy as np


class VideoRecorder:
    def __init__(self, output_dir: Path, width: int, height: int, fps: int = 30):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.is_recording = False
        self.is_paused = False
        self.current_file = None
        self.frame_count = 0

    def start_recording(self) -> bool:
        if self.is_recording:
            return False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = self.output_dir / f"recording_{timestamp}.mp4"
            command = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(self.current_file),
            ]
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=10**8
            )
            self.is_recording = True
            self.is_paused = False
            self.frame_count = 0
            print(f"Recording started: {self.current_file}")
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.is_recording = False
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        if not self.is_recording or self.process is None:
            return False
        if self.is_paused:
            return True
        try:
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False

    def pause_recording(self) -> bool:
        if not self.is_recording or self.is_paused:
            return False
        self.is_paused = True
        return True

    def resume_recording(self) -> bool:
        if not self.is_recording or not self.is_paused:
            return False
        self.is_paused = False
        return True

    def stop_recording(self) -> Optional[str]:
        if not self.is_recording or self.process is None:
            return None
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
            out = str(self.current_file)
            print(f"Recording stopped: {out} ({self.frame_count} frames)")
            return out
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None
        finally:
            self.is_recording = False
            self.is_paused = False
            self.current_file = None
            self.frame_count = 0
            self.process = None

    def cleanup(self) -> None:
        if self.is_recording:
            self.stop_recording()
