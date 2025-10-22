#!/usr/bin/env python3
"""
Basic Acoustic Imager Implementation

This script combines acoustic beamforming with real-time heatmap visualization:
1. Simulates a 16-microphone Fermat spiral array
2. Performs delay-and-sum beamforming in 2D space
3. Generates acoustic heatmaps showing sound source locations
4. Displays results in real-time with optional video recording

Usage:
    python basic_acoustic_imager.py

Dependencies:
    pip install numpy opencv-python matplotlib
    
System Requirements:
    - FFmpeg (optional, for video recording)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys
from typing import Tuple, Optional
import subprocess


class AcousticImager:
    """
    A basic acoustic imager that combines microphone array processing 
    with real-time heatmap visualization.
    """
    
    def __init__(self, 
                 n_mics: int = 16,
                 aperture_radius: float = 0.025,  # 2.5 cm
                 sample_rate: int = 48000,
                 frame_size: int = 1024,
                 heatmap_size: Tuple[int, int] = (640, 480)):
        """
        Initialize the acoustic imager.
        
        Args:
            n_mics: Number of microphones in the array
            aperture_radius: Radius of the microphone array in meters
            sample_rate: Audio sample rate in Hz
            frame_size: Number of samples per frame
            heatmap_size: (width, height) of output heatmap
        """
        self.n_mics = n_mics
        self.aperture_radius = aperture_radius
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.heatmap_width, self.heatmap_height = heatmap_size
        
        # Physical constants
        self.speed_of_sound = 343.0  # m/s at room temperature
        self.golden_angle = np.deg2rad(137.5)  # Golden angle for Fermat spiral
        
        # Generate microphone positions using Fermat spiral
        self.mic_positions = self._generate_mic_positions()
        
        # Time base for signal processing
        self.time_vector = np.arange(frame_size) / sample_rate
        
        # Define scanning angles for beamforming (2D grid)
        self.azimuth_angles = np.linspace(-90, 90, 181)  # -90° to +90°
        self.elevation_angles = np.linspace(-45, 45, 91)  # -45° to +45°
        
        print(f"Acoustic Imager initialized:")
        print(f"  - {n_mics} microphones in Fermat spiral array")
        print(f"  - Array radius: {aperture_radius*100:.1f} cm")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Frame size: {frame_size} samples")
        print(f"  - Heatmap resolution: {heatmap_size[0]}x{heatmap_size[1]}")
    
    def _generate_mic_positions(self) -> np.ndarray:
        """
        Generate microphone positions using Fermat spiral pattern.
        
        Returns:
            Array of shape (n_mics, 2) containing (x, y) coordinates
        """
        positions = []
        c = self.aperture_radius / np.sqrt(self.n_mics - 1)
        
        for n in range(self.n_mics):
            r = c * np.sqrt(n)
            theta = n * self.golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append([x, y])
        
        return np.array(positions)
    
    def generate_test_signals(self, 
                            source_frequency: float = 8000.0,
                            source_azimuth: float = 30.0,
                            source_elevation: float = 0.0,
                            noise_level: float = 0.02) -> np.ndarray:
        """
        Generate simulated microphone signals with a test tone from a specific direction.
        
        Args:
            source_frequency: Frequency of test tone in Hz
            source_azimuth: Azimuth angle of source in degrees
            source_elevation: Elevation angle of source in degrees
            noise_level: Amplitude of added white noise
            
        Returns:
            Array of shape (n_mics, frame_size) containing simulated signals
        """
        # Convert angles to radians
        az_rad = np.deg2rad(source_azimuth)
        el_rad = np.deg2rad(source_elevation)
        
        # Calculate unit vector pointing to source
        source_direction = np.array([
            np.cos(el_rad) * np.cos(az_rad),  # x component
            np.cos(el_rad) * np.sin(az_rad),  # y component
            np.sin(el_rad)                    # z component (elevation)
        ])
        
        signals = []
        
        for i, (x, y) in enumerate(self.mic_positions):
            # Calculate time delay for this microphone
            # For 2D array, assume z=0 for all microphones
            mic_position = np.array([x, y])
            delay = np.dot(mic_position, source_direction[:2]) / self.speed_of_sound
            
            # Generate delayed sinusoid
            delayed_time = self.time_vector - delay
            signal = np.sin(2 * np.pi * source_frequency * delayed_time)
            
            # Add white noise
            signal += noise_level * np.random.randn(len(signal))
            
            signals.append(signal)
        
        return np.array(signals)
    
    def beamform_2d(self, signals: np.ndarray) -> np.ndarray:
        """
        Perform 2D delay-and-sum beamforming across azimuth and elevation.
        
        Args:
            signals: Array of shape (n_mics, frame_size) containing microphone signals
            
        Returns:
            2D array of beamforming power values (elevation x azimuth)
        """
        power_map = np.zeros((len(self.elevation_angles), len(self.azimuth_angles)))
        
        for i, elevation in enumerate(self.elevation_angles):
            for j, azimuth in enumerate(self.azimuth_angles):
                # Convert to radians
                az_rad = np.deg2rad(azimuth)
                el_rad = np.deg2rad(elevation)
                
                # Calculate steering vector
                steering_direction = np.array([
                    np.cos(el_rad) * np.cos(az_rad),
                    np.cos(el_rad) * np.sin(az_rad)
                ])
                
                # Sum delayed signals
                summed_signal = np.zeros(self.frame_size)
                
                for mic_idx, (x, y) in enumerate(self.mic_positions):
                    mic_position = np.array([x, y])
                    delay = np.dot(mic_position, steering_direction) / self.speed_of_sound
                    
                    # Apply delay by phase shift in frequency domain (more accurate)
                    signal_fft = np.fft.fft(signals[mic_idx])
                    frequencies = np.fft.fftfreq(self.frame_size, 1/self.sample_rate)
                    phase_shift = np.exp(-1j * 2 * np.pi * frequencies * delay)
                    delayed_signal = np.real(np.fft.ifft(signal_fft * phase_shift))
                    
                    summed_signal += delayed_signal
                
                # Calculate power
                power_map[i, j] = np.sum(summed_signal**2)
        
        return power_map
    
    def power_to_heatmap(self, power_map: np.ndarray) -> np.ndarray:
        """
        Convert beamforming power map to normalized heatmap.
        
        Args:
            power_map: 2D array of power values
            
        Returns:
            Normalized heatmap suitable for visualization (0-255, uint8)
        """
        # Convert to dB scale
        power_db = 10 * np.log10(power_map / np.max(power_map) + 1e-10)
        
        # Normalize to 0-255 range
        min_db = np.min(power_db)
        max_db = np.max(power_db)
        normalized = ((power_db - min_db) / (max_db - min_db) * 255).astype(np.uint8)
        
        # Resize to desired heatmap dimensions
        heatmap = cv2.resize(normalized, (self.heatmap_width, self.heatmap_height))
        
        return heatmap
    
    def create_visualization_frame(self, 
                                 heatmap: np.ndarray,
                                 frame_number: int = 0,
                                 timestamp: float = 0.0,
                                 source_info: str = "") -> np.ndarray:
        """
        Create a complete visualization frame with heatmap and annotations.
        
        Args:
            heatmap: Normalized heatmap array (0-255, uint8)
            frame_number: Current frame number for display
            timestamp: Current timestamp in seconds
            source_info: String describing the simulated source
            
        Returns:
            BGR image ready for display/recording
        """
        # Apply color map to create colored heatmap
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create a copy for annotations
        output_frame = colored_heatmap.copy()
        
        # Add coordinate grid lines
        # Vertical lines for azimuth
        for i in range(0, self.heatmap_width, self.heatmap_width // 8):
            cv2.line(output_frame, (i, 0), (i, self.heatmap_height), (255, 255, 255), 1)
        
        # Horizontal lines for elevation
        for i in range(0, self.heatmap_height, self.heatmap_height // 6):
            cv2.line(output_frame, (0, i), (self.heatmap_width, i), (255, 255, 255), 1)
        
        # Add angle labels
        cv2.putText(output_frame, "-90°", (10, self.heatmap_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_frame, "+90°", (self.heatmap_width - 40, self.heatmap_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_frame, "-45°", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_frame, "+45°", (10, self.heatmap_height // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(output_frame, f"Frame: {frame_number}", 
                   (self.heatmap_width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Time: {timestamp:.2f}s", 
                   (self.heatmap_width - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add source information
        if source_info:
            cv2.putText(output_frame, source_info, 
                       (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add title
        cv2.putText(output_frame, "Acoustic Imager - Real-time Beamforming", 
                   (self.heatmap_width // 2 - 200, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame
    
    def run_simulation(self, 
                      duration_seconds: float = 10.0,
                      fps: int = 10,
                      source_frequency: float = 8000.0,
                      source_azimuth: float = 30.0,
                      source_elevation: float = 0.0,
                      moving_source: bool = False,
                      record_video: bool = True,
                      output_filename: str = "acoustic_imager_output.mp4"):
        """
        Run the acoustic imager simulation.
        
        Args:
            duration_seconds: How long to run the simulation
            fps: Frames per second for display and recording
            source_frequency: Frequency of test signal in Hz
            source_azimuth: Initial azimuth of source in degrees
            source_elevation: Initial elevation of source in degrees
            moving_source: Whether to simulate a moving source
            record_video: Whether to record output video
            output_filename: Name of output video file
        """
        print(f"\nStarting acoustic imager simulation:")
        print(f"  - Duration: {duration_seconds} seconds")
        print(f"  - Frame rate: {fps} FPS")
        print(f"  - Source: {source_frequency} Hz at ({source_azimuth}°, {source_elevation}°)")
        print(f"  - Moving source: {moving_source}")
        print(f"  - Recording: {record_video}")
        
        # Calculate total frames
        total_frames = int(duration_seconds * fps)
        frame_interval = 1.0 / fps
        
        # Setup video recording if requested
        ffmpeg_process = None
        if record_video:
            ffmpeg_process = self._setup_ffmpeg(output_filename, fps)
        
        # Create display window
        cv2.namedWindow('Acoustic Imager', cv2.WINDOW_AUTOSIZE)
        
        try:
            start_time = time.time()
            
            for frame_num in range(total_frames):
                frame_start = time.time()
                
                # Calculate current source position (if moving)
                current_azimuth = source_azimuth
                current_elevation = source_elevation
                
                if moving_source:
                    # Simple circular motion in azimuth
                    t = frame_num / total_frames * 2 * np.pi
                    current_azimuth = source_azimuth + 30 * np.sin(t)
                    current_elevation = source_elevation + 10 * np.cos(t)
                
                # Generate microphone signals
                signals = self.generate_test_signals(
                    source_frequency=source_frequency,
                    source_azimuth=current_azimuth,
                    source_elevation=current_elevation,
                    noise_level=0.02
                )
                
                # Perform beamforming
                power_map = self.beamform_2d(signals)
                
                # Convert to heatmap
                heatmap = self.power_to_heatmap(power_map)
                
                # Create visualization frame
                source_info = f"Source: {source_frequency}Hz @ ({current_azimuth:.1f}°, {current_elevation:.1f}°)"
                output_frame = self.create_visualization_frame(
                    heatmap, frame_num + 1, time.time() - start_time, source_info
                )
                
                # Display frame
                cv2.imshow('Acoustic Imager', output_frame)
                
                # Record frame if requested
                if record_video and ffmpeg_process:
                    self._send_frame_to_ffmpeg(output_frame, ffmpeg_process)
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user.")
                    break
                
                # Control frame rate
                frame_time = time.time() - frame_start
                if frame_time < frame_interval:
                    time.sleep(frame_interval - frame_time)
                
                # Progress update
                if (frame_num + 1) % (fps * 2) == 0:  # Every 2 seconds
                    elapsed = time.time() - start_time
                    print(f"Processed {frame_num + 1}/{total_frames} frames ({elapsed:.1f}s)")
            
            elapsed_time = time.time() - start_time
            actual_fps = total_frames / elapsed_time
            print(f"\nSimulation completed:")
            print(f"  - Total time: {elapsed_time:.2f} seconds")
            print(f"  - Average FPS: {actual_fps:.1f}")
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if ffmpeg_process:
                self._cleanup_ffmpeg(ffmpeg_process)
                print(f"Video saved as: {output_filename}")
    
    def _setup_ffmpeg(self, output_filename: str, fps: int) -> Optional[subprocess.Popen]:
        """Setup FFmpeg process for video recording."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.heatmap_width}x{self.heatmap_height}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                output_filename
            ]
            
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return process
        except FileNotFoundError:
            print("Warning: FFmpeg not found. Video recording disabled.")
            return None
    
    def _send_frame_to_ffmpeg(self, frame: np.ndarray, process: subprocess.Popen):
        """Send frame to FFmpeg process."""
        try:
            process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            pass  # FFmpeg process may have ended
    
    def _cleanup_ffmpeg(self, process: subprocess.Popen):
        """Clean up FFmpeg process."""
        try:
            process.stdin.close()
            process.wait(timeout=5)
        except:
            process.kill()


def main():
    """Main function to run the acoustic imager demonstration."""
    print("Basic Acoustic Imager Implementation")
    print("=" * 50)
    
    # Create acoustic imager instance
    imager = AcousticImager(
        n_mics=16,
        aperture_radius=0.025,  # 2.5 cm
        sample_rate=48000,
        frame_size=1024,
        heatmap_size=(640, 480)
    )
    
    # Run simulation with different scenarios
    print("\nRunning simulation scenarios...")
    print("Press 'q' at any time to quit early.")
    
    # Scenario 1: Static source
    print("\n1. Static source at 30° azimuth")
    imager.run_simulation(
        duration_seconds=5.0,
        fps=10,
        source_frequency=8000.0,
        source_azimuth=30.0,
        source_elevation=0.0,
        moving_source=False,
        record_video=True,
        output_filename="static_source.mp4"
    )
    
    # Scenario 2: Moving source
    print("\n2. Moving source (circular motion)")
    imager.run_simulation(
        duration_seconds=8.0,
        fps=15,
        source_frequency=6000.0,
        source_azimuth=0.0,
        source_elevation=0.0,
        moving_source=True,
        record_video=True,
        output_filename="moving_source.mp4"
    )
    
    print("\nDemo completed! Check the generated video files.")


if __name__ == "__main__":
    main()
