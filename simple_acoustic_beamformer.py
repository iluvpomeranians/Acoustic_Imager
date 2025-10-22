#!/usr/bin/env python3
"""
Simple Acoustic Beamformer - Core Implementation

This is a simplified version that focuses on the core acoustic processing
without heavy visualization dependencies. Good for understanding the basics
and testing on systems where OpenCV/FFmpeg might not be available.

Usage:
    python simple_acoustic_beamformer.py

Dependencies:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple


class SimpleAcousticBeamformer:
    """
    A simplified acoustic beamformer for educational purposes.
    Demonstrates the core concepts without complex visualization.
    """
    
    def __init__(self, 
                 n_mics: int = 16,
                 aperture_radius: float = 0.025,  # 2.5 cm
                 sample_rate: int = 48000,
                 frame_size: int = 1024):
        """
        Initialize the beamformer.
        
        Args:
            n_mics: Number of microphones
            aperture_radius: Array radius in meters
            sample_rate: Audio sample rate in Hz
            frame_size: Samples per processing frame
        """
        self.n_mics = n_mics
        self.aperture_radius = aperture_radius
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        
        # Constants
        self.speed_of_sound = 343.0  # m/s
        self.golden_angle = np.deg2rad(137.5)
        
        # Generate microphone positions
        self.mic_positions = self._generate_fermat_spiral()
        
        # Time vector for signal generation
        self.time_vector = np.arange(frame_size) / sample_rate
        
        print(f"Simple Acoustic Beamformer initialized:")
        print(f"  - {n_mics} microphones in Fermat spiral")
        print(f"  - Array radius: {aperture_radius*100:.1f} cm")
        print(f"  - Sample rate: {sample_rate} Hz")
    
    def _generate_fermat_spiral(self) -> np.ndarray:
        """Generate Fermat spiral microphone positions."""
        positions = []
        c = self.aperture_radius / np.sqrt(self.n_mics - 1)
        
        for n in range(self.n_mics):
            r = c * np.sqrt(n)
            theta = n * self.golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append([x, y])
        
        return np.array(positions)
    
    def simulate_signals(self, 
                        source_freq: float = 8000.0,
                        source_angle: float = 30.0,
                        noise_level: float = 0.02) -> np.ndarray:
        """
        Simulate microphone signals from a point source.
        
        Args:
            source_freq: Source frequency in Hz
            source_angle: Source angle in degrees
            noise_level: White noise amplitude
            
        Returns:
            Array of simulated signals (n_mics x frame_size)
        """
        angle_rad = np.deg2rad(source_angle)
        source_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        signals = []
        for x, y in self.mic_positions:
            # Calculate propagation delay
            mic_pos = np.array([x, y])
            delay = np.dot(mic_pos, source_direction) / self.speed_of_sound
            
            # Generate delayed signal
            delayed_time = self.time_vector - delay
            signal = np.sin(2 * np.pi * source_freq * delayed_time)
            
            # Add noise
            signal += noise_level * np.random.randn(len(signal))
            signals.append(signal)
        
        return np.array(signals)
    
    def beamform_1d(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform 1D delay-and-sum beamforming.
        
        Args:
            signals: Input signals (n_mics x frame_size)
            
        Returns:
            (angles, power) - scan angles and corresponding power levels
        """
        angles = np.linspace(-90, 90, 181)
        power = []
        
        for angle in angles:
            angle_rad = np.deg2rad(angle)
            steering_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            # Sum signals with appropriate delays
            summed_signal = np.zeros(self.frame_size)
            for i, (x, y) in enumerate(self.mic_positions):
                mic_pos = np.array([x, y])
                delay = np.dot(mic_pos, steering_direction) / self.speed_of_sound
                
                # Simple time-domain delay (for demonstration)
                delay_samples = int(delay * self.sample_rate)
                delay_samples = np.clip(delay_samples, -self.frame_size//2, self.frame_size//2)
                
                if delay_samples >= 0:
                    shifted_signal = np.roll(signals[i], delay_samples)
                    shifted_signal[:delay_samples] = 0
                else:
                    shifted_signal = np.roll(signals[i], delay_samples)
                    shifted_signal[delay_samples:] = 0
                
                summed_signal += shifted_signal
            
            # Calculate power
            power.append(np.sum(summed_signal**2))
        
        # Convert to dB
        power = np.array(power)
        power_db = 10 * np.log10(power / np.max(power) + 1e-10)
        
        return angles, power_db
    
    def plot_array_geometry(self):
        """Plot the microphone array geometry."""
        plt.figure(figsize=(8, 8))
        
        # Plot microphones
        x_coords = [pos[0] for pos in self.mic_positions]
        y_coords = [pos[1] for pos in self.mic_positions]
        
        plt.scatter(x_coords, y_coords, c='red', s=100, marker='o', label='Microphones')
        
        # Add microphone numbers
        for i, (x, y) in enumerate(self.mic_positions):
            plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Draw array boundary
        circle = plt.Circle((0, 0), self.aperture_radius, fill=False, linestyle='--', color='blue')
        plt.gca().add_patch(circle)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Fermat Spiral Microphone Array ({self.n_mics} mics)')
        plt.legend()
        plt.show()
    
    def plot_beampattern(self, angles: np.ndarray, power_db: np.ndarray, source_angle: float):
        """Plot the beamforming results."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(angles, power_db, 'b-', linewidth=2, label='Beamformer Output')
        plt.axvline(x=source_angle, color='red', linestyle='--', linewidth=2, label=f'True Source ({source_angle}°)')
        
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Relative Power (dB)')
        plt.title('Acoustic Beamforming Results')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-90, 90)
        plt.ylim(-40, 0)
        plt.show()
    
    def run_demo(self):
        """Run a complete demonstration."""
        print("\n" + "="*50)
        print("ACOUSTIC BEAMFORMING DEMONSTRATION")
        print("="*50)
        
        # 1. Show array geometry
        print("\n1. Displaying microphone array geometry...")
        self.plot_array_geometry()
        
        # 2. Test different source positions
        test_angles = [0, 30, -45, 60]
        test_frequencies = [5000, 8000, 12000]
        
        for freq in test_frequencies:
            print(f"\n2. Testing beamforming with {freq} Hz source...")
            
            plt.figure(figsize=(12, 8))
            
            for i, angle in enumerate(test_angles):
                # Generate signals
                signals = self.simulate_signals(
                    source_freq=freq,
                    source_angle=angle,
                    noise_level=0.02
                )
                
                # Perform beamforming
                beam_angles, power_db = self.beamform_1d(signals)
                
                # Plot results
                plt.subplot(2, 2, i+1)
                plt.plot(beam_angles, power_db, 'b-', linewidth=2)
                plt.axvline(x=angle, color='red', linestyle='--', linewidth=2)
                plt.xlabel('Angle (degrees)')
                plt.ylabel('Power (dB)')
                plt.title(f'Source at {angle}° ({freq} Hz)')
                plt.grid(True, alpha=0.3)
                plt.xlim(-90, 90)
                plt.ylim(-40, 0)
            
            plt.tight_layout()
            plt.suptitle(f'Beamforming Results - {freq} Hz', y=1.02)
            plt.show()
        
        # 3. Resolution analysis
        print("\n3. Analyzing angular resolution...")
        self._analyze_resolution()
        
        print("\nDemo completed!")
    
    def _analyze_resolution(self):
        """Analyze the angular resolution of the array."""
        # Test closely spaced sources
        separations = [5, 10, 15, 20, 30]  # degrees
        freq = 8000  # Hz
        
        plt.figure(figsize=(12, 8))
        
        for i, sep in enumerate(separations):
            # Two sources at ±sep/2 degrees
            signals1 = self.simulate_signals(freq, sep/2, 0.01)
            signals2 = self.simulate_signals(freq, -sep/2, 0.01)
            combined_signals = signals1 + signals2
            
            # Beamform
            angles, power_db = self.beamform_1d(combined_signals)
            
            plt.subplot(2, 3, i+1)
            plt.plot(angles, power_db, 'b-', linewidth=2)
            plt.axvline(x=sep/2, color='red', linestyle='--', alpha=0.7)
            plt.axvline(x=-sep/2, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Power (dB)')
            plt.title(f'Sources ±{sep/2}° apart')
            plt.grid(True, alpha=0.3)
            plt.xlim(-45, 45)
            plt.ylim(-20, 0)
        
        plt.tight_layout()
        plt.suptitle('Angular Resolution Analysis', y=1.02)
        plt.show()


def main():
    """Main demonstration function."""
    print("Simple Acoustic Beamformer")
    print("This demonstrates the core concepts of acoustic beamforming")
    print("using a simulated 16-microphone Fermat spiral array.")
    
    # Create beamformer
    beamformer = SimpleAcousticBeamformer(
        n_mics=16,
        aperture_radius=0.025,
        sample_rate=48000,
        frame_size=1024
    )
    
    # Run demonstration
    beamformer.run_demo()


if __name__ == "__main__":
    main()
