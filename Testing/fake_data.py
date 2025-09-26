import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Parameters ===
N = 16
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.025          # 2.5 cm radius
c = aperture_radius / np.sqrt(N-1)

speed_of_sound = 343.0
fs = 48000                       # Hz sample rate
frame_size = 1024                # samples per frame
f_source = 8000                  # Hz test tone
theta_source = 30 * np.pi/180    # radians (source at 30°)
noise_level = 0.02               # white noise amplitude

# === Generate Fermat spiral mic coordinates ===
mic_positions = []
for n in range(N):
    r = c * np.sqrt(n)
    theta = n * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    mic_positions.append([x, y])
mic_positions = np.array(mic_positions)

# === Time base ===
t = np.arange(frame_size) / fs

def generate_frame():
    """Simulate one frame of microphone signals."""
    signals = []
    for (x, y) in mic_positions:
        delay = (x*np.cos(theta_source) + y*np.sin(theta_source)) / speed_of_sound
        delayed_t = t - delay
        mic_signal = np.sin(2*np.pi*f_source*delayed_t)
        mic_signal += noise_level * np.random.randn(len(mic_signal))
        signals.append(mic_signal)
    return np.array(signals)

def beamform_frame(signals):
    """Simple delay-and-sum beamforming scan across -90° to 90°."""
    angles = np.linspace(-90, 90, 181)
    power = []
    for ang in angles:
        ang_rad = np.deg2rad(ang)
        summed = np.zeros(frame_size)
        for mic, (x, y) in zip(signals, mic_positions):
            delay = (x*np.cos(ang_rad) + y*np.sin(ang_rad)) / speed_of_sound
            delayed_t = t - delay
            steering = np.sin(2*np.pi*f_source*delayed_t)
            summed += mic * steering
        power.append(np.sum(summed**2))
    return angles, 10*np.log10(power/np.max(power))

# === Setup plotting ===
fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot([], [], lw=2)
ax.set_xlim(-90, 90)
ax.set_ylim(-40, 0)
ax.set_title("Live Beamforming (Simulated)")
ax.set_xlabel("Angle (degrees)")
ax.set_ylabel("Relative Power (dB)")
ax.grid(True)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    signals = generate_frame()
    angles, power = beamform_frame(signals)
    line.set_data(angles, power)
    return line,

ani = animation.FuncAnimation(fig, update, init_func=init,
                              frames=200, interval=100, blit=True)

plt.show()
