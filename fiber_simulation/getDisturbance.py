import numpy as np


def sinusoidal_phase(amplitude, frequency, initial_phase=0.0):
    def phase(t):
        return amplitude * np.sin(2.0 * np.pi * frequency * t + initial_phase)
    return phase


def gaussian_burst_phase(amplitude, frequency, center_time, width, initial_phase=0.0):
    def phase(t):
        envelope = np.exp(-0.5 * ((t - center_time) / width)**2)
        return amplitude * envelope * np.sin(2.0 * np.pi * frequency * (t - center_time) + initial_phase)
    return phase


def damped_ringdown_phase(amplitude, frequency, start_time, decay_time):
    def phase(t):
        elapsed = t - start_time
        return amplitude * (elapsed >= 0.0) * np.exp(-np.maximum(elapsed, 0.0) / decay_time) * np.sin(2.0 * np.pi * frequency * np.maximum(elapsed, 0.0))
    return phase


def custom_phase(phase_function):
    return phase_function
