import numpy as np


def linear_chirp(t, sweep_rate, start_frequency=0.0):
    phase = 2.0 * np.pi * start_frequency * t + np.pi * sweep_rate * t**2
    return np.exp(1j * phase)


def nonlinear_chirp(t, duration, sweep_span, nonlinearity=0.4, start_frequency=0.0):
    x = t / duration
    phase = 2.0 * np.pi * (start_frequency * t + sweep_span * duration * (0.5 * x**2 + nonlinearity * (x**3 / 3.0 - 0.5 * x**2)))
    return np.exp(1j * phase)


def custom_probe(t, waveform_function):
    return waveform_function(t)
