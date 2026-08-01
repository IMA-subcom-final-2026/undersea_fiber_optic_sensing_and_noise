import numpy as np


class LinearChirpPhase:
    """Paper-style detector for a constant sweep rate only."""

    C0 = 299792458.0

    def homodyne(self, backscatter, local_oscillator):
        return backscatter * np.conj(local_oscillator)

    def distance_spectrum(self, signal, dt, sweep_rate, n_eff):
        spectrum = signal.size * np.fft.ifft(signal)
        beat_frequency = np.fft.fftfreq(signal.size, d=dt)
        distance = self.C0 * beat_frequency / (2.0 * n_eff * sweep_rate)
        return distance, spectrum

    def tukey_spatial_window(self, distance, center, length, alpha=0.4):
        x = (distance - center + length / 2.0) / length
        window = np.zeros_like(distance)

        left = (x >= 0.0) & (x < alpha / 2.0)
        middle = (x >= alpha / 2.0) & (x <= 1.0 - alpha / 2.0)
        right = (x > 1.0 - alpha / 2.0) & (x <= 1.0)

        window[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0)))
        window[middle] = 1.0
        window[right] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0)))
        return window

    def isolate_channel(self, spectrum, distance, center, length):
        window = self.tukey_spatial_window(distance, center, length)
        channel_spectrum = window * spectrum
        channel_trace = np.fft.fft(channel_spectrum) / spectrum.size
        return channel_trace

    def cumulative_phase(self, stressed_channel, reference_channel, phase_passes=2.0):
        ratio_phase = np.angle(stressed_channel / reference_channel)
        return np.unwrap(ratio_phase) / phase_passes

    def strongest_channel_center(self, spectrum, distance, section_start, section_end, window_length):
        margin = 1.5 * window_length
        candidates = np.linspace(section_start + margin, section_end - margin, 101)
        scores = []

        for center in candidates:
            trace = self.isolate_channel(spectrum, distance, center, window_length)
            scores.append(np.min(np.abs(trace)))

        return candidates[np.argmax(scores)]
