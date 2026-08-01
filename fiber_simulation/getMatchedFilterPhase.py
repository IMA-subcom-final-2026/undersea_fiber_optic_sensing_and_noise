import numpy as np


class MatchedFilterPhase:
    """Coherent matched filtering for a known linear, nonlinear or custom probe."""

    C0 = 299792458.0

    def homodyne(self, backscatter, local_oscillator):
        return backscatter * np.conj(local_oscillator)

    def beat_template(self, probe, t, distance, n_eff):
        delay = 2.0 * n_eff * distance / self.C0
        return probe(t - delay) * np.conj(probe(t))

    def matched_filter_profile(self, detected, t, distance, probe, n_eff):
        profile = np.zeros(distance.size, dtype=complex)
        for i, z in enumerate(distance):
            template = self.beat_template(probe, t, z, n_eff)
            profile[i] = np.vdot(template, detected) / np.vdot(template, template).real
        return profile

    def strongest_channel_center(self, profile, distance, section_start, section_end, margin):
        inside = (distance >= section_start + margin) & (distance <= section_end - margin)
        section_distance = distance[inside]
        section_profile = profile[inside]
        return section_distance[np.argmax(np.abs(section_profile))]

    def short_time_channels(self, detected, t, channel_centers, probe, n_eff, time_centers, window_duration):
        channels = np.zeros((len(channel_centers), len(time_centers)), dtype=complex)
        half_width = window_duration / 2.0

        for i, center in enumerate(channel_centers):
            template = self.beat_template(probe, t, center, n_eff)
            for j, time_center in enumerate(time_centers):
                inside = np.abs(t - time_center) <= half_width
                x = (t[inside] - time_center) / half_width
                weight = 0.5 * (1.0 + np.cos(np.pi * x))
                numerator = np.vdot(template[inside], weight * detected[inside])
                denominator = np.vdot(template[inside], weight * template[inside]).real
                channels[i, j] = numerator / denominator

        return channels

    def cumulative_phase(self, stressed_channels, reference_channels, phase_passes=2.0):
        ratio_phase = np.angle(stressed_channels / reference_channels)
        return np.unwrap(ratio_phase, axis=-1) / phase_passes

    def wiener_delay_profile(self, backscatter, probe_samples, dt, n_eff, profile_psd, noise_psd):
        # Static finite-record convolution b = probe * profile + noise; use raw complex fields.
        number = 2 ** int(np.ceil(np.log2(2 * backscatter.size - 1)))
        backscatter_spectrum = np.fft.fft(backscatter, number)
        probe_spectrum = np.fft.fft(probe_samples, number)
        wiener_filter = np.conj(probe_spectrum) * profile_psd / (np.abs(probe_spectrum)**2 * profile_psd + noise_psd)
        profile = np.fft.ifft(wiener_filter * backscatter_spectrum)
        delay = np.arange(backscatter.size) * dt
        distance = self.C0 * delay / (2.0 * n_eff)
        return distance, profile[:backscatter.size]
