import numpy as np


class Fiber:
    def __init__(self, section_lengths, z_sections, a_sections, beta, gamma, attenuation_db_per_km=0.0):
        self.section_lengths = section_lengths
        self.z_sections = z_sections
        self.a_sections = a_sections
        self.beta = beta
        self.gamma = gamma
        self.attenuation_db_per_km = attenuation_db_per_km

    def quasistatic_phase(self, t, section_idx, phases):
        phase = np.zeros_like(t, dtype=float)
        for m in range(section_idx):
            phase += 2.0 * phases[m](t)
        return phase

    def dynamic_phase(self, t, section_idx, z, phases):
        section_start = sum(self.section_lengths[:section_idx])
        scat_pos = section_start + z
        phase = np.zeros_like(t, dtype=float)
        for m in range(section_idx):
            stret_pos = sum(self.section_lengths[:m + 1])
            return_time = t - self.gamma * stret_pos
            forward_time = t - self.gamma * (2.0 * scat_pos - stret_pos)
            phase += phases[m](return_time)
            phase += phases[m](forward_time)
        return phase

    def round_trip_attenuation(self, distance):
        return 10.0 ** (-self.attenuation_db_per_km * (distance / 1000.0) / 10.0)

    def simulate(self, t, probe, phases, mode='quasistatic'):
        ref_sections = []
        dstb_sections = []

        for i in range(len(self.section_lengths)):
            section_start = sum(self.section_lengths[:i])
            ref = np.zeros_like(t, dtype=complex)
            dstb = np.zeros_like(t, dtype=complex)

            if mode == 'quasistatic':
                phase_per_section = self.quasistatic_phase(t, i, phases)

            for zk, ak in zip(self.z_sections[i], self.a_sections[i]):
                scat_pos = section_start + zk
                attenuation = self.round_trip_attenuation(scat_pos)
                ref_pt_backscat = attenuation * np.exp(-2j * self.beta * scat_pos) * probe(t - 2.0 * self.gamma * scat_pos)
                ref += ak * ref_pt_backscat

                if mode == 'dynamic':
                    phase_per_section = self.dynamic_phase(t, i, zk, phases)

                dstb_pt_backscat = np.exp(1j * phase_per_section) * ref_pt_backscat
                dstb += ak * dstb_pt_backscat

            ref_sections.append(ref)
            dstb_sections.append(dstb)

        b_ref = np.sum(ref_sections, axis=0)
        b_dstb = np.sum(dstb_sections, axis=0)
        return b_ref, b_dstb, ref_sections, dstb_sections
