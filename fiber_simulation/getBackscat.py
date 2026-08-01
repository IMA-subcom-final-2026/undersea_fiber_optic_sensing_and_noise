# Choose Rayleigh backscattering center distribution
# Choose Rayleigh backscattering coefficients distribution

import numpy as np


class Backscat:
    def __init__(self, Ncent, center_distri, Ncoeff, coeff_distri, center_rng, coeff_rng):
        self.Ncent = Ncent
        self.center_distri = center_distri
        self.Ncoeff = Ncoeff
        self.coeff_distri = coeff_distri
        self.center_rng = center_rng
        self.coeff_rng = coeff_rng

    def get_scat_pos(self, length):
        if self.center_distri == "grid":
            return (np.arange(self.Ncent) + 0.5) * length / self.Ncent

        accepted = np.empty(0)
        while accepted.size < self.Ncent:
            if self.center_distri == "uniform":
                candidates = self.center_rng.uniform(0.0, length, self.Ncent)
            elif self.center_distri == "gaussian":
                candidates = self.center_rng.normal(length / 2.0, length / 6.0, self.Ncent)
            else:
                candidates = self.center_rng.rayleigh(length / 3.0, self.Ncent)

            inside = candidates[(candidates >= 0.0) & (candidates <= length)]
            accepted = np.r_[accepted, inside]

        return np.sort(accepted[:self.Ncent])

    def get_scat_coeff(self, mean_power=1.0):
        if self.coeff_distri == "uniform":
            limit = np.sqrt(3.0 * mean_power / 2.0)
            return self.coeff_rng.uniform(-limit, limit, self.Ncoeff) + 1j * self.coeff_rng.uniform(-limit, limit, self.Ncoeff)

        if self.coeff_distri == "gaussian":
            return np.sqrt(mean_power / 2.0) * (self.coeff_rng.normal(size=self.Ncoeff) + 1j * self.coeff_rng.normal(size=self.Ncoeff))

        mag = self.coeff_rng.rayleigh(np.sqrt(mean_power / 2.0), self.Ncoeff)
        phase = self.coeff_rng.uniform(-np.pi, np.pi, self.Ncoeff)
        return mag * np.exp(1j * phase)

    def sum_scat(self, probe, t, z_scat, a_scat, beta0, beta1, attenuation_db_per_km=0.0):
        H = np.zeros(t.size, dtype=complex)
        for zk, ak in zip(z_scat, a_scat):
            attenuation = 10.0 ** (-attenuation_db_per_km * (zk / 1000.0) / 10.0)
            H = H + attenuation * ak * np.exp(-2j * beta0 * zk) * probe(t - 2.0 * beta1 * zk)
        return H

    def make_fiber(self, L, stre_positions, scat_per_meter, center_distri, coeff_distri, min_scat_per_section=20):
        interval = np.r_[0.0, stre_positions, L]  # shape: (1+Nstre+1, )
        l_list = np.diff(interval)                # shape: (Nstre+1, )
        z_sections = []
        a_sections = []

        for l in l_list:
            Nscat_per_section = max(min_scat_per_section, int(scat_per_meter * l))
            section = Backscat(Nscat_per_section,
                               center_distri,
                               Nscat_per_section,
                               coeff_distri,
                               self.center_rng,
                               self.coeff_rng,)
            z_per_section = section.get_scat_pos(l)
            a_per_section = section.get_scat_coeff()
            z_sections.append(z_per_section)
            a_sections.append(a_per_section)

        return l_list, z_sections, a_sections
