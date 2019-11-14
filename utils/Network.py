import numpy as np

class Network:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, network_length,
                 avg_link_length):
        self.lam = smoothing
        self.u_f = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.L = network_length
        self.l = avg_link_length

    def getBaseSpeed(self):
        return self.u_f

    def MFD(self, N_eq, L_eq):
        maxDensity = 0.25
        if (N_eq / L_eq < maxDensity) & (N_eq / L_eq > 0.0):
            noCongestionN = (self.kappa * L_eq * self.w - L_eq * self.Q) / (self.u_f + self.w)
            N_eq = np.clip(N_eq, noCongestionN, None)
            v = - L_eq * self.lam / N_eq * np.log(
                np.exp(- self.u_f * N_eq / (L_eq * self.lam)) +
                np.exp(- self.Q / self.lam) +
                np.exp(-(self.kappa - N_eq / L_eq) * self.w / self.lam))
            return np.maximum(v, 0.0)
        else:
            return np.nan