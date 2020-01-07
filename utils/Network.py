import numpy as np
from typing import List, Dict


class Network:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, network_length,
                 avg_link_length, modes: List[str]):
        self.lam = smoothing
        self.u_f = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.L = network_length
        self.l = avg_link_length
        self.modes = modes
        self.N_eq = dict()
        self.L_blocked = dict()
        self.car_speed = free_flow_speed
        self.resetModes()

    def resetModes(self):
        for mode in self.modes:
            self.N_eq[mode] = 0.0
            self.L_blocked[mode] = 0.0

    def getBaseSpeed(self):
        return self.u_f

    def MFD(self):
        L_eq = self.L - sum(self.L_blocked.values())
        N_eq = sum(self.N_eq.values())
        maxDensity = 0.25
        if (N_eq / L_eq < maxDensity) & (N_eq / L_eq >= 0.0):
            noCongestionN = (self.kappa * L_eq * self.w - L_eq * self.Q) / (self.u_f + self.w)
            if N_eq <= noCongestionN:
                peakFlowSpeed = - L_eq * self.lam / noCongestionN * np.log(
                    np.exp(- self.u_f * noCongestionN / (L_eq * self.lam)) +
                    np.exp(- self.Q / self.lam) +
                    np.exp(-(self.kappa - noCongestionN / L_eq) * self.w / self.lam))
                v = self.u_f - N_eq / noCongestionN * (self.u_f - peakFlowSpeed)
            else:
                v = - L_eq * self.lam / N_eq * np.log(
                    np.exp(- self.u_f * N_eq / (L_eq * self.lam)) +
                    np.exp(- self.Q / self.lam) +
                    np.exp(-(self.kappa - N_eq / L_eq) * self.w / self.lam))
            self.car_speed = np.maximum(v, 0.0)
        else:
            self.car_speed = np.nan

    def containsMode(self, mode: str) -> bool:
        return mode in self.modes

    def addDensity(self, mode, N_eq):
        self.N_eq[mode] += N_eq


class NetworkCollection:
    def __init__(self, network=None):
        self._networks = list()
        if isinstance(network, Network):
            self._networks.append(network)
        elif isinstance(network, List):
            self._networks = network

    def append(self, network: Network):
        self._networks.append(network)

    def __getitem__(self, item):
        networks = []
        for n in self._networks:
            if n.containsMode(item):
                networks.append(n)
        return OneModeNetworkCollection(networks, item)

    def addVehicles(self, mode, N_eq):
        self[mode].addModeVehicles(N_eq)
        self.updateMFD()

    def updateMFD(self):
        for n in self._networks:
            n.MFD()

    def __str__(self):
        return str([n.car_speed for n in self._networks])


class OneModeNetworkCollection(NetworkCollection):
    def __init__(self, networks: list, mode: str):
        super().__init__(networks)
        self.mode = mode
        self.N_eq = 0.0
        self.L = [n.L for n in self._networks]
        self.updateN()
        self.isAtEquilibrium = True

    def updateN(self):
        self.N_eq = sum(n.N_eq[self.mode] for n in self._networks)

    def addModeVehicles(self, N_eq):
        speeds = []
        times = []
        for n in self._networks:
            speeds.append(n.car_speed)
            times.append(n.L / n.car_speed)
        T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
        for ind, n in enumerate(self._networks):
            n.addDensity(self.mode, N_eq * lengths[ind] / speeds[ind] / T_tot)
        self.updateN()

    # def rebalanceVehicles(self):
    #     if self.mode == "car":



def main():
    network_mixed = Network(0.068, 15.42, 1.88, 0.145, 0.177, 250, 50, ['car', 'bus'])
    network_car = Network(0.068, 15.42, 1.88, 0.145, 0.177, 750, 50, ['car'])
    network_bus = Network(0.068, 15.42, 1.88, 0.145, 0.177, 500, 50, ['bus'])

    nc = NetworkCollection([network_mixed, network_car, network_bus])
    nc.updateMFD()
    nc.addVehicles('bus', 1.0)
    nc.addVehicles('car', 2.0)
    carnc = nc['car']
    busnc = nc['bus']
    nc.updateMFD()
    print('DONE')


if __name__ == "__main__":
    main()
