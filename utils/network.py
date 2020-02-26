import numpy as np
from typing import List, Dict


class NetworkFlowParams:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, avg_link_length):
        self.lam = smoothing
        self.u_f = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.l = avg_link_length


class ModeParams:
    def __init__(self, name: str):
        self.name = name


class BusModeParams(ModeParams):
    def __init__(self, buses_in_service=3.0, relative_length=3.0, min_stop_time=15., stop_spacing=500.,
                 passenger_wait=5.):
        super().__init__("bus")
        self.relative_length = relative_length
        self.buses_in_service = buses_in_service
        self.min_stop_time = min_stop_time
        self.stop_spacing = stop_spacing
        self.passenger_wait = passenger_wait


class Mode:
    def __init__(self, networks: List, name: str):
        self.name = name
        self.N_fixed = 0.0
        self.N = dict()
        self.L_blocked = dict()
        self.relative_length = 1.0
        self.networks = networks
        for n in networks:
            n.addMode(self)
            self.N[n] = 0.0
            self.L_blocked[n] = 0.0

    def updateBlockedDistance(self):
        for n in self.networks:
            self.L_blocked[n] = n.L_blocked[self.name]

    def addVehicles(self, n):
        self.allocateVehicles(n)

    def getTotalNumberOfVehicles(self):
        return sum([n for n in self.N.values()])

    def allocateVehicles(self, n_tot):
        current_allocation = []
        speeds = []
        times = []
        lengths = []
        for n in self.networks:
            current_allocation.append(self.N[n])
            speeds.append(n.car_speed)
            times.append(n.L / n.car_speed)
            lengths.append(n.L)
        L_tot = sum(lengths)
        T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
        v_av = L_tot / T_tot
        n_new = [lengths[i] / speeds[i] / T_tot * n_tot for i in range(len(lengths))]
        for ind, n in enumerate(self.networks):
            n.N_eq[self.name] = n_new[ind]
            self.N[n] = n.N_eq[self.name]

    def __str__(self):
        return str([self.name + ': N=' + str(self.N) + ', L_blocked=' + str(self.L_blocked)])


class BusMode(Mode):
    def __init__(self, networks, busNetworkParams: BusModeParams) -> None:
        super().__init__(networks, "bus")
        self.N_fixed = busNetworkParams.buses_in_service
        self.relative_length = busNetworkParams.relative_length
        self.min_stop_time = busNetworkParams.min_stop_time
        self.stop_spacing = busNetworkParams.stop_spacing
        self.passenger_wait = busNetworkParams.passenger_wait
        self.trip_start_rate = 0.0
        self.trip_end_rate = 0.0
        self.fixed_density = self.getFixedDensity()

    def getRouteLength(self):
        return sum([n.L for n in self.networks])

    def getFixedDensity(self):
        return self.N_fixed / self.getRouteLength()

    def getSpeed(self, carSpeed):
        return carSpeed * (
                1 - self.passenger_wait * (self.trip_start_rate + self.trip_end_rate) / self.fixed_density) / (
                       1 + carSpeed / self.stop_spacing * self.min_stop_time)

    def getBlockedDistance(self, network):
        if network.car_speed > 0:
            busSpeed = self.getSpeed(network.car_speed)
            out = busSpeed / self.stop_spacing * self.fixed_density * network.L * self.min_stop_time * network.l
        else:
            out = 0
        return out

    def updateBlockedDistance(self):
        for n in self.networks:
            assert (isinstance(n, Network))
            L_blocked = self.getBlockedDistance(n)
            n.L_blocked[self.name] = L_blocked * self.getRouteLength() / n.L

    def allocateVehicles(self, n_tot):
        speeds = []
        times = []
        lengths = []
        for n in self.networks:
            spd = self.getSpeed(n.car_speed)
            speeds.append(spd)
            times.append(n.L / spd)
            lengths.append(n.L)
        T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
        for ind, n in enumerate(self.networks):
            n.N_eq[self.name] = n_tot * lengths[ind] / speeds[ind] / T_tot
            self.N[n] = n.N_eq[self.name]


class Network:
    def __init__(self, L: float, networkFlowParams: NetworkFlowParams):
        self.L = L
        self.lam = networkFlowParams.lam
        self.u_f = networkFlowParams.u_f
        self.w = networkFlowParams.w
        self.kappa = networkFlowParams.kappa
        self.Q = networkFlowParams.Q
        self.l = networkFlowParams.l
        self.N_eq = dict()
        self.L_blocked = dict()
        self.modes = dict()
        self.car_speed = networkFlowParams.u_f

    def __str__(self):
        return str(list(self.N_eq.keys()))

    def resetModes(self):
        for mode in self.modes.values():
            self.N_eq[mode.name] = mode.N[self] * mode.relative_length
            self.L_blocked[mode.name] = mode.L_blocked[self]

    def getBaseSpeed(self):
        return self.u_f

    def updateBlockedDistance(self):
        for mode in self.modes.values():
            # assert(isinstance(mode, Mode) | issubclass(mode, Mode))
            mode.updateBlockedDistance()

    def MFD(self):
        L_eq = self.L - self.getBlockedDistance()
        N_eq = self.getN_eq()
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
        return mode in self.modes.keys()

    def addDensity(self, mode, N_eq):
        self.modes[mode].N_eq += N_eq

    def getBlockedDistance(self):
        if self.L_blocked:
            return sum(list(self.L_blocked.values()))
        else:
            return 0.0

    def getN_eq(self):
        if self.N_eq:
            return sum(list(self.N_eq.values()))
        else:
            return 0.0

    def addMode(self, mode: Mode):
        self.modes[mode.name] = mode
        return self


class NetworkCollection:
    def __init__(self, network=None):
        self._networks = list()
        if isinstance(network, Network):
            self._networks.append(network)
        elif isinstance(network, List):
            self._networks = network
        self._modes = dict()
        self.updateModes()

    def updateModes(self):
        allModes = [list(n.modes.values()) for n in self._networks]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        self._modes = dict()
        for m in uniqueModes:
            m.allocateVehicles(m.N_fixed)
            self._modes[m.name] = m
        self.updateNetworks()

    def updateNetworks(self):
        for n in self._networks:
            n.resetModes()

    def append(self, network: Network):
        self._networks.append(network)
        self.updateModes()

    def __getitem__(self, item):
        return self._modes[item]

    def addVehicles(self, mode, N):
        self[mode].addVehicles(N)
        self.updateMFD()

    def updateMFD(self):
        for n in self._networks:
            n.updateBlockedDistance()
            n.MFD()
        for m in self._modes.values():
            m.allocateVehicles(m.getTotalNumberOfVehicles())

    def __str__(self):
        return str([n.car_speed for n in self._networks])

    def addMode(self, networks: list, mode: Mode):
        for network in networks:
            assert (isinstance(network, Network))
            if network not in self._networks:
                self.append(network)
            self.modes[mode.name] = mode
        return self


def main():
    network_params_mixed = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_params_car = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_params_bus = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_car = Network(250, network_params_car)
    network_car.addMode(ModeParams("car"))
    network_bus = Network(750, network_params_bus)
    network_bus.addMode(BusModeParams())
    network_mixed = Network(500, network_params_mixed)
    network_mixed.addMode(BusModeParams())
    network_mixed.addMode(ModeParams("car"))

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
