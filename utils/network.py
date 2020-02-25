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
    def __init__(self, buses_in_service=3.0, relative_length=3.0, min_stop_time=15., stop_spacing=1. / 500.,
                 passenger_wait=5.):
        super().__init__("bus")
        self.relative_length = relative_length
        self.buses_in_service = buses_in_service
        self.min_stop_time = min_stop_time
        self.stop_spacing = stop_spacing
        self.passenger_wait = passenger_wait


class Mode:
    def __init__(self, network, name: str):
        self.name = name
        self.N_eq = 0.0
        self.L_blocked = 0.0
        self.network = network

    def updateBlockedDistance(self, v):
        self.L_blocked = 0.0


class BusMode(Mode):
    def __init__(self, network, busNetworkParams: BusModeParams) -> None:
        super().__init__(network, "bus")
        self.buses_in_service = busNetworkParams.buses_in_service
        self.relative_length = busNetworkParams.relative_length
        self.min_stop_time = busNetworkParams.min_stop_time
        self.stop_spacing = busNetworkParams.stop_spacing
        self.passenger_wait = busNetworkParams.passenger_wait
        self.trip_start_rate = 0.0
        self.trip_end_rate = 0.0

    def getFixedDensity(self):
        return self.fixed_density

    def getDwellTime(self, v):
        if v > 0:
            out = 1. / (self.stop_spacing * v) * (
                    v * self.fixed_density * self.min_stop_time * self.stop_spacing +
                    self.passenger_wait * 2 * (self.trip_start_rate + self.trip_end_rate)) / (
                          self.fixed_density - self.passenger_wait * (self.trip_start_rate + self.trip_end_rate))
        else:
            out = np.nan
        return out

    def updateBlockedDistance(self, v):
        dwellTime = self.getDwellTime(v)
        blocked_distance = self.network.l * self.passenger_wait * self.fixed_density * dwellTime * v / self.network.l
        self.L_blocked = blocked_distance

class Network:
    def __init__(self, L: float, networkFlowParams: NetworkFlowParams):
        self.L = L
        self.lam = networkFlowParams.lam
        self.u_f = networkFlowParams.u_f
        self.w = networkFlowParams.w
        self.kappa = networkFlowParams.kappa
        self.Q = networkFlowParams.Q
        self.l = networkFlowParams.l
        self.modes = dict()
        self.car_speed = networkFlowParams.u_f
        self.resetModes()

    def resetModes(self):
        for mode in self.modes.values():
            mode.N_eq = 0.0
            mode.L_blocked = 0.0

    def getBaseSpeed(self):
        return self.u_f

    def updateBlockedDistance(self):
        for mode in self.modes.values():
            #assert(isinstance(mode, Mode) | issubclass(mode, Mode))
            mode.updateBlockedDistance(self.car_speed)

    def MFD(self):
        L_eq = self.L - self.getBlockedDistance()
        N_eq = self.getN()
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
        return sum([mode.L_blocked for mode in self.modes.values()])

    def getN(self):
        return sum([mode.N_eq for mode in self.modes.values()])

    def addMode(self, modeParams: ModeParams):
        if isinstance(modeParams, BusModeParams):
            self.modes[modeParams.name] = BusMode(self, modeParams)
        else:
            self.modes[modeParams.name] = Mode(self, modeParams.name)
        return self


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
            n.updateBlockedDistance()
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
        self.N_eq = sum(n.modes[self.mode].N_eq for n in self._networks)

    def addModeVehicles(self, N_eq):
        speeds = []
        times = []
        lengths = []
        for n in self._networks:
            speeds.append(n.car_speed)
            times.append(n.L / n.car_speed)
            lengths.append(n.L)
        T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
        for ind, n in enumerate(self._networks):
            n.addDensity(self.mode, N_eq * lengths[ind] / speeds[ind] / T_tot)
        self.updateN()

    # def rebalanceVehicles(self):
    #     if self.mode == "car":


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
