import numpy as np
from typing import List, Dict
from utils.supply import TravelDemand, TravelDemands


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
        self._N = dict()
        self._L_blocked = dict()
        self.relative_length = 1.0
        self._networks = networks
        self._averageDistanceInSystem = 0.0
        for n in networks:
            n.addMode(self)
            self._N[n] = 0.0
            self._L_blocked[n] = 0.0
        self.travelDemand = TravelDemand()
        self.densityFixed = False

    def updateBlockedDistance(self):
        for n in self._networks:
            self._L_blocked[n] = n.L_blocked[self.name]

    def addVehicles(self, n):
        self.allocateVehicles(n)

    def getTotalNumberOfVehicles(self):
        return sum([n for n in self._N.values()])

    def allocateVehicles(self, n_tot):
        "for constant car speed"
        current_allocation = []
        blocked_lengths = []
        lengths = []
        other_mode_n_eq = []
        for n in self._networks:
            other_modes = list(n.N_eq.keys())
            if self.name in other_modes:
                other_modes.remove(self.name)
            current_allocation.append(self._N[n])
            blocked_lengths.append(n.getBlockedDistance())
            lengths.append(n.L)
            other_mode_n_eq.append(sum([n.N_eq[m] for m in other_modes]))
        n_eq_other = sum(other_mode_n_eq)
        L_tot = sum(lengths)
        L_blocked_tot = sum(blocked_lengths)
        density_av = (n_tot + n_eq_other) / (L_tot - L_blocked_tot) * self.relative_length
        if n_tot > 0:
            n_new = [density_av * (lengths[i] - blocked_lengths[i]) - other_mode_n_eq[i] for i in range(len(lengths))]
        else:
            n_new = [0] * len(lengths)
        for ind, n in enumerate(self._networks):
            n.N_eq[self.name] = n_new[ind] * self.relative_length
            self._N[n] = n.N_eq[self.name]

    def __str__(self):
        return str([self.name + ': N=' + str(self._N) + ', L_blocked=' + str(self._L_blocked)])

    def getSpeed(self):
        return max(self._N, key=self._N.get).car_speed

    def getN(self, network):
        return self._N[network]

    def getLblocked(self, network):
        return self._L_blocked[network]

    def getLittlesLawN(self, rateOfPMT: float, averageDistanceInSystem: float):
        if self.densityFixed:
            return self.N_fixed
        else:
            speed = self.getSpeed()
            averageTimeInSystem = averageDistanceInSystem / speed
            return rateOfPMT / averageDistanceInSystem * averageTimeInSystem


class BusMode(Mode):
    def __init__(self, networks, busNetworkParams: BusModeParams) -> None:
        super().__init__(networks, "bus")
        self.N_fixed = busNetworkParams.buses_in_service
        self.relative_length = busNetworkParams.relative_length
        self.min_stop_time = busNetworkParams.min_stop_time
        self.stop_spacing = busNetworkParams.stop_spacing
        self.passenger_wait = busNetworkParams.passenger_wait
        self.fixed_density = self.getFixedDensity()
        self.densityFixed = True
        self.routeAveragedSpeed = super().getSpeed()

    def addVehicles(self, n: float):
        self.N_fixed = n
        self.allocateVehicles(n)

    def getRouteLength(self):
        return sum([n.L for n in self._networks])

    def getFixedDensity(self):
        return self.N_fixed / self.getRouteLength()

    def getSubNetworkSpeed(self, car_speed):
        averageStopDuration = self.min_stop_time + self.passenger_wait * (
                self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) / (
                                      self.routeAveragedSpeed / self.stop_spacing * self.N_fixed)
        return car_speed / (1 + averageStopDuration * car_speed / self.stop_spacing)

    def getSpeeds(self):
        speeds = []
        for n in self._networks:
            carSpeed = n.car_speed
            bus_speed = self.getSubNetworkSpeed(carSpeed)
            speeds.append(bus_speed)
        return speeds

    def getSpeed(self):
        meters = []
        seconds = []
        for n in self._networks:
            n_bus = self._N[n]
            bus_speed = self.getSubNetworkSpeed(n.car_speed)
            seconds.append(n_bus)
            meters.append(n_bus * bus_speed)
        if sum(seconds) == 0:
            return next(iter(self._networks)).car_speed
        spd = sum(meters) / sum(seconds)
        return spd

    def getBlockedDistance(self, network):
        if network.car_speed > 0:
            busSpeed = self.getSubNetworkSpeed(network.car_speed)
            out = busSpeed / self.stop_spacing * self.fixed_density * network.L * self.min_stop_time * network.l
        else:
            out = 0
        return out

    def updateBlockedDistance(self):
        for n in self._networks:
            assert (isinstance(n, Network))
            L_blocked = self.getBlockedDistance(n)
            n.L_blocked[self.name] = L_blocked * self.getRouteLength() / n.L

    def allocateVehicles(self, n_tot: float):
        "Poisson likelihood"
        speeds = self.getSpeeds()
        times = []
        lengths = []
        for ind, n in enumerate(self._networks):
            spd = speeds[ind]
            times.append(n.L / spd)
            lengths.append(n.L)
        T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
        for ind, n in enumerate(self._networks):
            n.N_eq[self.name] = n_tot * lengths[ind] / speeds[ind] / T_tot * self.relative_length
            self._N[n] = n.N_eq[self.name] * self.relative_length
        self.routeAveragedSpeed = self.getSpeed()


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
        self._modes = dict()
        self.car_speed = networkFlowParams.u_f

    def __str__(self):
        return str(list(self.N_eq.keys()))

    def resetModes(self):
        for mode in self._modes.values():
            self.N_eq[mode.name] = mode.getN(self) * mode.relative_length
            self.L_blocked[mode.name] = mode.getLblocked(self)

    def getBaseSpeed(self):
        return self.car_speed

    def updateBlockedDistance(self):
        for mode in self._modes.values():
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
        return mode in self._modes.keys()

    def addDensity(self, mode, N_eq):
        self._modes[mode].N_eq += N_eq

    def getBlockedDistance(self) -> float:
        if self.L_blocked:
            return sum(list(self.L_blocked.values()))
        else:
            return 0.0

    def getN_eq(self) -> float:
        if self.N_eq:
            return sum(list(self.N_eq.values()))
        else:
            return 0.0

    def addMode(self, mode: Mode):
        self._modes[mode.name] = mode
        return self

    def getModeNames(self) -> list:
        return list(self._modes.keys())

    def getModeValues(self) -> list:
        return list(self._modes.values())


class NetworkCollection:
    def __init__(self, network=None):
        self._networks = list()
        if isinstance(network, Network):
            self._networks.append(network)
        elif isinstance(network, List):
            self._networks = network
        self.modes = dict()
        self.demands = TravelDemands([])
        self.resetModes()

    def resetModes(self):
        allModes = [n.getModeValues() for n in self._networks]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        self.modes = dict()
        for m in uniqueModes:
            m.allocateVehicles(m.N_fixed)
            self.modes[m.name] = m
            self.demands[m.name] = m.travelDemand
        # self.updateNetworks()

    def updateModes(self, n: int = 10):
        allModes = [n.getModeValues() for n in self._networks]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        for it in range(n):
            for m in uniqueModes:
                mode_demand = self.demands[m.name]
                n_new = m.getLittlesLawN(mode_demand.rateOfPMT, mode_demand.averageDistanceInSystem)
                m.allocateVehicles(n_new)
            self.updateNetworks()
            self.updateMFD()
            print(str(self))

    def updateNetworks(self):
        for n in self._networks:
            n.resetModes()

    def append(self, network: Network):
        self._networks.append(network)
        self.resetModes()

    def __getitem__(self, item):
        return self.modes[item]

    def addVehicles(self, mode: str, N: float, n=5):
        self[mode].addVehicles(N)
        self.updateMFD(n)

    def updateMFD(self, n=1):
        for i in range(n):
            for n in self._networks:
                n.updateBlockedDistance()
                n.MFD()
            for m in self.modes.values():
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

    def getModeNames(self) -> list:
        return list(self.modes.keys())


def main():
    network_params_mixed = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_params_car = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_params_bus = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_car = Network(250, network_params_car)
    network_bus = Network(750, network_params_bus)
    network_mixed = Network(500, network_params_mixed)

    car = Mode([network_mixed, network_car], 'car')
    bus = BusMode([network_mixed, network_bus], BusModeParams(1.0))
    nc = NetworkCollection([network_mixed, network_car, network_bus])

    nc.addVehicles('car', 6.0)
    nc.updateMFD()
    print('DONE')


if __name__ == "__main__":
    main()
