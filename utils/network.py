import os
from typing import List, Dict

import numpy as np
import pandas as pd

from utils.supply import TravelDemand, TravelDemands

np.seterr(all='ignore')


class NetworkFlowParams:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, avg_link_length):
        self.lam = smoothing
        self.u_f = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.l = avg_link_length


class ModeParams:
    def __init__(self, name: str, relative_length=1.0):
        self.name = name
        self.relative_length = relative_length


class AutoModeParams(ModeParams):
    def __init__(self):
        super().__init__("auto")


class BusModeParams(ModeParams):
    def __init__(self, headway_in_sec=600, relative_length=3.0, min_stop_time=5., stop_spacing=500.,
                 passenger_wait=5.):
        super().__init__("bus")
        self.relative_length = relative_length
        self.headway_in_sec = headway_in_sec
        self.min_stop_time = min_stop_time
        self.stop_spacing = stop_spacing
        self.passenger_wait = passenger_wait


class Costs:
    def __init__(self, per_meter=0.0, per_start=0.0, per_end=0.0, vott_multiplier=1.0):
        self.per_meter = per_meter
        self.per_start = per_start
        self.per_end = per_end
        self.vott_multiplier = vott_multiplier


class Mode:
    def __init__(self, networks: List, params: ModeParams):
        self.name = params.name
        self._N_tot = 0.0
        self._N = dict()
        self._L_blocked = dict()
        self.params = params
        self._networks = networks
        self._averagePassengerDistanceInSystem = 0.0
        self.costs = Costs(0.0, 0.0, 0.0, 1.0)
        for n in networks:
            n.addMode(self)
            self._N[n] = 0.0
            self._L_blocked[n] = 0.0
        self.travelDemand = TravelDemand()

    # def reset(self):
    #     for key, value in self._N.items():
    #         self._N[key] = self.fixed_density * key.L
    #         self._L_blocked[key] = 0.0

    # def getFixedDensity(self):
    #     return 0.0

    def updateModeBlockedDistance(self):
        for n in self._networks:
            self._L_blocked[n] = n.L_blocked[self.name]

    def addVehicles(self, n):
        self._N_tot += n
        self.allocateVehicles()

    def allocateVehicles(self):
        """for constant car speed"""
        current_allocation = []
        blocked_lengths = []
        lengths = []
        other_mode_n_eq = []
        jammed = []
        for n in self._networks:
            other_modes = list(n.N_eq.keys())
            if self.name in other_modes:
                other_modes.remove(self.name)
            current_allocation.append(self._N[n])
            blocked_lengths.append(n.getBlockedDistance())
            lengths.append(n.L)
            other_mode_n_eq.append(sum([n.N_eq[m] for m in other_modes]))
            jammed.append(n.isJammed)
        n_eq_other = sum(other_mode_n_eq)
        L_tot = sum(lengths)
        L_blocked_tot = sum(blocked_lengths)
        density_av = (self._N_tot + n_eq_other) / (L_tot - L_blocked_tot) * self.params.relative_length
        if self._N_tot > 0:
            n_new = np.nan_to_num(np.array(
                [density_av * (lengths[i] - blocked_lengths[i]) - other_mode_n_eq[i] for i in range(len(lengths))]))
        else:
            n_new = np.array([0.0] * len(lengths))
        should_be_empty = (n_new < 0) | np.array(jammed)
        to_reallocate = np.sum(n_new[should_be_empty])
        n_new[~should_be_empty] += to_reallocate * n_new[~should_be_empty] / np.sum(n_new[~should_be_empty])
        n_new[should_be_empty] = 0
        for ind, n in enumerate(self._networks):
            n.N_eq[self.name] = n_new[ind] * self.params.relative_length
            self._N[n] = n_new[ind]

    def __str__(self):
        return str([self.name + ': N=' + str(self._N) + ', L_blocked=' + str(self._L_blocked)])

    def getSpeed(self):
        return max(self._N, key=self._N.get).car_speed

    def getN(self, network):
        return self._N[network]

    def getNs(self, network):
        return list(self._N.values())

    def getBlockedDistance(self, network):
        return self._L_blocked[network]

    def updateN(self, demand: TravelDemand):
        n_new = self.getLittlesLawN(demand.rateOfPMT, demand.averageDistanceInSystem)
        self._N_tot = n_new
        self.allocateVehicles()

    def getLittlesLawN(self, rateOfPMT: float, averageDistanceInSystem: float):
        speed = self.getSpeed()
        averageTimeInSystem = averageDistanceInSystem / speed
        return rateOfPMT / averageDistanceInSystem * averageTimeInSystem

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self._networks]):
            return 0.0
        else:
            return self.travelDemand.rateOfPMT


class AutoMode(Mode):
    def __init__(self, networks, modeParams: ModeParams) -> None:
        assert (isinstance(modeParams, AutoModeParams))
        super().__init__(networks, modeParams)


class BusMode(Mode):
    def __init__(self, networks, busModeParams: ModeParams) -> None:
        assert (isinstance(busModeParams, BusModeParams))
        super().__init__(networks, busModeParams)
        self._params = busModeParams
        self.routeAveragedSpeed = super().getSpeed()
        self.addVehicles(self.getRouteLength() / self.routeAveragedSpeed / self._params.headway_in_sec)
        self.routeAveragedSpeed = self.getSpeed()
        self.occupancy = 0.0
        self.updateModeBlockedDistance()

    def updateN(self, demand: TravelDemand):
        n_new = self.getRouteLength() / self.routeAveragedSpeed / self._params.headway_in_sec
        self._N_tot = n_new
        self.allocateVehicles()

    def getRouteLength(self):
        return sum([n.L for n in self._networks])

    def getSubNetworkSpeed(self, car_speed):
        # averageStopDuration = self.min_stop_time + self.passenger_wait * (
        #         self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) / (
        #                               self.routeAveragedSpeed / self.stop_spacing * self.N_tot)
        # return car_speed / (1 + averageStopDuration * car_speed / self.stop_spacing)
        car_travel_time = self.getRouteLength() / car_speed
        passengers_per_stop = (
                                          self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) * self._params.headway_in_sec / 3600.
        stopping_time = self.getRouteLength() / self._params.stop_spacing * self._params.min_stop_time
        stopped_time = self._params.passenger_wait * passengers_per_stop + stopping_time
        spd = self.getRouteLength() * car_speed / (stopped_time * car_speed + self.getRouteLength())
        return spd

    def getSpeeds(self):
        speeds = []
        for n in self._networks:
            carSpeed = n.car_speed
            if np.isnan(carSpeed):
                print("AAAH")
            bus_speed = self.getSubNetworkSpeed(carSpeed)
            if np.isnan(bus_speed):
                print("AAAH")
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
        if sum(seconds) > 0:
            spd = sum(meters) / sum(seconds)
            return spd
        else:
            return next(iter(self._networks)).car_speed

    def calculateBlockedDistance(self, network) -> float:
        if network.car_speed > 0:
            out = network.l / (
                    self._params.min_stop_time + self._params.headway_in_sec * self._params.passenger_wait * (
                    self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) / (
                            self.getRouteLength() / self._params.stop_spacing)) / self._params.headway_in_sec
            # busSpeed = self.getSubNetworkSpeed(network.car_speed)
            # out = busSpeed / self.stop_spacing * self.N_tot * self.min_stop_time * network.l
        else:
            out = 0
        return out

    def updateModeBlockedDistance(self):
        for n in self._networks:
            assert (isinstance(n, Network))
            L = n.L
            L_blocked = self.calculateBlockedDistance(n)
            self._L_blocked[n] = L_blocked
            n.L_blocked[self.name] = L_blocked  # * self.getRouteLength() / n.L

    def allocateVehicles(self):
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
            if speeds[ind] > 0:
                n.N_eq[self.name] = self._N_tot * lengths[ind] / speeds[ind] / T_tot * self._params.relative_length
                self._N[n] = n.N_eq[self.name] / self._params.relative_length
            else:
                n.N_eq[self.name] = n_tot / lengths[ind] * self._params.relative_length
                self._N[n] = self._N_tot / lengths[ind]
        self.routeAveragedSpeed = self.getSpeed()
        self.occupancy = self.getOccupancy()
        if np.isnan(self.occupancy):
            print("AAAH")

    def getOccupancy(self) -> float:
        return self.travelDemand.averageDistanceInSystem / self.routeAveragedSpeed * self.travelDemand.tripStartRate / self._N_tot

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self._networks]):
            return 0.0
        elif self.occupancy > 100:
            return np.nan
        else:
            return self.travelDemand.rateOfPMT


class Network:
    def __init__(self, L: float, networkFlowParams: NetworkFlowParams):
        self.networkFlowParams = networkFlowParams
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
        self.isJammed = False

    def __str__(self):
        return str(list(self.N_eq.keys()))

    def resetAll(self):
        self.N_eq = dict()
        self.L_blocked = dict()
        self._modes = dict()
        self.car_speed = self.networkFlowParams.u_f
        self.isJammed = False

    def resetModes(self):
        for mode in self._modes.values():
            self.N_eq[mode.name] = mode.getN(self) * mode.params.relative_length
            self.L_blocked[mode.name] = mode.getBlockedDistance(self)
        self.isJammed = False
        self.car_speed = self.networkFlowParams.u_f
        # mode.reset()

    def getBaseSpeed(self):
        return self.car_speed

    def updateBlockedDistance(self):
        for mode in self._modes.values():
            # assert(isinstance(mode, Mode) | issubclass(mode, Mode))
            mode.updateModeBlockedDistance()

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
        self._modes[mode].addVehicles(N_eq)

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
        self.L_blocked[mode.name] = 0.0
        self.N_eq[mode.name] = 0.0
        return self

    def getModeNames(self) -> list:
        return list(self._modes.keys())

    def getModeValues(self) -> list:
        return list(self._modes.values())


class NetworkCollection:
    def __init__(self, networksAndModes=None, modeParams=None, verbose=True):
        self._networks = list()
        if isinstance(networksAndModes, Dict) and isinstance(modeParams, Dict):
            self.populateNetworksAndModes(networksAndModes, modeParams)
        self.modes = dict()
        self.demands = TravelDemands([])
        self.verbose = verbose
        self.resetModes()

    def populateNetworksAndModes(self, networksAndModes, modeParams):
        modeToNetwork = dict()
        if isinstance(networksAndModes, Dict):
            for (network, modeNames) in networksAndModes.items():
                assert (isinstance(network, Network))
                self._networks.append(network)
                for modeName in modeNames:
                    if modeName in modeToNetwork:
                        modeToNetwork[modeName].append(network)
                    else:
                        modeToNetwork[modeName] = [network]
        else:
            print('Bad NetworkCollection Input')
        for (modeName, networks) in modeToNetwork.items():
            assert (isinstance(modeName, str))
            assert (isinstance(networks, List))
            params = modeParams[modeName]
            if isinstance(params, BusModeParams):
                BusMode(networks, params)
            elif isinstance(params, ModeParams):
                Mode(networks, params)

    def isJammed(self):
        return np.any([n.isJammed for n in self._networks])

    def resetModes(self):
        allModes = [n.getModeValues() for n in self._networks]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        for n in self._networks:
            n.isJammed = False
        self.modes = dict()
        for m in uniqueModes:
            m.updateN(TravelDemand())
            self.modes[m.name] = m
            self.demands[m.name] = m.travelDemand
        # self.updateNetworks()

    def updateModes(self, n: int = 50):
        allModes = [n.getModeValues() for n in self._networks]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        oldSpeeds = self.getModeSpeeds()
        for it in range(n):
            for m in uniqueModes:
                m.updateN(self.demands[m.name])
            # self.updateNetworks()
            self.updateMFD()
            if self.verbose:
                print(str(self))
            if np.any([n.isJammed for n in self._networks]):
                break
            newSpeeds = self.getModeSpeeds()
            if np.sum(np.power(oldSpeeds - newSpeeds, 2)) < 0.000001:
                break
            else:
                oldSpeeds = newSpeeds

    def updateNetworks(self):
        for n in self._networks:
            n.resetModes()

    def append(self, network: Network):
        self._networks.append(network)
        self.resetModes()

    def __getitem__(self, item):
        return [n for n in self._networks if item in n.getModeNames()]

    # def addVehicles(self, mode: str, N: float, n=5):
    #     self[mode].addVehicles(N)
    #     self.updateMFD(n)

    def updateMFD(self, iters=1):
        for i in range(iters):
            for n in self._networks:
                n.updateBlockedDistance()
                n.MFD()
                if np.isnan(n.car_speed):
                    n.isJammed = True
            # for m in self.modes.values():
            #     m.updateN(self.demands[m.name])

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

    def getModeSpeeds(self) -> np.array:
        return np.array([m.getSpeed() for m in self.modes.values()])


class ModeParamFactory:
    def __init__(self, path: str):
        self.path = path
        self.modeParams = dict()
        self.readFiles()

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    def readFiles(self):
        (_, _, filenames) = next(os.walk(os.path.join(self.path, "modes")))
        for file in filenames:
            self.modeParams[file.split(".")[0]] = pd.read_csv(os.path.join(self.path, "modes", file))

    def get(self, modeName: str, microtypeID: str) -> (ModeParams, Costs):
        if modeName.lower() == "bus":
            data = self.modeParams["bus"]
            data = data.loc[data["MicrotypeID"] == microtypeID].iloc[0]
            costs = Costs(data.PerMileCost / 1609.34, data.PerStartCost, 0.0, 1.0)
            modeParams = BusModeParams(data.Headway, data.VehicleSize, 15., data.StopSpacing, 5.)
            return modeParams, costs
        elif modeName.lower() == "auto":
            data = self.modeParams["auto"]
            data = data.loc[data["MicrotypeID"] == microtypeID].iloc[0]
            return AutoModeParams(), Costs(data.PerMileCost, 0.0, data.PerEndCost, 1.0)
        else:
            return AutoModeParams, Costs()


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
