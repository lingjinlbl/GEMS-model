from math import sqrt
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.supply import TravelDemand, TravelDemands

np.seterr(all='ignore')


class TotalOperatorCosts:
    def __init__(self):
        self.__costs = dict()
        self.__revenues = dict()
        self.__net = dict()

    def __setitem__(self, key: str, value: (float, float)):
        self.__costs[key] = value[0]
        self.__revenues[key] = value[1]
        self.__net[key] = value[0] - value[1]

    def __getitem__(self, item) -> float:
        return self.__net[item]

    def __iter__(self):
        return iter(self.__net.items())

    def __mul__(self, other):
        output = TotalOperatorCosts()
        for key in self.__costs.keys():
            output[key] = (self.__costs[key] * other, self.__revenues[key] * other)
        return output

    def __add__(self, other):
        output = TotalOperatorCosts()
        for key in self.__costs.keys():
            if key in self.__costs:
                output[key] = (self.__costs[key] + other.__costs[key], self.__revenues[key] + other.__revenues[key])
            else:
                output[key] = (other.__costs[key], other.__revenues[key])
        return output

    def __str__(self):
        return [key + ' ' + str(item) for key, item in self.__costs.items()]


class NetworkFlowParams:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, avg_link_length):
        self.lam = smoothing
        self.freeFlowSpeed = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.avgLinkLength = avg_link_length


class Costs:
    def __init__(self, per_meter=0.0, per_start=0.0, per_end=0.0, vott_multiplier=1.0):
        self.perMeter = per_meter
        self.perStart = per_start
        self.perEnd = per_end
        self.vottMultiplier = vott_multiplier


class Mode:
    def __init__(self, networks=None, params=None, idx=None, name=None):
        self.name = name
        self.params = params
        self._idx = idx
        # self._inds = self.initInds(idx)
        self.networks = networks
        # self._N_tot = 0.0
        # self._N = dict()
        self._L_blocked = dict()
        self._averagePassengerDistanceInSystem = 0.0
        self._VMT_tot = 0.0
        self._VMT = dict()
        self._speed = dict()
        self.__bad = False
        if networks is not None:
            for n in networks:
                n.addMode(self)
                # self._N[n] = 0.0
                self._L_blocked[n] = 0.0
                self._VMT[n] = 0.0
                self._speed[n] = n.base_speed
        self.travelDemand = TravelDemand()

    # def initInds(self, idx):
    #     inds = dict()
    #     for column in self.params.columns():
    #         inds[column] = [(self.params.index.get_loc(idx), self.params.columns.get_loc(column))]
    #     return inds

    @property
    def relativeLength(self):
        # return self.params.to_numpy()[self._inds["VehicleSize"]]
        return self.params.at[self._idx, "VehicleSize"]

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self.params.at[self._idx, "PerStartCost"]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self.params.at[self._idx, "PerEndCost"]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self._idx, "PerMileCost"]

    def updateDemand(self, travelDemand: TravelDemand):
        self.travelDemand = travelDemand
        self._VMT_tot = travelDemand.rateOfPmtPerHour * self.relativeLength

    def getDemandForVmtPerHour(self):
        return self.travelDemand.rateOfPmtPerHour * self.relativeLength

    def getAccessDistance(self) -> float:
        return 0.0

    def updateModeBlockedDistance(self):
        for n in self.networks:
            self._L_blocked[n] = n.L_blocked[self.name]

    # def addVehicles(self, n):
    #     self._N_tot += n
    #     self.allocateVehicles()

    def getSpeedDifference(self, allocation: list):
        speeds = np.array([n.NEF(a * self._VMT_tot, self.name) for n, a in zip(self.networks, allocation)])
        return np.linalg.norm(speeds - np.mean(speeds))

    def assignVmtToNetworks(self):
        Ltot = sum([n.L for n in self.networks])
        for n in self.networks:
            VMT = self._VMT_tot * n.L / Ltot
            self._VMT[n] = VMT
            n.setVMT(self.name, self._VMT[n])
            self._speed[n] = n.NEF(VMT, self.name)

    # def allocateVehicles(self):
    #     """even"""
    #     n_networks = len(self.networks)
    #     for n in self.networks:
    #         n.N_eq[self.name] = self._N_tot / n_networks * self.relativeLength
    #         self._N[n] = self._N_tot / n_networks

    def __str__(self):
        return str([self.name + ': VMT=' + str(self._VMT_tot) + ', L_blocked=' + str(self._L_blocked)])

    def getSpeed(self):
        return self.networks[0].getBaseSpeed()

    def getN(self, network):
        return self._VMT[network] / self._speed[network] / self.relativeLength

    def getNs(self):
        return [self.getN(n) for n in self.networks]

    def getBlockedDistance(self, network):
        return self._L_blocked[network]

    # def updateN(self, demand: TravelDemand):
    #     n_new = self.getLittlesLawN(demand.rateOfPmtPerHour, demand.averageDistanceInSystemInMiles)
    #     self._N_tot = n_new
    #     self.allocateVehicles()

    # def getLittlesLawN(self, rateOfPmtPerHour: float, averageDistanceInSystemInMiles: float):
    #     speedInMilesPerHour = self.getSpeed() * 2.23694
    #     if not (speedInMilesPerHour >= 2.0):
    #         self.__bad = True
    #         speedInMilesPerHour = 2.0
    #     else:
    #         self.__bad = False
    #     return rateOfPmtPerHour / speedInMilesPerHour

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self.networks]):
            return 0.0
        else:
            return self.travelDemand.rateOfPmtPerHour

    def getOperatorCosts(self) -> float:
        return 0.0

    def getOperatorRevenues(self) -> float:
        return 0.0

    def getPortionDedicated(self) -> float:
        return 0.0


class WalkMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super(WalkMode, self).__init__()
        self.name = "walk"
        self.params = modeParams
        self._idx = idx
        # self._inds = self.initInds(idx)
        self.networks = networks
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._speed[n] = n.base_speed

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self.params.at[self._idx, "SpeedInMetersPerSecond"]

    def getSpeed(self):
        return self.speedInMetersPerSecond


class BikeMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super(BikeMode, self).__init__()
        self.name = "bike"
        self.params = modeParams
        self._idx = idx
        # self._inds = self.initInds(idx)
        self.networks = networks
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._speed[n] = n.base_speed
        self.bikeLanePreference = 2.0

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self.params.at[self._idx, "SpeedInMetersPerSecond"]

    def getSpeed(self):
        return self.speedInMetersPerSecond

    # def allocateVehicles(self):
    #     """by length"""
    #     L_tot = sum([(n.L + n.L * (self.bikeLanePreference - 1) * n.dedicated) for n in self.networks])
    #     for n in self.networks:
    #         n.N_eq[self.name] = (n.L + n.L * (
    #                 self.bikeLanePreference - 1) * n.dedicated) * self._N_tot / L_tot * self.relativeLength
    #         self._N[n] = (n.L + n.L * (self.bikeLanePreference - 1) * n.dedicated) * self._N_tot / L_tot

    def getPortionDedicated(self) -> float:
        if self._VMT_tot > 0:
            tot = 0.0
            tot_dedicated = 0.0
            for key, val in self._VMT.items():
                tot += val
                if key.dedicated:
                    tot_dedicated += val
            return tot_dedicated / tot
        else:
            return 0.0


class RailMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super(RailMode, self).__init__()
        self.name = "rail"
        self.params = modeParams
        self._idx = idx
        self.networks = networks
        # self.initInds(idx)
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._speed[n] = n.base_speed

    @property
    def routeAveragedSpeed(self):
        return self.params.at[self._idx, "SpeedInMetersPerSecond"]

    @property
    def vehicleOperatingCostPerHour(self):
        # return self.params.to_numpy()[self._inds["VehicleOperatingCostsPerHour"]]
        return self.params.at[self._idx, "VehicleOperatingCostsPerHour"]

    @property
    def fare(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self.params.at[self._idx, "PerStartCost"]

    @property
    def headwayInSec(self):
        # return self.params.to_numpy()[self._inds["Headway"]]
        return self.params.at[self._idx, "Headway"]

    @property
    def stopSpacingInMeters(self):
        # return self.params.to_numpy()[self._inds["StopSpacing"]]
        return self.params.at[self._idx, "StopSpacing"]

    @property
    def portionAreaCovered(self):
        # return self.params.to_numpy()[self._inds["CoveragePortion"]]
        return self.params.at[self._idx, "CoveragePortion"]

    def updateDemand(self, travelDemand: TravelDemand):
        self.travelDemand = travelDemand
        self._VMT_tot = self.getRouteLength() / self.headwayInSec

    def getAccessDistance(self) -> float:
        return self.stopSpacingInMeters / 4.0 / self.portionAreaCovered ** 2.0

    def getSpeed(self):
        return self.routeAveragedSpeed

    def getRouteLength(self):
        return sum([n.L for n in self.networks])

    def getOperatorCosts(self) -> float:
        return sum(self.getNs()) * self.vehicleOperatingCostPerHour

    def getOperatorRevenues(self) -> float:
        return self.travelDemand.tripStartRatePerHour * self.fare

    def getDemandForVmtPerHour(self):
        return self.getRouteLength() / self.headwayInSec * 3600.

    def updateN(self, demand: TravelDemand):
        n_new = self.getRouteLength() / self.routeAveragedSpeed / self.headwayInSec
        self._N_tot = n_new
        self.allocateVehicles()

    def allocateVehicles(self):
        """ Assumes just one subNetwork """
        n = self.networks[0]
        n.N_eq[self.name] = self._N_tot
        self._N[n] = n.N_eq[self.name]


class AutoMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super(AutoMode, self).__init__()
        self.name = "auto"
        self.params = modeParams
        self._idx = idx
        self.networks = networks
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._speed[n] = n.base_speed

    @property
    def relativeLength(self):
        return self.params.at[self._idx, "VehicleSize"]

    def x0(self):
        return np.array([1. / len(self.networks)] * len(self.networks))

    def constraints(self):
        return dict(type='eq', fun=lambda x: sum(x) - 1.0, jac=lambda x: [1.0] * len(x))

    def bounds(self):
        return [(0.0, 1.0)] * len(self.networks)

    def assignVmtToNetworks(self):
        res = minimize(self.getSpeedDifference, self.x0(), constraints=self.constraints(), bounds=self.bounds())
        for n, a in zip(self.networks, res.x):
            self._VMT[n] = a * self._VMT_tot
            self._speed[n] = n.NEF(a * self._VMT_tot, self.name)
            n.setVMT(self.name, self._VMT[n])

    # def allocateVehicles(self):
    #     """for constant car speed"""
    #     current_allocation = []
    #     blocked_lengths = []
    #     lengths = []
    #     other_mode_n_eq = []
    #     jammed = []
    #     for n in self.networks:
    #         other_modes = list(n.N_eq.keys())
    #         if self.name in other_modes:
    #             other_modes.remove(self.name)
    #         current_allocation.append(self._N[n])
    #         blocked_lengths.append(n.getBlockedDistance())
    #         lengths.append(n.L)
    #         other_mode_n_eq.append(sum([n.N_eq[m] for m in other_modes]))
    #         jammed.append(n.isJammed)
    #     n_eq_other = sum(other_mode_n_eq)
    #     L_tot = sum(lengths)
    #     L_blocked_tot = sum(blocked_lengths)
    #     density_av = (self._N_tot + n_eq_other) / (L_tot - L_blocked_tot) * self.relativeLength
    #     if any(jammed):
    #         print(density_av)
    #     if self._N_tot > 0:
    #         n_new = np.nan_to_num(np.array(
    #             [density_av * (lengths[i] - blocked_lengths[i]) - other_mode_n_eq[i] for i in range(len(lengths))]))
    #     else:
    #         n_new = np.array([0.0] * len(lengths))
    #     should_be_empty = (n_new < 0) | np.array(jammed)
    #     to_reallocate = np.sum(n_new[should_be_empty])
    #     n_new[~should_be_empty] += to_reallocate * n_new[~should_be_empty] / np.sum(n_new[~should_be_empty])
    #     n_new[should_be_empty] = 0
    #     for ind, n in enumerate(self.networks):
    #         n.N_eq[self.name] = n_new[ind] * self.relativeLength
    #         self._N[n] = n_new[ind]


class BusMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super(BusMode, self).__init__()
        self.name = "bus"
        self.params = modeParams
        self._idx = idx
        self.networks = networks
        for n in networks:
            n.addMode(self)
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._speed[n] = n.base_speed
        self.routeAveragedSpeed = super().getSpeed()
        self.routeLength = self.getRouteLength()
        self.travelDemand = TravelDemand()
        self.routeAveragedSpeed = self.getSpeed()
        self.occupancy = 0.0
        self.updateModeBlockedDistance()

    @property
    def headwayInSec(self):
        return self.params.at[self._idx, "Headway"]

    @property
    def passengerWaitInSec(self):
        return self.params.at[self._idx, "PassengerWait"]

    @property
    def passengerWaitInSecDedicated(self):
        return self.params.at[self._idx, "PassengerWaitDedicated"]

    @property
    def stopSpacingInMeters(self):
        return self.params.at[self._idx, "StopSpacing"]

    @property
    def minStopTimeInSec(self):
        return self.params.at[self._idx, "MinStopTime"]

    @property
    def fare(self):
        return self.params.at[self._idx, "PerStartCost"]

    @property
    def vehicleOperatingCostPerHour(self):
        return self.params.at[self._idx, "VehicleOperatingCostPerHour"]

    @property
    def portionAreaCovered(self):
        return self.params.at[self._idx, "CoveragePortion"]

    def updateDemand(self, travelDemand: TravelDemand):
        self.travelDemand = travelDemand
        self._VMT_tot = self.getRouteLength() / self.headwayInSec

    def getAccessDistance(self) -> float:
        return self.stopSpacingInMeters / 4.0 / self.portionAreaCovered ** 2.0

    def getDemandForVmtPerHour(self):
        return self.getRouteLength() / self.headwayInSec * 3600.

    def getN(self, network):
        return network.L / self.routeAveragedSpeed / self.headwayInSec

    def getRouteLength(self):
        return sum([n.L for n in self.networks])

    def getSubNetworkSpeed(self, network):
        # averageStopDuration = self.min_stop_time + self.passenger_wait * (
        #         self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) / (
        #                               self.routeAveragedSpeed / self.stop_spacing * self.N_tot)
        # return base_speed / (1 + averageStopDuration * base_speed / self.stop_spacing)
        bs = network.base_speed
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        numberOfStopsInSubnetwork = network.L / self.stopSpacingInMeters
        numberOfStopsInRoute = self.getRouteLength() / self.stopSpacingInMeters
        pass_per_stop = (self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour
                         ) / numberOfStopsInRoute * self.headwayInSec / 3600.
        stopping_time = numberOfStopsInSubnetwork * self.minStopTimeInSec
        stopped_time = perPassenger * pass_per_stop * numberOfStopsInSubnetwork + stopping_time
        driving_time = network.L / network.base_speed
        spd = network.L / (stopped_time + driving_time)
        if np.isnan(spd):
            spd = 0.1
            self.__bad = True
        else:
            self.__bad = False
        return spd

    def getSpeeds(self):
        speeds = []
        for n in self.networks:
            if n.L == 0:
                speeds.append(np.inf)
            else:
                bus_speed = self.getSubNetworkSpeed(n)
                speeds.append(bus_speed)
        return speeds

    def getSpeed(self):
        meters = np.zeros(len(self.networks), dtype=float)
        seconds = np.zeros(len(self.networks), dtype=float)
        # meters = []
        # seconds = []
        for idx, n in enumerate(self.networks):
            if n.L > 0:
                n_bus = self.getN(n)
                bus_speed = self.getSubNetworkSpeed(n)
                seconds[idx] = n_bus
                meters[idx] = n_bus * bus_speed
        if np.sum(seconds) > 0:
            spd = np.sum(meters) / np.sum(seconds)
            return spd
        else:
            return next(iter(self.networks)).getBaseSpeed()

    def calculateBlockedDistance(self, network) -> float:
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        bs = network.base_speed
        if network.base_speed > 0:
            numberOfStops = self.routeLength / self.stopSpacingInMeters
            meanTimePerStop = (self.minStopTimeInSec + self.headwayInSec * perPassenger * (
                    self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour) / (
                                       numberOfStops * 3600.0))
            portionOfTimeStopped = min([meanTimePerStop * meanTimePerStop / self.headwayInSec, 1.0])
            out = network.avgLinkLength * portionOfTimeStopped
            portionOfRouteBlocked = out / self.routeLength
            # busSpeed = self.getSubNetworkSpeed(network.base_speed)
            # out = busSpeed / self.stop_spacing * self.N_tot * self.min_stop_time * network.l
        else:
            out = 0
        return out

    def updateModeBlockedDistance(self):
        for n in self.networks:
            L_blocked = self.calculateBlockedDistance(n)
            self._L_blocked[n] = L_blocked
            n.L_blocked[self.name] = L_blocked  # * self.getRouteLength() / n.L

    def assignVmtToNetworks(self):
        speeds = self.getSpeeds()
        times = []
        lengths = []
        for ind, n in enumerate(self.networks):
            spd = speeds[ind]
            times.append(n.L / spd)
            lengths.append(n.L)
        T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
        for ind, n in enumerate(self.networks):
            assert isinstance(n, Network)
            if speeds[ind] > 0:
                VMT = self._VMT_tot * lengths[ind] / speeds[ind] / T_tot
                self._VMT[n] = VMT
                n.setVMT(self.name, self._VMT[n])
                n.updateBaseSpeed()
                self._speed[n] = self.getSubNetworkSpeed(n)
            else:
                print("BAD WHY IS THIS SPEED NEGATIVE")
        self.updateCommercialSpeed()


    def updateCommercialSpeed(self):
        self.routeAveragedSpeed = self.getRouteLength() / sum([n.L / spd for n, spd in self._speed.items()])

    # def allocateVehicles(self):
    #     "Poisson likelihood"
    #     speeds = self.getSpeeds()
    #     times = []
    #     lengths = []
    #     for ind, n in enumerate(self.networks):
    #         spd = speeds[ind]
    #         times.append(n.L / spd)
    #         lengths.append(n.L)
    #     T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
    #     for ind, n in enumerate(self.networks):
    #         if speeds[ind] > 0:
    #             n_new = self._N_tot * lengths[ind] / speeds[ind] / T_tot
    #             n.N_eq[self.name] = n_new * self.relativeLength
    #             self._N[n] = n_new
    #         else:
    #             n_new = self._N_tot / lengths[ind]
    #             n.N_eq[self.name] = n_new * self.relativeLength
    #             self._N[n] = n_new
    #     self.routeAveragedSpeed = self.getSpeed()
    #     self.occupancy = self.getOccupancy()
    #     # if np.isnan(self.occupancy):
    #     #     print("AAAH")

    def getOccupancy(self) -> float:
        return self.travelDemand.averageDistanceInSystemInMiles / (
                self.routeAveragedSpeed * 2.23694) * self.travelDemand.tripStartRatePerHour / self.getN()

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self.networks]):
            return 0.0
        elif self.occupancy > 100:
            return np.nan
        else:
            return self.travelDemand.rateOfPmtPerHour

    def getOperatorCosts(self) -> float:
        return sum(self.getNs()) * self.vehicleOperatingCostPerHour

    def getOperatorRevenues(self) -> float:
        return self.travelDemand.tripStartRatePerHour * self.fare


class Network:
    def __init__(self, data, idx):
        self.data = data
        self._idx = idx
        self.L_blocked = dict()
        self._modes = dict()
        self.base_speed = self.freeFlowSpeed
        self.dedicated = data.loc[idx, "Dedicated"]
        self.isJammed = False
        self._VMT = dict()

    @property
    def type(self):
        return self.data.at[self._idx, "Type"]

    @property
    def avgLinkLength(self):
        return self.data.at[self._idx, "avgLinkLength"]

    @property
    def freeFlowSpeed(self):
        return self.data.at[self._idx, "vMax"]

    @property
    def jamDensity(self):
        return self.data.at[self._idx, "densityMax"]

    @property
    def L(self):
        return self.data.at[self._idx, "Length"]

    def __str__(self):
        return str(list(self._VMT.keys()))

    def resetAll(self):
        self.L_blocked = dict()
        self._modes = dict()
        self.base_speed = self.freeFlowSpeed
        self.isJammed = False

    def resetModes(self):
        for mode in self._modes.values():
            # self.N_eq[mode.name] = mode.getN(self) * mode.params.relativeLength
            self._VMT[mode] = mode._VMT[self]
            self.L_blocked[mode.name] = mode.getBlockedDistance(self)
        self.isJammed = False
        self.base_speed = self.freeFlowSpeed
        # mode.reset()

    def setVMT(self, mode: str, VMT: float):
        self._VMT[mode] = VMT

    def updateBaseSpeed(self):
        self.base_speed = self.NEF()

    def NEF(self, Q=None, modeIgnored=None) -> float:
        if Q is None:
            Qtot = sum([VMT for VMT in self._VMT.values()])
        else:
            Qtot = Q
            for mode, Qmode in self._VMT.items():
                if mode != modeIgnored:
                    Qtot += Qmode
        q = Qtot / (self.L - self.getBlockedDistance())
        qMax = self.freeFlowSpeed * self.jamDensity / 2.0
        if q <= qMax:
            return 0.5 * self.freeFlowSpeed * (sqrt(1 - q / qMax) + 1)
        else:
            return max([0.5 * self.freeFlowSpeed * (1 - sqrt(1 + q / qMax)), 0.1])

    def getBaseSpeed(self):
        if self.base_speed > 0.01:
            return self.base_speed
        else:
            return 0.01

    def updateBlockedDistance(self):
        for mode in self._modes.values():
            # assert(isinstance(mode, Mode) | issubclass(mode, Mode))
            mode.updateModeBlockedDistance()

    # def MFD(self):
    #     if self.L <= 0:
    #         self.base_speed = 1.0
    #         return
    #     L_eq = self.L - self.getBlockedDistance()
    #     N_eq = self.getN_eq()
    #     maxDensity = 0.25
    #     if (N_eq / L_eq < maxDensity) & (N_eq / L_eq >= 0.0):
    #         noCongestionN = (self.kappa * L_eq * self.w - L_eq * self.Q) / (self.u_f + self.w)
    #         if N_eq <= noCongestionN:
    #             peakFlowSpeed = - L_eq * self.lam / noCongestionN * np.log(
    #                 np.exp(- self.u_f * noCongestionN / (L_eq * self.lam)) +
    #                 np.exp(- self.Q / self.lam) +
    #                 np.exp(-(self.kappa - noCongestionN / L_eq) * self.w / self.lam))
    #             v = self.u_f - N_eq / noCongestionN * (self.u_f - peakFlowSpeed)
    #         else:
    #             v = - L_eq * self.lam / N_eq * np.log(
    #                 np.exp(- self.u_f * N_eq / (L_eq * self.lam)) +
    #                 np.exp(- self.Q / self.lam) +
    #                 np.exp(-(self.kappa - N_eq / L_eq) * self.w / self.lam))
    #         self.base_speed = np.maximum(v, 0.25)
    #     else:
    #         self.base_speed = np.nan

    def containsMode(self, mode: str) -> bool:
        return mode in self._modes.keys()

    # def addDensity(self, mode, N_eq):
    #     self._modes[mode].addVehicles(N_eq)

    def getBlockedDistance(self) -> float:
        if self.L_blocked:
            return sum(list(self.L_blocked.values()))
        else:
            return 0.0

    # def getN_eq(self) -> float:
    #     if self.N_eq:
    #         return sum(list(self.N_eq.values()))
    #     else:
    #         return 0.0

    def addMode(self, mode: Mode):
        self._modes[mode.name] = mode
        self.L_blocked[mode.name] = 0.0
        self._VMT[mode.name] = 0.0
        return self

    def getModeNames(self) -> list:
        return list(self._modes.keys())

    def getModeValues(self) -> list:
        return list(self._modes.values())


class NetworkCollection:
    def __init__(self, networksAndModes=None, modeToModeData=None, microtypeID=None, verbose=False):
        self._networks = list()
        self.modeToNetwork = dict()
        if isinstance(networksAndModes, Dict) and isinstance(modeToModeData, Dict):
            self.populateNetworksAndModes(networksAndModes, modeToModeData, microtypeID)
        self.modes = dict()

        self.demands = TravelDemands([])
        self.verbose = verbose
        # self.resetModes()

    def populateNetworksAndModes(self, networksAndModes, modeToModeData, microtypeID):
        # modeToNetwork = dict()
        if isinstance(networksAndModes, Dict):
            for (network, modeNames) in networksAndModes.items():
                assert (isinstance(network, Network))
                self._networks.append(network)
                for modeName in modeNames:
                    if modeName in self.modeToNetwork:
                        self.modeToNetwork[modeName].append(network)
                    else:
                        self.modeToNetwork[modeName] = [network]

        else:
            print('Bad NetworkCollection Input')
        for (modeName, networks) in self.modeToNetwork.items():
            assert (isinstance(modeName, str))
            assert (isinstance(networks, List))
            params = modeToModeData[modeName]
            if modeName == "bus":
                BusMode(networks, params, microtypeID)
            elif modeName == "auto":
                AutoMode(networks, params, microtypeID)
            elif modeName == "walk":
                WalkMode(networks, params, microtypeID)
            elif modeName == "bike":
                BikeMode(networks, params, microtypeID)
            elif modeName == "rail":
                RailMode(networks, params, microtypeID)
            else:
                print("BAD!")
                Mode(networks, params, microtypeID, "bad")

    def isJammed(self):
        return np.any([n.isJammed for n in self._networks])

    def resetModes(self):
        allModes = [n.getModeValues() for n in self._networks]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        for n in self._networks:
            n.isJammed = False
        self.modes = dict()
        for m in uniqueModes:
            # m.updateN(TravelDemand())
            self.modes[m.name] = m
            self.demands[m.name] = m.travelDemand
        # self.updateNetworks()

    def updateModes(self, n: int = 5):
        # allModes = [n.getModeValues() for n in self._networks]
        # uniqueModes = set([item for sublist in allModes for item in sublist])
        oldSpeeds = self.getModeSpeeds()
        for m in self.modes.values():
            m.updateDemand(self.demands[m.name])
        for it in range(n):
            for m in self.modes.values():  # uniqueModes:
                m.assignVmtToNetworks()
                for n in m.networks:
                    n.updateBaseSpeed()
                m.updateModeBlockedDistance()
                # m.updateN(self.demands[m.name])
            # self.updateNetworks()
            # self.updateMFD()
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

    # def updateMFD(self, iters=1):
    #     for i in range(iters):
    #         for n in self._networks:
    #             n.updateBlockedDistance()
    #             n.MFD()
    #             if np.isnan(n.base_speed):
    #                 n.isJammed = True
    #         # for m in self.modes.values():
    #         #     m.updateN(self.demands[m.name])

    def __str__(self):
        return str([n.base_speed for n in self._networks])

    # def addMode(self, networks: list, mode: Mode):
    #     for network in networks:
    #         assert (isinstance(network, Network))
    #         if network not in self._networks:
    #             self.append(network)
    #         self.modes[mode.name] = mode
    #     return self

    def getModeNames(self) -> list:
        return list(self.modeToNetwork.keys())

    def getModeSpeeds(self) -> np.array:
        return np.array([m.getSpeed() for m in self.modes.values()])

    def getModeOperatingCosts(self):
        out = TotalOperatorCosts()
        for name, mode in self.modes.items():
            out[name] = (mode.getOperatorCosts(), mode.getOperatorRevenues())
        return out
