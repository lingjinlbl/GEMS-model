from math import sqrt, cosh, sinh, cos, sin
from typing import List, Dict

import numba as nb
import numpy as np
import pandas as pd
# from line_profiler_pycharm import profile
from scipy.optimize import minimize

from utils.supply import TravelDemand, TravelDemands

np.seterr(all='ignore')

mph2mps = 1609.34 / 3600


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
                if key in other.__costs:
                    output[key] = (self.__costs[key] + other.__costs[key], self.__revenues[key] + other.__revenues[key])
                else:
                    output[key] = (self.__costs[key], self.__revenues[key])
            else:
                if key in other.__costs:
                    output[key] = (other.__costs[key], other.__revenues[key])
                else:
                    output[key] = (0., 0.)
        return output

    def __str__(self):
        return [key + ' ' + str(item) for key, item in self.__costs.items()]

    def toDataFrame(self, index=None):
        return pd.DataFrame(self.__costs, index=index)


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
    def __init__(self, networks=None, params=None, microtypeID=None, name=None, travelDemandData=None, speedData=None,
                 numpySubnetworkSpeed=None, numpySubnetworkAccumulation=None, numpySubnetworkBlockedDistance=None):
        self.name = name
        self.params = params
        self.microtypeID = microtypeID
        self._idx = params.index.get_loc(microtypeID)
        self.modeParamsColumnToIdx = {i: params.columns.get_loc(i) for i in params.columns}
        self._params = params.to_numpy()
        # self._inds = self.initInds(idx)
        self.networks = networks
        # self._N_tot = 0.0
        # self._N_eff = dict()
        self._networkAccumulation = dict()
        self._networkBlockedDistance = dict()
        self._networkSpeed = dict()
        # self._L_blocked = dict()
        self._averagePassengerDistanceInSystem = 0.0
        # self._VMT_tot = 0.0
        # self._VMT = dict()
        # self._speed = dict()
        self.__bad = False
        self.fixedVMT = True
        self.travelDemand = TravelDemand(travelDemandData)
        self._PMT = self.travelDemand.rateOfPmtPerHour
        self._speedData = speedData

    # def initInds(self, idx):
    #     inds = dict()
    #     for column in self.params.columns():
    #         inds[column] = [(self.params.index.get_loc(idx), self.params.columns.get_loc(column))]
    #     return inds

    @property
    def microtypeSpeed(self):
        return self._speedData[0]

    @microtypeSpeed.setter
    def microtypeSpeed(self, newSpeed):
        self._speedData[0] = newSpeed

    @property
    def relativeLength(self):
        # return self.params.to_numpy()[self._inds["VehicleSize"]]
        return self.params.at[self.microtypeID, "VehicleSize"]

    def updateScenarioInputs(self):
        pass

    def updateRouteAveragedSpeed(self):
        pass

    def updateDemand(self, travelDemand=None):
        # I THINK WE CAN GET RID OF THIS
        a = self._PMT.copy()
        if travelDemand is None:
            travelDemand = self.travelDemand
        else:
            self.travelDemand.adopt(travelDemand)

        b = self._PMT.copy()
        if a != b:
            print('Demand before: ' + str(a))
            print('Demand after: ' + str(b))
            print('What it should be: ' + str(travelDemand.rateOfPmtPerHour))
            print('---------')
        if self._VMT_tot != travelDemand.rateOfPmtPerHour * self.relativeLength:
            print('hdfasdfa: ', str(self._VMT_tot), str(travelDemand.rateOfPmtPerHour * self.relativeLength))
        self._VMT_tot = travelDemand.rateOfPmtPerHour * self.relativeLength

    def getDemandForVmtPerHour(self):
        return self.travelDemand.rateOfPmtPerHour * self.relativeLength

    def getAccessDistance(self) -> float:
        return 0.0

    def updateModeBlockedDistance(self):
        pass
        # for n in self.networks:
        #     self._L_blocked[n] = n.L_blocked[self.name]

    def getSpeedDifference(self, allocation: list):
        speeds = np.array([n.NEF(a * self._VMT_tot * mph2mps, self.name) for n, a in zip(self.networks, allocation)])
        return np.linalg.norm(speeds - np.mean(speeds))

    def assignVmtToNetworks(self):
        Ltot = sum([n.L for n in self.networks])
        for n in self.networks:
            assert (isinstance(n, Network))
            VMT = self._PMT * n.L / Ltot
            # self._VMT[n] = VMT
            # n.setVMT(self.name, self._VMT[n])
            n.setModeAccumulation(self.name, VMT / n.modeSpeed(self.name))
            # self._speed[n] = n.NEF()  # n.NEF(VMT * mph2mps, self.name)
            # self._N_eff[n] = VMT / self._speed[n] * self.relativeLength
            # n.setN(self.name, self._N_eff[n])

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
        return network.modeAccumulation(self.name)[0]  # self._VMT[network] / self._speed[network] / self.relativeLength

    def getNs(self):
        return [self.getN(n) for n in self.networks]

    def getBlockedDistance(self, network):
        return network.getBlockedDistance()

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
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, travelDemandData=None,
                 speedData=None) -> None:
        super(WalkMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID, name="walk",
                                       travelDemandData=travelDemandData, speedData=speedData)
        self.fixedVMT = False
        for n in networks:
            n.addMode(self)
            self._networkAccumulation[n] = n.modeAccumulation(self.name)
            self._networkSpeed[n] = n.modeSpeed(self.name)
            self._networkBlockedDistance[n] = n.modeBlockedDistance(self.name)
            # self._L_blocked[n] = 0.0
            # self._VMT[n] = 0.0
            # self._N_eff[n] = 0.0
            # self._speed[n] = n.base_speed
            n.setModeSpeed(self.name, self.speedInMetersPerSecond)

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["SpeedInMetersPerSecond"]]

    def getSpeed(self):
        return self.speedInMetersPerSecond


class BikeMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, travelDemandData=None,
                 speedData=None) -> None:
        super(BikeMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID, name="bike",
                                       travelDemandData=travelDemandData, speedData=speedData)
        self.fixedVMT = False
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._networkAccumulation[n] = n.modeAccumulation(self.name)
            self._networkSpeed[n] = n.modeSpeed(self.name)
            self._networkBlockedDistance[n] = n.modeBlockedDistance(self.name)
            # self._L_blocked[n] = 0.0
            # self._VMT[n] = 0.0
            # self._N_eff[n] = 0.0
            # self._speed[n] = n.base_speed
            n.setModeSpeed(self.name, self.speedInMetersPerSecond)
        self.bikeLanePreference = 2.0

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["SpeedInMetersPerSecond"]]

    @property
    def dedicatedLanePreference(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["DedicatedLanePreference"]]

    def getSpeed(self):
        return self.speedInMetersPerSecond

    def distanceOnDedicatedLanes(self, capacityTot, capacityDedicated) -> (float, float):
        capacityMixed = capacityTot - capacityDedicated
        portionMixed = (
                capacityDedicated / capacityTot + self.dedicatedLanePreference * capacityMixed / capacityTot)
        N = self._PMT[0] / self.speedInMetersPerSecond
        N_dedicated = min([capacityDedicated, (1 - portionMixed) * N])
        N_mixed = N - N_dedicated
        return N_dedicated * self.speedInMetersPerSecond, N_mixed * self.speedInMetersPerSecond

    def assignVmtToNetworks(self):
        capacityTot = sum([n.L * n.jamDensity for n in self.networks])
        capacityDedicated = sum([n.L * n.jamDensity for n in self.networks if n.dedicated])
        capacityMixed = capacityTot - capacityDedicated
        VMT_dedicated, VMT_mixed = self.distanceOnDedicatedLanes(capacityTot, capacityDedicated)
        for n in self.networks:
            if n.dedicated | (n.L == 0):
                if VMT_dedicated == 0:
                    VMT = 0
                else:
                    VMT = VMT_dedicated * n.L * n.jamDensity / capacityDedicated
            else:
                if VMT_mixed == 0:
                    VMT = 0
                else:
                    VMT = VMT_mixed * n.L * n.jamDensity / capacityMixed
            # self._VMT[n] = VMT
            # n.setVMT(self.name, self._VMT[n])
            if np.isnan(VMT / n.modeSpeed(self.name)):
                print('STOPPPPP')
            n.setModeAccumulation(self.name, VMT / n.modeSpeed(self.name))
            # self._N_eff[n] = VMT / self._speed[n] * self.relativeLength
            # n.setN(self.name, self._N_eff[n])

    def getPortionDedicated(self) -> float:
        if self._PMT > 0:
            tot = 0.0
            tot_dedicated = 0.0
            for key, val in self._networkAccumulation.items():
                tot += val
                if key.dedicated:
                    tot_dedicated += val
            return np.nan_to_num(tot_dedicated / tot)
        else:
            return 0.0

    def updateScenarioInputs(self):
        pass
        # self._params = self.params.to_numpy()
        # for n in self.networks:
        # self._L_blocked[n] = 0.0
        # self._VMT[n] = 0.0
        # self._N_eff[n] = 0.0
        # self._speed[n] = n.base_speed
        # self.__operatingL[n] = self.updateOperatingL(n)


class RailMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, travelDemandData=None,
                 speedData=None) -> None:
        super(RailMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID, name="rail",
                                       travelDemandData=travelDemandData, speedData=speedData)
        self.fixedVMT = True
        # self.initInds(idx)
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._networkAccumulation[n] = n.modeAccumulation(self.name)
            self._networkSpeed[n] = n.modeSpeed(self.name)
            self._networkBlockedDistance[n] = n.modeBlockedDistance(self.name)
            # self._L_blocked[n] = 0.0
            # self._VMT[n] = 0.0
            # self._N_eff[n] = 0.0
            # self._speed[n] = n.base_speed
            n.setModeSpeed(self.name, n.base_speed)

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def routeAveragedSpeed(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["SpeedInMetersPerSecond"]]

    @property
    def vehicleOperatingCostPerHour(self):
        # return self.params.to_numpy()[self._inds["VehicleOperatingCostsPerHour"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleOperatingCostsPerHour"]]

    @property
    def fare(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerStartCost"]]

    @property
    def headwayInSec(self):
        # return self.params.to_numpy()[self._inds["Headway"]]
        return self.params.at[self.microtypeID, "Headway"]

    @property
    def stopSpacingInMeters(self):
        # return self.params.to_numpy()[self._inds["StopSpacing"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["StopSpacing"]]

    @property
    def portionAreaCovered(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["CoveragePortion"]]
        # return self.params.at[self.microtypeID, "CoveragePortion"]

    def updateDemand(self, travelDemand=None):
        if travelDemand is not None:
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

    # def assignVmtToNetworks(self):
    #     Ltot = sum([n.L for n in self.networks])
    #     for n in self.networks:
    #         VMT = self._VMT_tot * n.L / Ltot
    #         self._VMT[n] = VMT
    #         n.setVMT(self.name, self._VMT[n])
    #         self._speed[n] = self.routeAveragedSpeed
    #         self._N_eff[n] = VMT / self._speed[n] * self.relativeLength
    #         n.setN(self.name, self._N_eff[n])

    def updateScenarioInputs(self):
        self._params = self.params.to_numpy()
        # for n in self.networks:
        #     # self._L_blocked[n] = 0.0
        #     self._VMT[n] = 0.0
        # self._N_eff[n] = 0.0
        # self._speed[n] = n.base_speed
        # self.__operatingL[n] = self.updateOperatingL(n)


class AutoMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, travelDemandData=None,
                 speedData=None) -> None:
        super(AutoMode, self).__init__(networks=networks, params=modeParams, microtypeID=microtypeID, name="auto",
                                       travelDemandData=travelDemandData, speedData=speedData)
        self.MFDmode = "single"
        self.fixedVMT = False
        for n in networks:
            n.addMode(self)
            self._networkAccumulation[n] = n.modeAccumulation(self.name)
            self._networkSpeed[n] = n.modeSpeed(self.name)
            self._networkBlockedDistance[n] = n.modeBlockedDistance(self.name)
            # self._N[n] = 0.0
            # self._L_blocked[n] = 0.0
            # self._VMT[n] = 0.0
            # self._N_eff[n] = 0.0
            # self._speed[n] = n.base_speed
            n.setModeSpeed(self.name, n.base_speed)

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self.microtypeID, "PerMileCost"]

    @property
    def relativeLength(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleSize"]]

    def getSpeed(self):
        return self._speedData[0]

    def x0(self):
        return np.array([1. / len(self.networks)] * len(self.networks))

    def constraints(self):
        return dict(type='eq', fun=lambda x: sum(x) - 1.0, jac=lambda x: [1.0] * len(x))

    def bounds(self):
        return [(0.0, 1.0)] * len(self.networks)

    def updateDemand(self, travelDemand=None):  # TODO: Why did I add this?
        if travelDemand is None:
            travelDemand = self.travelDemand
        else:
            self.travelDemand = travelDemand
        self._VMT_tot = travelDemand.rateOfPmtPerHour * self.relativeLength

    # @profile
    def assignVmtToNetworks(self):
        # FIXME:  This doesn't work
        if len(self.networks) == 1:
            if self.MFDmode == "single":
                n = self.networks[0]
                # assert (isinstance(n, Network))
                # self._VMT[n] = self._VMT_tot
                n.setModeAccumulation(self.name, self._PMT / n.modeSpeed(self.name))
                # self._speed[n] = n.NEF(self._VMT_tot * mph2mps, self.name, self.override)
                # n.setVMT(self.name, self._VMT[n])
                # self._N_eff[n] = self._VMT_tot / self._speed[n] * self.relativeLength

                # self._networkAccumulation[n][0] = self._VMT_tot / self._speed[n]
                # n.setN(self.name, self._N_eff[n])
            else:
                n = self.networks[0]
                # self._speed[n] = n.getTransitionMatrixMeanSpeed()  # TODO: Check Units
                self._VMT[n] = self._VMT_tot
                n.setVMT(self.name, self._VMT[n])
                # self._N_eff[n] = n.getNetworkStateData().finalAccumulation * self.relativeLength  # TODO: take avg
                self._networkAccumulation[n][0] = n.getNetworkStateData().finalAccumulation
                n.setN(self.name, self._N_eff[n])
        elif len(self.networks) > 1:
            res = minimize(self.getSpeedDifference, self.x0(), constraints=self.constraints(), bounds=self.bounds())
            for ind, (n, a) in enumerate(zip(self.networks, res.x)):
                VMT = a * self._VMT_tot
                spd = n.NEF(a * self._VMT_tot * mph2mps, self.name)
                N_eff = VMT / spd
                self._VMT[n] = VMT
                self._speed[n] = spd
                n.setVMT(self.name, VMT)
                self._N_eff[n] = N_eff
                n.setN(self.name, self._N_eff[n])
                self._networkAccumulation[n][ind] = N_eff
        else:
            print("OH NO!")

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
    def __init__(self, networks, modeParams: pd.DataFrame, microtypeID: str, travelDemandData=None,
                 speedData=None) -> None:
        super().__init__(networks=networks, params=modeParams, microtypeID=microtypeID, name="bus",
                         travelDemandData=travelDemandData, speedData=speedData)
        self.fixedVMT = True
        self.__operatingL = dict()
        self.__availableRoadNetworkDistance = sum([n.L for n in self.networks])

        for n in networks:
            n.addMode(self)
            self._networkAccumulation[n] = n.modeAccumulation(self.name)
            self._networkSpeed[n] = n.modeSpeed(self.name)
            self._networkBlockedDistance[n] = n.modeBlockedDistance(self.name)
            # self._L_blocked[n] = 0.0
            # self._VMT[n] = 0.0
            # self._N_eff[n] = 0.0
            # self._speed[n] = n.base_speed
            self.__operatingL[n] = self.updateOperatingL(n)

        self.__routeLength = self.updateRouteLength()
        self.travelDemand = TravelDemand(travelDemandData)
        self.routeAveragedSpeed = self.getSpeed()
        self.occupancy = 0.0
        self.updateModeBlockedDistance()
        self.__N = self.getN()

    @property
    def routeAveragedSpeed(self):
        return self._speedData[0]

    @routeAveragedSpeed.setter
    def routeAveragedSpeed(self, spd):
        self._speedData[0] = spd
        self.__N = self.getN()

    @property
    def headwayInSec(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["Headway"]]

    @property
    def passengerWaitInSec(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PassengerWait"]]

    @property
    def passengerWaitInSecDedicated(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PassengerWaitDedicated"]]

    @property
    def stopSpacingInMeters(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["StopSpacing"]]

    @property
    def minStopTimeInSec(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["MinStopTime"]]

    @property
    def fare(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PerStartCost"]]

    @property
    def perStart(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PerStartCost"]]

    @property
    def perEnd(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["PerMileCost"]]

    @property
    def vehicleOperatingCostPerHour(self):
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleOperatingCostPerHour"]]

    @property
    def routeDistanceToNetworkDistance(self) -> float:
        """
        Changed January 2021: Removed need for car-only subnnetworks.
        Now buses only run on a fixed portion of the bus/car subnetwork
        """
        return self._params[self._idx, self.modeParamsColumnToIdx["CoveragePortion"]]

    @property
    def relativeLength(self):
        # return self.params.to_numpy()[self._inds["VehicleSize"]]
        return self._params[self._idx, self.modeParamsColumnToIdx["VehicleSize"]]

    def updateScenarioInputs(self):
        self._params = self.params.to_numpy()
        for n in self.networks:
            # self._L_blocked[n] = 0.0
            # self._VMT[n] = 0.0
            # self._N_eff[n] = 0.0
            # self._speed[n] = n.base_speed
            self.__operatingL[n] = self.updateOperatingL(n)

    # def updateDemand(self, travelDemand=None):
    #     if travelDemand is not None:
    #         self.travelDemand = travelDemand
    #     self._VMT_tot = self.getDemandForVmtPerHour()

    def getAccessDistance(self) -> float:
        """Order of magnitude estimate for average walking distance to nearest stop"""
        return self.stopSpacingInMeters / 4.0 / self.routeDistanceToNetworkDistance / 2.0

    def getDemandForVmtPerHour(self):
        return self.getRouteLength() / self.headwayInSec * 3600. / 1609.34

    def getOperatingL(self, network=None):
        if network is None:
            return self.__operatingL
        else:
            return self.__operatingL[network]

    def updateOperatingL(self, network) -> float:
        """Changed January 2021: Buses only operate on a portion of subnetwork"""
        if network.dedicated:
            return network.L
        else:
            dedicatedDistanceToBus = sum([n.L for n in self.networks if n.dedicated])
            totalDistance = self.__availableRoadNetworkDistance
            dedicatedDistanceToOther = totalDistance - sum([n.L for n in self.networks])
            undedicatedDistance = totalDistance - dedicatedDistanceToBus
            return max(0, (self.routeDistanceToNetworkDistance * totalDistance - dedicatedDistanceToBus) * (
                    network.L / (undedicatedDistance - dedicatedDistanceToOther)))
            # return self.routeDistanceToNetworkDistance * totalDistance * network.L / (totalDistance - dedicatedDistance)

    def getN(self, network=None):
        """Changed January 2021: Buses only operate on a portion of subnetwork"""
        if network:
            return self.getOperatingL(network) / self.routeAveragedSpeed / self.headwayInSec
        else:
            return self.getRouteLength() / self.routeAveragedSpeed / self.headwayInSec

    def getRouteLength(self):
        return self.__routeLength

    def updateRouteLength(self):
        return sum([n.L for n in self.networks]) * self.routeDistanceToNetworkDistance

    def getSubNetworkSpeed(self, network):
        # averageStopDuration = self.min_stop_time + self.passenger_wait * (
        #         self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) / (
        #                               self.routeAveragedSpeed / self.stop_spacing * self.N_tot)
        # return base_speed / (1 + averageStopDuration * base_speed / self.stop_spacing)
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        autoSpeed = network.autoSpeed
        numberOfStopsInSubnetwork = self.getOperatingL(network) / self.stopSpacingInMeters
        numberOfStopsInRoute = self.getRouteLength() / self.stopSpacingInMeters
        pass_per_stop = (self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour
                         ) / numberOfStopsInRoute * self.headwayInSec / 3600.
        stopping_time = numberOfStopsInSubnetwork * self.minStopTimeInSec
        stopped_time = perPassenger * pass_per_stop * numberOfStopsInSubnetwork + stopping_time
        driving_time = self.getOperatingL(network) / autoSpeed
        spd = self.getOperatingL(network) / (stopped_time + driving_time)
        if np.isnan(spd):
            spd = 0.001
            self.__bad = True
        else:
            self.__bad = False
        return spd

    # TODO: store speeds in array
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
        speeds = np.zeros(len(self.networks), dtype=float)
        for idx, n in enumerate(self.networks):
            if n.L > 0:
                # n_bus = self.getN(n)
                bus_speed = self.getSubNetworkSpeed(n)
                meters[idx] = self.getOperatingL(n)
                seconds[idx] = self.getOperatingL(n) / bus_speed
                speeds[idx] = meters[idx] / seconds[idx]
        if np.sum(seconds) > 0:
            spd = np.sum(meters) / np.sum(seconds)
            out = spd
        else:
            out = next(iter(self.networks)).getBaseSpeed()
        return out

    def updateRouteAveragedSpeed(self):
        self.routeAveragedSpeed = self.getSpeed()

    def calculateBlockedDistance(self, network) -> float:
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        # bs = network.base_speed
        if network.autoSpeed > 0:
            numberOfStops = self.getRouteLength() / self.stopSpacingInMeters
            # numberOfBuses = self.getN(network)
            meanTimePerStop = (self.minStopTimeInSec + self.headwayInSec * perPassenger * (
                    self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour) / (
                                       numberOfStops * 3600.0))
            portionOfTimeStopped = min([meanTimePerStop * meanTimePerStop / self.headwayInSec, 1.0])
            # TODO: Think through this more fully. Is this the right way to scale up this time to distance?
            out = portionOfTimeStopped * network.avgLinkLength * self.getN(network)
            out = min(out, numberOfStops * network.avgLinkLength / 100,
                      (network.L - network.getBlockedDistance()) * 0.5)
            # portionOfRouteBlocked = out / self.routeLength
        else:
            out = 0
        return out

    def updateModeBlockedDistance(self):
        for n in self.networks:
            L_blocked = self.calculateBlockedDistance(n)
            self._networkBlockedDistance[n] = L_blocked
            n.getNetworkStateData().blockedDistance = L_blocked  # HACK: Only one mode can block distance at a time
            if n.getNetworkStateData().blockedDistance > self.getRouteLength():
                print('HMMMMM')

    # @profile
    def assignVmtToNetworks(self):
        speeds = self.getSpeeds()
        times = []
        lengths = []
        for ind, n in enumerate(self.networks):
            spd = speeds[ind]
            if spd < 0.1:
                # print("Speed to small: ", spd)
                spd = 0.1
            times.append(self.getOperatingL(n) / spd)
            lengths.append(self.getOperatingL(n))
        for ind, n in enumerate(self.networks):
            assert isinstance(n, Network)
            if speeds[ind] >= 0:
                # VMT = self._VMT_tot * lengths[ind] / self.getRouteLength()
                # self._VMT[n] = VMT
                # n.setVMT(self.name, self._VMT[n])
                # n.updateBaseSpeed()
                if n.dedicated:
                    n.runSingleNetworkMFD()
                # self._speed[n] = self.getSubNetworkSpeed(n)
                n.setModeSpeed(self.name, self.getSubNetworkSpeed(n))
                networkAccumulation = self.__N * times[ind] / sum(times)
                # self._N_eff[n] = min(VMT / self._speed[n] * self.relativeLength,
                #                      self.getRouteLength() / n.avgLinkLength / 2)  # Why was this divided by 100?
                self._networkAccumulation[n][0] = min(
                    networkAccumulation, self.getRouteLength() / n.avgLinkLength / 2 * self.relativeLength)
                # n.setN(self.name, self._N_eff[n])

    # def updateCommercialSpeed(self):
    #     self.routeAveragedSpeed = self.getRouteLength() / sum(
    #         [self.getOperatingL(n) / spd for n, spd in self._speed.items()])

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

    def getPortionDedicated(self) -> float:
        if self._PMT > 0:
            tot = 0.0
            tot_dedicated = 0.0
            for key, val in self._networkAccumulation.items():  # TODO: Take into account different speeds
                tot += val
                if key.dedicated:
                    tot_dedicated += val
            if tot == 0:
                return 0.
            else:
                return np.nan_to_num(tot_dedicated / tot)
        else:
            return 0.0


class Network:
    def __init__(self, data, characteristics, subNetworkId, diameter=None, microtypeID=None, microtypeSpeed=None,
                 modeSpeed=None, modeAccumulation=None, modeBlockedDistance=None, modeVehicleSize=None,
                 networkLength=None, modeToIdx=None):
        self.data = data

        self.characteristics = characteristics
        self.charColumnToIdx = {i: characteristics.columns.get_loc(i) for i in characteristics.columns}
        self.dataColumnToIdx = {i: data.columns.get_loc(i) for i in data.columns}
        self.microtypeID = microtypeID
        self.subNetworkId = subNetworkId
        self._iloc = data.index.get_loc(subNetworkId)
        self.__data = data.iloc[self._iloc, :].to_numpy()
        self.type = self.characteristics.iat[self._iloc, self.charColumnToIdx["Type"]]
        # self.L_blocked = dict()
        self._modes = dict()
        self.dedicated = characteristics.loc[subNetworkId, "Dedicated"]
        self.isJammed = False
        # self._VMT = dict()
        # self._N_eff = dict()
        self._networkStateData = NetworkStateData().initFromNetwork(self)
        self._N_init = 0.0
        self._N_final = 0.0
        self._V_mean = self.freeFlowSpeed
        # These are for debugging and can likely be removed
        self._Q_prev = 0.0
        self._Q_curr = 0.0
        self._V_init = 0.0
        self._V_final = self.freeFlowSpeed
        self._V_steadyState = self.freeFlowSpeed
        self.__modeToIdx = modeToIdx
        self.__microtypeSpeed = microtypeSpeed
        self.__modeSpeed = modeSpeed
        self.__modeAccumulation = modeAccumulation
        self.__modeVehicleSize = modeVehicleSize
        self.__networkLength = networkLength
        self.__modeBlockedDistance = modeBlockedDistance
        np.copyto(self.__networkLength, self.__data[self.dataColumnToIdx["Length"]])
        self.MFD = self.defineMFD()
        if diameter is None:
            self.__diameter = 1.0
        else:
            self.__diameter = diameter

    def runSingleNetworkMFD(self):
        Ntot = (self.__modeAccumulation * self.__modeVehicleSize).sum()
        Leff = self.L - self.__modeBlockedDistance.sum()
        self.autoSpeed = self.MFD(Ntot / Leff)

    def modeAccumulation(self, mode):
        return self.__modeAccumulation[self.__modeToIdx[mode], None]

    def getAccumulationExcluding(self, mode: str):
        return np.sum(self.__modeAccumulation[idx] for m, idx in self.__modeToIdx.items() if m != mode)

    def setModeAccumulation(self, mode, accumulation: float):
        np.copyto(self.__modeAccumulation[self.__modeToIdx[mode], None], accumulation)

    def modeSpeed(self, mode):
        return self.__modeSpeed[self.__modeToIdx[mode]]

    def setModeSpeed(self, mode, speed: float):
        np.copyto(self.__modeSpeed[self.__modeToIdx[mode], None], speed)

    def modeBlockedDistance(self, mode):
        return self.__modeBlockedDistance[self.__modeToIdx[mode], None]

    def setModeBlockedDistance(self, mode, blockedDistance: float):
        np.copyto(self.__modeBlockedDistance[self.__modeToIdx[mode], None], blockedDistance)

    def setModeVehicleSize(self, mode, vehicleSize: float):
        np.copyto(self.__modeVehicleSize[self.__modeToIdx[mode], None], vehicleSize)

    def recompileMFD(self):
        self.__data = self.data.iloc[self._iloc, :].to_numpy()
        self.MFD = self.defineMFD()

    def defineMFD(self):
        if (self.characteristics.iat[self._iloc, self.charColumnToIdx["Type"]] == "Road") & ~self.characteristics.iat[
            self._iloc, self.charColumnToIdx["Dedicated"]]:
            if self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "loder":
                vMax = self.__data[self.dataColumnToIdx["vMax"]]
                densityMax = self.__data[self.dataColumnToIdx["densityMax"]]
                capacityFlow = self.__data[self.dataColumnToIdx["capacityFlow"]]
                smoothingFactor = self.__data[self.dataColumnToIdx["smoothingFactor"]]
                waveSpeed = self.__data[self.dataColumnToIdx["waveSpeed"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    if density == 0:
                        return vMax
                    else:
                        speedExp = smoothingFactor / density * np.log(
                            np.exp(- vMax * density / smoothingFactor) + np.exp(
                                -capacityFlow / smoothingFactor) + np.exp(
                                - (density - densityMax) * waveSpeed / smoothingFactor))
                        speedLinear = vMax * (1. - density / densityMax)
                    return max(min(speedLinear, speedExp), 0.05)

            elif self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "quadratic":
                vMax = self.__data[self.dataColumnToIdx["vMax"]]
                densityMax = self.__data[self.dataColumnToIdx["densityMax"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    return max(vMax * (1. - density / densityMax), 0.05)

            elif self.characteristics.iat[self._iloc, self.charColumnToIdx["MFD"]] == "bottleneck":
                vMax = self.__data[self.dataColumnToIdx["vMax"]]
                capacityFlow = self.__data[self.dataColumnToIdx["capacityFlow"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(density):
                    if density > (capacityFlow / vMax):
                        return capacityFlow / density
                    else:
                        return vMax

            else:
                vMax = self.__data[self.dataColumnToIdx["vMax"]]

                @nb.cfunc("float64(float64)", fastmath=True, parallel=False, cache=True)
                def _MFD(_):
                    return vMax
        else:
            vMax = self.__data[self.dataColumnToIdx["vMax"]]

            def _MFD(_):
                return vMax

        return _MFD

    def updateNetworkData(self):  # CONSOLIDATE
        np.copyto(self.__data, self.data.iloc[self._iloc, :].to_numpy())
        self.MFD = self.defineMFD()

    # @property
    # def type(self):
    #     return self.characteristics.iat[self._idx, self.charColumnToIdx["Type"]]

    @property
    def base_speed(self):
        return self.getNetworkStateData().averageSpeed

    @property
    def autoSpeed(self):
        return self.__modeSpeed[self.__modeToIdx['auto']]

    @autoSpeed.setter
    def autoSpeed(self, newSpeed):
        self.__modeSpeed[self.__modeToIdx['auto']] = newSpeed

    @base_speed.setter
    def base_speed(self, spd):
        self.getNetworkStateData().averageSpeed = spd

    @property
    def avgLinkLength(self):
        return self.__data[self.dataColumnToIdx["avgLinkLength"]]

    @property
    def freeFlowSpeed(self):
        return self.__data[self.dataColumnToIdx["vMax"]]

    @property
    def jamDensity(self):
        densityMax = self.__data[self.dataColumnToIdx["densityMax"]]
        if np.isnan(densityMax) | (densityMax <= 0.0):
            return self.__data[self.dataColumnToIdx["capacityFlow"]] / self.__data[self.dataColumnToIdx["vMax"]] * 4.
        else:
            return densityMax

    @property
    def L(self):
        return self.__networkLength[0]
        # return self.__data[self.dataColumnToIdx["Length"]]

    @property
    def diameter(self):
        return self.__diameter

    @property
    def modesAllowed(self):
        return self.characteristics['ModesAllowed'].iloc[self._iloc]

    def __str__(self):
        return str(tuple(self._VMT.keys()))

    def __contains__(self, mode):
        return mode in self._modes

    def updateScenarioInputs(self):
        np.copyto(self.__data, self.data.iloc[self._iloc, :].to_numpy())

    def resetAll(self):
        # self.L_blocked = dict()
        self._modes = dict()
        self.base_speed = self.freeFlowSpeed
        self.isJammed = False

    def resetSpeeds(self):
        # self.base_speed = self.freeFlowSpeed
        self.isJammed = False
        # for key in self.L_blocked.keys():
        #     self.L_blocked[key] = 0.0

    # def resetModes(self):
    #     for mode in self._modes.values():
    #         # self.N_eq[mode.name] = mode.getN(self) * mode.params.relativeLength
    #         self._VMT[mode] = mode._VMT[self]
    #         self._N_eff[mode] = mode._N_eff[self]
    #         self.L_blocked[mode.name] = mode.getBlockedDistance(self)
    #     self.isJammed = False
    #     self.base_speed = self.freeFlowSpeed
    # mode.reset()

    def setVMT(self, mode: str, VMT: float):
        self._VMT[mode] = VMT

    def setN(self, mode: str, N: float):
        self._N_eff[mode] = N

    # def updateBaseSpeed(self, override=False):
    #     # out = self.NEF(overrideMatrix=override)
    #     if self.dedicated:
    #         self.base_speed = self.NEF(overrideMatrix=override)

    def getSpeedFromMFD(self, N):
        L_tot = self.L - self.getBlockedDistance()
        N_0 = self.jamDensity * L_tot
        return self.freeFlowSpeed * (1. - N / N_0)

    def NEF2(self) -> float:
        if self.type == "Road":
            return self._V_mean
        else:
            return self.freeFlowSpeed

    def NEF(self, Q=None, modeIgnored=None, overrideMatrix=False) -> float:
        if self.type == 'Road':
            if 'auto' in self.getModeNames() and not overrideMatrix:
                # print("THIS WILL BREAK THINGS")
                return self.__microtypeSpeed[self.__modeToIdx['auto']]
                # return self._networkStateData.averageSpeed
            else:
                if Q is None:
                    Qtot = sum([VMT for VMT in self._VMT.values()]) * mph2mps
                else:
                    Qtot = Q
                    for mode, Qmode in self._VMT.items():
                        if mode != modeIgnored:
                            Qtot += Qmode * mph2mps
                if Qtot == 0:
                    return self.freeFlowSpeed
                self._Q_curr = Qtot
                L_tot = self.L - self.getBlockedDistance()
                L_0 = 10 * 1609.34  # TODO: Get average distance, don't hardcode
                t = 3 * 3600.  # TODO: Add timestep duration in seconds
                N_0 = self.jamDensity * L_tot
                V_0 = self.freeFlowSpeed
                N_init = self._N_init
                if N_0 ** 2. / 4. >= N_0 * Qtot / V_0:
                    # Stable state
                    A = sqrt(N_0 ** 2. / 4. - N_0 * Qtot / V_0)
                    var = A * V_0 * t / (N_0 * L_0)
                    N_final = N_0 / 2 - A * ((N_0 / 2 - N_init) * cosh(var) + A * sinh(var)) / (
                            (N_0 / 2 - N_init) * sinh(var) + A * cosh(var))
                    V_init = self.getSpeedFromMFD(N_init)
                    V_final = self.getSpeedFromMFD(N_final)
                    V_steadyState = self.getSpeedFromMFD(N_0 / 2 - A)
                else:
                    Aprime = sqrt(N_0 * Qtot / V_0 - N_0 ** 2. / 4.)
                    var = Aprime * V_0 * t / (N_0 * L_0)
                    N_final = N_0 / 2 - Aprime * ((N_0 / 2 - N_init) * cos(var) + Aprime * sin(var)) / (
                            (N_0 / 2 - N_init) * sin(var) + Aprime * cos(var))
                    V_init = self.getSpeedFromMFD(N_init)
                    V_final = self.getSpeedFromMFD(N_final)
                    V_steadyState = 0
                self._networkStateData.N_final = N_final
                self._networkStateData.V_init = V_init
                self._networkStateData.V_final = V_final
                self._networkStateData.V_steadyState = V_steadyState
                if overrideMatrix:
                    self.base_speed = max([0.1, (V_init + V_final) / 2.0])
                return max([2.0, (V_init + V_final) / 2.0])  # TODO: Actually take the integral
        else:
            return self.freeFlowSpeed

    def getBaseSpeed(self):
        if self.base_speed > 0.01:
            return self.base_speed
        else:
            return 0.01

    # def updateBlockedDistance(self):
    #     for mode in self._modes.values():
    #         mode.updateModeBlockedDistance()

    def containsMode(self, mode: str) -> bool:
        return mode in self._modes.keys()

    def getBlockedDistance(self) -> float:
        return self.__modeBlockedDistance.sum()

    def addMode(self, mode: Mode):
        self.setModeVehicleSize(mode.name, mode.relativeLength)
        self._modes[mode.name] = mode
        # self.L_blocked[mode.name] = 0.0
        # self._VMT[mode.name] = 0.0
        # self._N_eff[mode.name] = 0.0
        return self

    def getModeNames(self) -> list:
        return list(self._modes.keys())

    def getModeValues(self) -> list:
        return list(self._modes.values())

    def updateFromMFD(self, v, n):
        self._networkStateData.finalSpeed = v[-1]
        self._networkStateData.finalAccumulation = n[-1]
        # self._networkStateData.averageSpeed = np.mean(v)

    def getVMT(self, mode):
        return self.modeAccumulation(mode) * self.modeSpeed(mode)

    def getNetworkStateData(self):
        return self._networkStateData

    def setInitialStateData(self, oldNetworkStateData):
        if len(oldNetworkStateData.n > 0):
            self._networkStateData.initialAccumulation = oldNetworkStateData.n[-1]
            self._networkStateData.initialSpeed = oldNetworkStateData.v[-1]
            self._networkStateData.initialTime = oldNetworkStateData.t[-1]
        # self._networkStateData.nonAutoAccumulation = oldNetworkStateData.nonAutoAccumulation
        # self._networkStateData.blockedDistance = oldNetworkStateData.blockedDistance


class NetworkCollection:
    def __init__(self, networksAndModes, modeToModeData, microtypeID, demandData, speedData, dataToIdx, modeToIdx,
                 verbose=False):
        self._networks = dict()
        self.modeToNetwork = dict()
        self.__modes = dict()

        self.__demandData = demandData
        self._speedData = speedData
        self.__dataToIdx = dataToIdx
        self.__modeToIdx = modeToIdx

        if isinstance(networksAndModes, Dict) and isinstance(modeToModeData, Dict):
            self.populateNetworksAndModes(networksAndModes, modeToModeData, microtypeID)
        self.modes = dict()
        self.demands = TravelDemands([])
        self.verbose = verbose
        # self.resetModes()

    def getMode(self, mode):
        return self.__modes[mode]

    def getModeVMT(self, mode):
        return sum([n.getVMT(mode) for n in self[mode]])

    def updateNetworkData(self):
        for n in self._networks.values():
            n.updateNetworkData()

    def populateNetworksAndModes(self, networksAndModes, modeToModeData, microtypeID):
        # modeToNetwork = dict()
        if isinstance(networksAndModes, Dict):
            for (network, modeNames) in networksAndModes.items():
                assert (isinstance(network, Network))
                sortedModeNames = tuple(sorted(modeNames))
                self._networks[sortedModeNames] = network
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

            # numpySubnetworkSpeed=None, numpySubnetworkAccumulation=None, numpySubnetworkBlockedDistance=None

            if modeName == "bus":
                self._speedData[self.__modeToIdx[modeName]] = networks[0].base_speed
                mode = BusMode(networks, params, microtypeID,
                               travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                               speedData=self._speedData[self.__modeToIdx[modeName], None])
                self.__modes["bus"] = mode
            elif modeName == "auto":
                self._speedData[self.__modeToIdx[modeName]] = networks[0].base_speed
                mode = AutoMode(networks, params, microtypeID,
                                travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                speedData=self._speedData[self.__modeToIdx[modeName], None])
                self.__modes["auto"] = mode
            elif modeName == "walk":
                self._speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                mode = WalkMode(networks, params, microtypeID,
                                travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                speedData=self._speedData[self.__modeToIdx[modeName], None])
                self.__modes["walk"] = mode
            elif modeName == "bike":
                self._speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                mode = BikeMode(networks, params, microtypeID,
                                travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                speedData=self._speedData[self.__modeToIdx[modeName], None])
                self.__modes["bike"] = mode
            elif modeName == "rail":
                self._speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                mode = RailMode(networks, params, microtypeID,
                                travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                speedData=self._speedData[self.__modeToIdx[modeName], None])
                self.__modes["rail"] = mode
            else:
                print("BAD!")
                Mode(networks, params, microtypeID, "bad")

    def updateModeData(self):
        for m in self.__modes.values():
            m.updateScenarioInputs()
            m.updateModeBlockedDistance()

    def isJammed(self):
        return np.any([n.isJammed for n in self._networks])

    def resetModes(self):
        allModes = [n.getModeValues() for n in self._networks.values()]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        for n in self._networks.values():
            n.isJammed = False
            n.resetSpeeds()
        self.modes = dict()
        for m in uniqueModes:
            # m.updateN(TravelDemand())
            self.modes[m.name] = m
            self.demands[m.name] = m.travelDemand
        # self.updateNetworks()

    # @profile
    def updateModes(self, nIters: int = 1):
        # allModes = [n.getModeValues() for n in self._networks]
        # uniqueModes = set([item for sublist in allModes for item in sublist])
        # oldSpeeds = self.getModeSpeeds()

        # TODO: Check ifthis is needed
        # for m in self.modes.values():
        #     m.updateDemand(self.demands[m.name])

        for m in self.modes.values():  # uniqueModes:
            # replace this with assign accumulation to networks
            m.assignVmtToNetworks()
            # for n in m.networks:
            #     n.updateBaseSpeed()  # this will now be taken care of in microtype mfd calculations
            m.updateModeBlockedDistance()
            m.updateRouteAveragedSpeed()

        for modes, n in self:
            nonAutoAccumulation = n.getAccumulationExcluding('auto')
            n.getNetworkStateData().nonAutoAccumulation = nonAutoAccumulation

    # def updateNetworks(self):
    #     for n in self._networks:
    #         n.resetModes()

    def __getitem__(self, item):
        return [n for idx, n in self._networks.items() if item in idx]

    def __str__(self):
        return str([n.base_speed for n in self._networks])

    def __iter__(self):
        return iter(self._networks.items())

    def __contains__(self, mode):
        return mode in self.modeToNetwork

    def getModeNames(self) -> list:
        return list(self.modeToNetwork.keys())

    def getModeSpeeds(self) -> np.array:
        # TODO: This can get slow
        return np.array([m.getSpeed() for m in self.modes.values()])

    def getModeOperatingCosts(self):
        out = TotalOperatorCosts()
        for name, mode in self.modes.items():
            out[name] = (mode.getOperatorCosts(), mode.getOperatorRevenues())
        return out

    def iterModes(self):
        return iter(self.__modes)


class NetworkStateData:
    def __init__(self, data=None):
        if data is None:
            self.__data = dict()
            self.finalAccumulation = 0.0
            self.finalProduction = 0.0
            self.initialSpeed = np.inf
            self.finalSpeed = np.inf
            self.steadyStateSpeed = np.inf
            self.initialAccumulation = 0.0
            self.nonAutoAccumulation = 0.0
            self.blockedDistance = 0.0
            self.averageSpeed = 0.0
            self.initialTime = 0.0
            self.defaultSpeed = 0.0
            self.inflow = np.zeros(0)
            self.outflow = np.zeros(0)
            self.flowMatrix = np.zeros(0)
            self.v = np.zeros(0)
            self.n = np.zeros(0)
            self.t = np.zeros(0)
        else:
            self.finalAccumulation = data.finalAccumulation
            self.finalProduction = data.finalProduction
            self.initialSpeed = data.initialSpeed
            self.finalSpeed = data.finalSpeed
            self.steadyStateSpeed = data.steadyStateSpeed
            self.initialAccumulation = data.initialAccumulation
            self.nonAutoAccumulation = data.nonAutoAccumulation
            self.blockedDistance = data.blockedDistance
            self.averageSpeed = data.averageSpeed
            self.initialTime = data.initialTime
            self.inflow = data.inflow
            self.outflow = data.outflow
            self.flowMatrix = data.flowMatrix
            self.defaultSpeed = data.defaultSpeed
            self.v = data.v
            self.n = data.n
            self.t = data.t

    def initFromNetwork(self, network: Network):
        self.initialSpeed = network.freeFlowSpeed
        self.finalSpeed = network.freeFlowSpeed
        self.steadyStateSpeed = network.freeFlowSpeed
        self.averageSpeed = network.freeFlowSpeed
        self.defaultSpeed = network.freeFlowSpeed
        return self

    def resetBlockedDistance(self):
        self.blockedDistance = 0.0

    def resetNonAutoAccumulation(self):
        self.nonAutoAccumulation = 0.0

    def reset(self):
        self.__data = dict()
        self.finalAccumulation = 0.0
        self.finalProduction = 0.0
        self.initialSpeed = np.inf
        self.finalSpeed = np.inf
        self.steadyStateSpeed = np.inf
        self.initialAccumulation = 0.0
        self.nonAutoAccumulation = 0.0
        self.blockedDistance = 0.0
        self.averageSpeed = self.defaultSpeed
        self.initialTime = 0.0
        self.inflow = np.zeros(0)
        self.outflow = np.zeros(0)
        self.flowMatrix = np.zeros(0)
        self.v = np.zeros(0)
        self.n = np.zeros(0)
        self.t = np.zeros(0)


class CollectedNetworkStateData:
    def __init__(self):
        self.__data = dict()

    def __setitem__(self, key, value: NetworkStateData):
        self.__data[key] = value

    def __getitem__(self, item) -> NetworkStateData:
        return self.__data[item]

    def resetAll(self):
        for _, val in self.__data.items():
            val.reset()
        return self

    def getAutoProduction(self):
        prods = []
        for (mID, modes), val in self.__data.items():
            if "auto" in modes:
                if len(val.t) == 0:
                    continue
                else:
                    prod = np.zeros(len(val.t) - 1)
                    for i in np.arange(len(val.t) - 1):
                        dt = val.t[i + 1] - val.t[i]
                        prod[i] = val.v[i] * val.n[i] * dt
                    prods.append(prod)
                    # prods.append(np.sum(val.v * val.n) * (val.t[1] - val.t[0]))
        return np.array(prods)

    def addMicrotype(self, microtype):
        for modes, network in microtype.networks:
            self[(microtype.microtypeID, modes)] = network.getNetworkStateData()

    def adoptPreviousMicrotypeState(self, microtype):
        for modes, network in microtype.networks:
            network.setInitialStateData(self[(microtype.microtypeID, modes)])

    def getAutoSpeeds(self):
        speeds = []
        ns = []
        inflows = []
        outflows = []
        matrices = []
        ts = None
        labels = []
        for (mID, modes), val in self.__data.items():
            if "auto" in modes:
                speeds.append(val.v[:-1])
                ns.append(val.n[:-1])
                inflows.append(val.inflow[:-1])
                outflows.append(val.outflow[:-1])
                if len(val.flowMatrix.shape) == 2:
                    matrices.append(val.flowMatrix[:, :-1])
                else:
                    matrices.append(val.flowMatrix[:-1])
                if ts is None:
                    ts = val.t[:-1]
                labels.append((mID, modes))
        return ts, np.stack(speeds, axis=-1), np.stack(ns, axis=-1), np.stack(
            inflows, axis=-1), np.stack(outflows, axis=-1), np.stack(matrices, axis=-1), labels

    def __bool__(self):
        return len(self.__data) > 0

    def __iter__(self):
        return iter(self.__data.items())
