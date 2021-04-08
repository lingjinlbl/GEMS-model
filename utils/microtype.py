#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .OD import TransitionMatrix, Allocation
from .choiceCharacteristics import ChoiceCharacteristics
from .network import Network, NetworkCollection, Costs, TotalOperatorCosts, CollectedNetworkStateData


class CollectedTotalOperatorCosts:
    def __init__(self):
        self.__costs = dict()
        self.total = 0.

    def __setitem__(self, key: str, value: TotalOperatorCosts):
        self.__costs[key] = value
        self.updateTotals(value)

    def __getitem__(self, item: str) -> TotalOperatorCosts:
        return self.__costs[item]

    def updateTotals(self, value: TotalOperatorCosts):
        for mode, cost in value:
            self.total += cost

    def __mul__(self, other):
        out = CollectedTotalOperatorCosts()
        for mode in self.__costs.keys():
            out[mode] = self[mode] * other
        return out

    def __add__(self, other):
        out = CollectedTotalOperatorCosts()
        for mode in other.__costs.keys():
            if mode in self.__costs:
                out[mode] = self[mode] + other[mode]
            else:
                out[mode] = other[mode]
        return out

    def __iadd__(self, other):
        for mode in other.__costs.keys():
            if mode in self.__costs:
                self[mode] = self[mode] + other[mode]
            else:
                self[mode] = other[mode]
        return self

    def toDataFrame(self):
        return pd.concat([val.toDataFrame([key]) for key, val in self.__costs.items()])


class Microtype:
    def __init__(self, microtypeID: str, networks: NetworkCollection, costs=None):
        self.microtypeID = microtypeID
        if costs is None:
            costs = dict()
        self.mode_names = set(networks.getModeNames())
        self.networks = networks
        self.updateModeCosts(costs)

    def updateModeCosts(self, costs):
        for (mode, modeCosts) in costs.items():
            assert (isinstance(mode, str) and isinstance(modeCosts, Costs))
            self.networks.modes[mode].costs = modeCosts

    def updateNetworkSpeeds(self, nIters=None):
        self.networks.updateModes(nIters)

    def getModeSpeeds(self) -> dict:
        return {mode: self.getModeSpeed(mode) for mode in self.mode_names}

    def getModeSpeed(self, mode) -> float:
        return self.networks.modes[mode].getSpeed()

    def getModeFlow(self, mode) -> float:
        return self.networks.demands.getRateOfPMT(mode)

    def getModeDemandForPMT(self, mode):
        return self.networks.demands.getRateOfPMT(mode)

    def addModeStarts(self, mode, demand):
        self.networks.demands.addModeStarts(mode, demand)

    def addModeEnds(self, mode, demand):
        self.networks.demands.addModeEnds(mode, demand)

    def addModeDemandForPMT(self, mode, demand, trip_distance_in_miles):
        self.networks.demands.addModeThroughTrips(mode, demand, trip_distance_in_miles)

    def setModeDemand(self, mode, demand, trip_distance_in_miles):
        self.networks.demands.setSingleDemand(mode, demand, trip_distance_in_miles)
        self.networks.updateModes()

    def resetDemand(self):
        self.networks.resetModes()
        self.networks.demands.resetDemand()

    def getModeStartAndEndRate(self, mode: str) -> (float, float):
        return self.networks.demands.getStartRate(mode), self.networks.demands.getStartRate(mode)

    def getModeStartRate(self, mode: str) -> float:
        return self.networks.demands.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self.networks.demands.getAverageDistance(mode)

    def getThroughTimeCostWait(self, mode: str, distanceInMiles: float) -> ChoiceCharacteristics:
        speedMilesPerHour = np.max([self.getModeSpeed(mode), 0.01]) * 2.23694
        if np.isnan(speedMilesPerHour):
            speedMilesPerHour = self.getModeSpeed("auto")
        timeInHours = distanceInMiles / speedMilesPerHour
        cost = distanceInMiles * self.networks.modes[mode].perMile
        wait = 0.
        accessTime = 0.
        protectedDistance = self.networks.modes[mode].getPortionDedicated() * distanceInMiles
        return ChoiceCharacteristics(timeInHours, cost, wait, accessTime, protectedDistance, distanceInMiles)

    def getStartTimeCostWait(self, mode: str) -> ChoiceCharacteristics:
        time = 0.
        cost = self.networks.modes[mode].perStart
        if mode in ['bus', 'rail']:
            wait = self.networks.modes[
                       'bus'].headwayInSec / 3600. / 4.  # TODO: Something better than average of start and end
        else:
            wait = 0.
        walkAccessTime = self.networks.modes[mode].getAccessDistance() * self.networks.modes[
            'walk'].speedInMetersPerSecond / 3600.0
        return ChoiceCharacteristics(time, cost, wait, walkAccessTime)

    def getEndTimeCostWait(self, mode: str) -> ChoiceCharacteristics:
        time = 0.
        cost = self.networks.modes[mode].perEnd
        if mode == 'bus':
            wait = self.networks.modes['bus'].headwayInSec / 3600. / 4.
        else:
            wait = 0.
        walkEgressTime = self.networks.modes[mode].getAccessDistance() * self.networks.modes[
            'walk'].speedInMetersPerSecond / 3600.0
        return ChoiceCharacteristics(time, cost, wait, walkEgressTime)

    def getFlows(self):
        return [mode.getPassengerFlow() for mode in self.networks.modes.values()]

    def getSpeeds(self):
        return [mode.getSpeed() for mode in self.networks.modes.values()]

    def getDemandsForPMT(self):
        return [mode.getPassengerFlow() for mode in
                self.networks.modes.values()]

    # def getPassengerOccupancy(self):
    #     return [self.getModeOccupancy(mode) for mode in self.modes]

    # def getTravelTimes(self):
    #     speeds = np.array(self.getSpeeds())
    #     speeds[~(speeds > 0)] = np.nan
    #     distances = np.array([self.getModeMeanDistance(mode) for mode in self.modes])
    #     return distances / speeds

    # def getTotalTimes(self):
    #     speeds = np.array(self.getSpeeds())
    #     demands = np.array(self.getDemandsForPMT())
    #     times = speeds * demands
    #     times[speeds == 0.] = np.inf
    #     return times

    def __str__(self):
        return 'Demand: ' + str(self.getFlows()) + ' , Speed: ' + str(self.getSpeeds())


class MicrotypeCollection:
    def __init__(self, modeData: dict):
        self.__microtypes = dict()
        self.modeData = modeData
        self.transitionMatrix = None
        self.collectedNetworkStateData = CollectedNetworkStateData()
        self.__modeToMicrotype = dict()

    def __setitem__(self, key: str, value: Microtype):
        self.__microtypes[key] = value

    def __getitem__(self, item: str) -> Microtype:
        return self.__microtypes[item]

    def __contains__(self, item):
        return item in self.__microtypes

    def __len__(self):
        return len(self.__microtypes)

    def microtypeNames(self):
        return list(self.__microtypes.keys())

    def getModeStartRatePerSecond(self, mode):
        return np.array([microtype.getModeStartRate(mode) / 3600. for mID, microtype in self])

    def importMicrotypes(self, subNetworkData: pd.DataFrame, subNetworkCharacteristics: pd.DataFrame,
                         modeToSubNetworkData: pd.DataFrame, microtypeData: pd.DataFrame):
        # uniqueMicrotypes = subNetworkData["MicrotypeID"].unique()
        self.transitionMatrix = TransitionMatrix(microtypeData.MicrotypeID.to_list(),
                                                 diameters=microtypeData.DiameterInMiles.to_list())
        self.__modeToMicrotype = dict()
        for microtypeID, diameter in microtypeData.itertuples(index=False):
            if microtypeID in self:
                self[microtypeID].resetDemand()
            else:
                subNetworkToModes = dict()
                modeToModeData = dict()
                allModes = set()
                for idx in subNetworkCharacteristics.loc[subNetworkCharacteristics["MicrotypeID"] == microtypeID].index:
                    joined = modeToSubNetworkData.loc[
                        modeToSubNetworkData['SubnetworkID'] == idx]
                    subNetwork = Network(subNetworkData, subNetworkCharacteristics, idx, diameter, microtypeID)
                    for n in joined.itertuples():
                        subNetworkToModes.setdefault(subNetwork, []).append(n.ModeTypeID.lower())
                        allModes.add(n.ModeTypeID.lower())
                        self.__modeToMicrotype.setdefault(n.ModeTypeID.lower(), set()).add(microtypeID)
                for mode in allModes:
                    modeToModeData[mode] = self.modeData[mode]
                networkCollection = NetworkCollection(subNetworkToModes, modeToModeData, microtypeID)
                self[microtypeID] = Microtype(microtypeID, networkCollection)
                self.collectedNetworkStateData.addMicrotype(self[microtypeID])

                print("|  Loaded ",
                      len(subNetworkCharacteristics.loc[subNetworkCharacteristics["MicrotypeID"] == microtypeID].index),
                      " subNetworks in microtype ", microtypeID)

    def transitionMatrixMFD(self, durationInHours, collectedNetworkStateData=None, tripStartRate=None):
        if collectedNetworkStateData is None:
            collectedNetworkStateData = self.collectedNetworkStateData
            writeData = True
        else:
            writeData = False

        if tripStartRate is None:
            tripStartRate = self.getModeStartRatePerSecond("auto")

        def v(n, v_0, n_0, n_other, minspeed=0.1):
            n_eff = n + n_other
            v = v_0 * (1. - n_eff / n_0)
            v[v < minspeed] = minspeed
            v[v > v_0] = v_0[v > v_0]
            return v

        def outflow(n, L, v_0, n_0, n_other):
            return v(n, v_0, n_0, n_other, 1.0) * n / L

        def inflow(n, X, L, v_0, n_0, n_other):
            os = X @ (v(n, v_0, n_0, n_other, 1.0) * n / L)
            return os

        def dn_dt(n, demand, L, X, v_0, n_0, n_other):
            inflowval = inflow(n, X, L, v_0, n_0, n_other)
            outflowval = outflow(n, L, v_0, n_0, n_other)
            return demand + inflow(n, X, L, v_0, n_0, n_other) - outflow(n, L, v_0, n_0, n_other)

        # print(tripStartRate)
        characteristicL = np.zeros((len(self)))
        V_0 = np.zeros((len(self)))
        N_0 = np.zeros((len(self)))
        n_other = np.zeros((len(self)))
        n_init = np.zeros((len(self)))
        for microtypeID, microtype in self:
            idx = self.transitionMatrix.idx(microtypeID)
            for modes, autoNetwork in microtype.networks:
                if "auto" in autoNetwork:
                    # for autoNetwork in microtype.networks["auto"]:
                    networkStateData = collectedNetworkStateData[(microtypeID, modes)]
                    # nsd2 = autoNetwork.getNetworkStateData()
                    assert (isinstance(autoNetwork, Network))
                    L_eff = autoNetwork.L - networkStateData.blockedDistance
                    characteristicL[idx] += autoNetwork.diameter * 1609.34
                    V_0[idx] = autoNetwork.freeFlowSpeed
                    N_0[idx] = L_eff * autoNetwork.jamDensity
                    n_other[idx] = networkStateData.nonAutoAccumulation
                    n_init[idx] = networkStateData.initialAccumulation
        #            tripStartRate[idx] = microtype.getModeStartRate("auto") / 3600.

        X = np.transpose(self.transitionMatrix.matrix.values)

        dt = 0.02 * 3600.
        ts = np.arange(0, durationInHours * 3600., dt)
        ns = np.zeros((len(self), np.size(ts)))
        vs = np.zeros((len(self), np.size(ts)))
        n_t = n_init.copy()

        for i, ti in enumerate(ts):
            dn = dn_dt(n_t, tripStartRate, characteristicL, X, V_0, N_0, n_other) * dt
            n_t += dn
            n_t[n_t > (N_0 - n_other)] = N_0[n_t > (N_0 - n_other)]
            pct = n_t / N_0
            ns[:, i] = np.squeeze(n_t)
            vs[:, i] = np.squeeze(v(n_t, V_0, N_0, n_other))

        # self.transitionMatrix.setAverageSpeeds(np.mean(vs, axis=1))
        averageSpeeds = np.mean(vs, axis=1)
        print(averageSpeeds)
        if writeData:
            for microtypeID, microtype in self:
                idx = self.transitionMatrix.idx(microtypeID)
                for modes, autoNetwork in microtype.networks:
                    if "auto" in autoNetwork:
                        networkStateData = collectedNetworkStateData[(microtypeID, modes)]
                        networkStateData.finalAccumulation = ns[idx, -1]
                        networkStateData.finalSpeed = vs[idx, -1]
                        networkStateData.averageSpeed = averageSpeeds[idx]
        return {"t": np.transpose(ts), "v": np.transpose(vs), "n": np.transpose(ns), "v_av": averageSpeeds,
                "max_accumulation": N_0}

    def __iter__(self) -> (str, Microtype):
        return iter(self.__microtypes.items())

    def getModeSpeeds(self) -> dict:
        return {idx: m.getModeSpeeds() for idx, m in self}

    def getOperatorCosts(self) -> CollectedTotalOperatorCosts:
        operatorCosts = CollectedTotalOperatorCosts()
        for mID, microtype in self:
            assert isinstance(microtype, Microtype)
            operatorCosts[mID] = microtype.networks.getModeOperatingCosts()
        return operatorCosts

    def getStateData(self) -> CollectedNetworkStateData:
        data = CollectedNetworkStateData()
        for mID, microtype in self:
            data.addMicrotype(microtype)
        return data

    def importPreviousStateData(self, networkStateData: CollectedNetworkStateData):
        for mID, microtype in self:
            networkStateData.adoptPreviousMicrotypeState(microtype)

    def updateTransitionMatrix(self, transitionMatrix: TransitionMatrix):
        if self.transitionMatrix.names == transitionMatrix.names:
            self.transitionMatrix = transitionMatrix
        else:
            print("MICROTYPE NAMES IN TRANSITION MATRIX DON'T MATCH")

    def emptyTransitionMatrix(self):
        return TransitionMatrix(self.transitionMatrix.names)

    def filterAllocation(self, mode, inputAllocation: Allocation):
        validMicrotypes = self.__modeToMicrotype[mode]
        if inputAllocation.keys() == validMicrotypes:
            return inputAllocation.mapping
        else:
            return inputAllocation.filterAllocation(validMicrotypes)
