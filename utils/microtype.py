#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .network import Network, NetworkCollection, NetworkFlowParams, Costs, ModeParamFactory


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

    def addModeDemandForPMT(self, mode, demand, trip_distance):
        self.networks.demands.addModeThroughTrips(mode, demand, trip_distance)

    def setModeDemand(self, mode, demand, trip_distance):
        self.networks.demands.setSingleDemand(mode, demand, trip_distance)
        self.networks.updateModes()

    def resetDemand(self):
        self.networks.demands.resetDemand()

    def getStartAndEndRate(self, mode: str) -> (float, float):
        return self.networks.demands.getStartRate(mode), self.networks.demands.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self.networks.demands.getAverageDistance(mode)

    def getThroughTimeCostWait(self, mode: str, distanceInMiles: float) -> (float, float, float):
        speedMilesPerHour = np.max([self.getModeSpeed(mode), 0.01]) * 2.23694
        if np.isnan(speedMilesPerHour):
            speed = self.getModeSpeed("auto")
        timeInHours = distanceInMiles / speedMilesPerHour * self.networks.modes[mode].costs.vott_multiplier
        cost = distanceInMiles * self.networks.modes[mode].costs.per_meter * 1609.34
        wait = 0.
        return timeInHours, cost, wait

    def getStartTimeCostWait(self, mode: str) -> (float, float, float):
        time = 0.
        cost = self.networks.modes[mode].costs.per_start
        if mode == 'bus':
            wait = self.networks.modes['bus'].params.headway_in_sec / 3600. / 2.  # TODO: Make getter
        else:
            wait = 0.
        return time, cost, wait

    def getEndTimeCostWait(self, mode: str) -> (float, float, float):
        time = 0.
        cost = self.networks.modes[mode].costs.per_end
        wait = 0.
        return time, cost, wait

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
    def __init__(self, path: str):
        self.__microtypes = dict()
        self.path = path

    def __setitem__(self, key: str, value: Microtype):
        self.__microtypes[key] = value

    def __getitem__(self, item: str) -> Microtype:
        return self.__microtypes[item]

    def importMicrotypes(self, subNetworkData: pd.DataFrame, modeToSubNetworkData: pd.DataFrame):
        modeParamFactory = ModeParamFactory(self.path)
        for microtypeID, grouped in subNetworkData.groupby('MicrotypeID'):
            subNetworkToModes = dict()
            modeToModeParams = dict()
            costs = dict()
            allModes = set()
            for row in grouped.itertuples():
                joined = modeToSubNetworkData.loc[modeToSubNetworkData['SubnetworkID'] == row.SubnetworkID]
                subNetwork = Network(row.Length, NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50))
                for n in joined.itertuples():
                    subNetworkToModes.setdefault(subNetwork, []).append(n.ModeTypeID.lower())
                    allModes.add(n.ModeTypeID.lower())
            for mode in allModes:
                (modeToModeParams[mode], costs[mode]) = modeParamFactory.get(mode, microtypeID)
            networkCollection = NetworkCollection(subNetworkToModes, modeToModeParams)
            self[microtypeID] = Microtype(microtypeID, networkCollection, costs)

    def __iter__(self) -> (str, Microtype):
        return iter(self.__microtypes.items())
