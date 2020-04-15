#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy

from utils.network import Network, NetworkCollection, NetworkFlowParams, Mode, BusMode, BusModeParams, Costs


class Microtype:
    def __init__(self, networks: NetworkCollection, costs=None):
        if costs is None:
            costs = dict()
        self.mode_names = list(networks.getModeNames())
        self.networks = networks
        self.updateModeCosts(costs)

    def updateModeCosts(self, costs):
        for (mode, modeCosts) in costs.items():
            assert(isinstance(mode, str) and isinstance(modeCosts, Costs))
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
        self.networks.demands.addModePMT(mode, demand, trip_distance)

    def setModeDemand(self, mode, demand, trip_distance):
        self.networks.demands.setSingleDemand(mode, demand, trip_distance)
        self.networks.updateModes()

    def resetDemand(self):
        self.networks.demands.resetDemand()

    def getStartAndEndRate(self, mode: str) -> (float, float):
        return self.networks.demands.getStartRate(mode), self.networks.demands.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self.networks.demands.getAverageDistance(mode)

    def getThroughTimeCostWait(self, mode: str, distance: float) -> (float, float, float):
        speed = np.max([self.getModeSpeed(mode), 0.01])
        time = distance / speed * self.networks.modes[mode].costs.vott_multiplier
        cost = distance * self.networks.modes[mode].costs.per_meter
        wait = 0.
        return time, cost, wait

    def getStartTimeCostWait(self, mode: str) -> (float, float, float):
        time = 0.
        cost = self.networks.modes[mode].costs.per_start
        if mode == 'bus':
            wait = self.networks.modes['bus'].getHeadway() / 2.
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

    def getPassengerOccupancy(self):
        return [self.getModeOccupancy(mode) for mode in self.modes]

    def getTravelTimes(self):
        speeds = np.array(self.getSpeeds())
        speeds[~(speeds > 0)] = np.nan
        distances = np.array([self.getModeMeanDistance(mode) for mode in self.modes])
        return distances / speeds

    def getTotalTimes(self):
        speeds = np.array(self.getSpeeds())
        demands = np.array(self.getDemandsForPMT())
        times = speeds * demands
        times[speeds == 0.] = np.inf
        return times

    def print(self):
        print('------------')
        print('Modes:')
        print(self.modes)
        print('Supply Characteristics:')
        print(self._mode_characteristics)
        print('Demand Characteristics:')
        print(self._travel_demand)
        print('------------')

    def __str__(self):
        return 'Demand: ' + str(self._travel_demand) + ' , Speed: ' + str(self._baseSpeed)


def main():
    network_params_mixed = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_params_car = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_params_bus = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
    network_car = Network(250, network_params_car)
    network_bus = Network(750, network_params_bus)
    network_mixed = Network(500, network_params_mixed)

    Mode([network_mixed, network_car], 'car')
    BusMode([network_mixed, network_bus], BusModeParams(0.6))
    nc = NetworkCollection([network_mixed, network_car, network_bus])

    m = Microtype(nc)
    m.setModeDemand('car', 40 / (10 * 60), 1000.0)
    m.setModeDemand('bus', 2 / (10 * 60), 1000.0)



if __name__ == "__main__":
    main()
