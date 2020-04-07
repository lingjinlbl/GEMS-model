#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy

from utils.network import Network, NetworkCollection, NetworkFlowParams, Mode, BusMode, BusModeParams, Costs
from utils.supply import DemandCharacteristics, BusDemandCharacteristics, TravelDemand, ModeParams, BusParams, \
    TravelDemands
import utils.supply as supply


class Microtype:
    def __init__(self, networks: NetworkCollection, costs=None):
        if costs is None:
            costs = dict()
        self.mode_names = list(networks.getModeNames())
        self.networks = networks
        self.costs = costs
        # self.updateDemandCharacteristics()

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
        time = distance / speed * self.costs[mode].vott_multiplier
        cost = distance * self.costs[mode].per_meter
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
        cost = self.costs[mode].per_end
        wait = 0.
        return time, cost, wait

    def getFlows(self):
        return [mode.getPassengerFlow() for mode in
                self.networks.modes.values()]

    def getSpeeds(self):
        return [mode.getSpeed() for mode in
                self.networks.modes.values()]

    def getDemandsForPMT(self):
        return [mode.getPassengerFlow() for mode in
                self.networks.modes.values()]

    def getTravelTimes(self):
        speeds = np.array(self.getSpeeds())
        speeds[~(speeds > 0)] = np.nan
        distances = np.array([self.getModeMeanDistance(mode) for mode in self.modes])
        return distances / speeds

    def getTotalTimes(self):
        speeds = self.getSpeeds()
        demands = self.getDemandsForPMT()
        return np.array(speeds) * np.array(demands)

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
    network_params_default = Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
    bus_params_default = BusParams(road_network_fraction=1000, relative_length=3.0,
                                   fixed_density=150. / 100., min_stop_time=15., stop_spacing=1. / 500.,
                                   passenger_wait=5.)

    car_params_default = ModeParams(relative_length=1.0)

    modeCharacteristics = CollectedModeCharacteristics()
    modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default)
    modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params_default)

    m = Microtype(network_params_default, modeCharacteristics)
    m.setModeDemand('car', 70 / (10 * 60), 1000.0)
    m.setModeDemand('bus', 10 / (10 * 60), 1000.0)
    m.print()


def getDefaultDemandCharacteristics(mode):
    """

    :param mode: str
    :return: DemandCharacteristics
    """
    if mode == 'car':
        return supply.DemandCharacteristics(15., 0.0)
    elif mode == 'bus':
        return supply.BusDemandCharacteristics(15., 0.0, 0.0, 0.0, 0.0)
    else:
        return supply.DemandCharacteristics(15., 0.0)


def getDefaultSupplyCharacteristics():
    return supply.SupplyCharacteristics(0.0, 0.0, 0.0)


def getBusdwellTime(v, params_bus, trip_start_rate, trip_end_rate):
    if v > 0:
        out = 1. / (params_bus.s_b * v) * (
                v * params_bus.k * params_bus.t_0 * params_bus.s_b +
                params_bus.gamma_s * 2 * (trip_start_rate + trip_end_rate)) / (
                      params_bus.k - params_bus.gamma_s * (trip_start_rate + trip_end_rate))
    else:
        out = np.nan
    return out

if __name__ == "__main__":
    main()
