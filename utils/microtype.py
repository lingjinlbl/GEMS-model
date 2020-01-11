#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy

from utils.Network import Network
from utils.supply import DemandCharacteristics, BusDemandCharacteristics, TravelDemand, ModeParams, BusParams
import utils.supply as supply


class Costs:
    def __init__(self, per_meter, per_start, per_end, vott_multiplier):
        self.per_meter = per_meter
        self.per_start = per_start
        self.per_end = per_end
        self.vott_multiplier = vott_multiplier


class ModeCharacteristics:
    def __init__(self, mode_name: str, params: supply.ModeParams, demand: float = 0.0):
        self.mode_name = mode_name
        self.params = params
        self.demand_characteristics = getDefaultDemandCharacteristics(mode_name)
        self.supply_characteristics = getDefaultSupplyCharacteristics()
        self.demand = demand

    def __str__(self):
        return self.mode_name.upper() + ': ' + str(self.demand_characteristics) + ', ' + str(
            self.supply_characteristics)

    def setSupplyCharacteristics(self, supply_characteristics: supply.SupplyCharacteristics):
        self.supply_characteristics = supply_characteristics

    def setDemandCharacteristics(self, demand_characteristics: supply.DemandCharacteristics):
        self.demand_characteristics = demand_characteristics

    def getSpeed(self):
        return self.demand_characteristics.speed

    def getFlow(self):
        return self.demand_characteristics.passenger_flow

    def getPassengerOccupancy(self):
        return self.demand_characteristics.passenger_flow / self.demand_characteristics.speed

    def __add__(self, other):
        if isinstance(other, CollectedModeCharacteristics):
            other[self.mode_name] = self
            return other
        elif isinstance(other, ModeCharacteristics):
            out = CollectedModeCharacteristics()
            out += self
            out += other
            return out


class CollectedModeCharacteristics:
    def __init__(self):
        self._data = dict()

    def __setitem__(self, mode_name: str, mode_info: ModeCharacteristics):
        self._data[mode_name] = mode_info

    def __getitem__(self, mode_name) -> ModeCharacteristics:
        return self._data[mode_name]

    def getModes(self):
        return list(self._data.keys())

    def __str__(self):
        return str([str(self._data[key]) for key in self._data])

    #    def setModeDemand(self, mode: str, new_demand: float):
    #        self._data[mode].demand = new_demand

    def addModeDemand(self, mode: str, demand: float):
        self._data[mode].demand += demand

    def __iadd__(self, other):
        if isinstance(other, ModeCharacteristics):
            self[other.mode_name] = other
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

    def __add__(self, other):
        if isinstance(other, ModeCharacteristics):
            self[other.mode_name] = other
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self


#    def getModeSpeed(self, mode: str) -> float:
#        return self._data[mode].demand_characteristics.passenger_flow

class Microtype:
    def __init__(self, network_params: Network, mode_characteristics: CollectedModeCharacteristics,
                 costs=None):
        mode_characteristics = copy.deepcopy(mode_characteristics)
        if costs is None:
            costs = dict()
        self.modes = mode_characteristics.getModes()
        self.network_params = network_params
        self._baseSpeed = network_params.getBaseSpeed()
        self._mode_characteristics = mode_characteristics
        self._travel_demand = TravelDemand(self.modes)
        self.costs = costs
        self.updateSupplyCharacteristics()
        self.updateDemandCharacteristics()

    def getModeSpeed(self, mode) -> float:
        return self.getModeCharacteristics(mode).demand_characteristics.getSpeed()

    def getModeOccupancy(self, mode) -> float:
        return self.getModeCharacteristics(mode).getPassengerOccupancy()

    def getBaseSpeed(self) -> float:
        return self._baseSpeed

    def getModeFlow(self, mode) -> float:
        return self._travel_demand.getRateOfPMT(mode)

    def getModeDemandForPMT(self, mode):
        return self._travel_demand.getRateOfPMT(mode)

    def addModeStarts(self, mode, demand):
        self._travel_demand.addModeStarts(mode, demand)

    def addModeEnds(self, mode, demand):
        self._travel_demand.addModeEnds(mode, demand)

    def addModeDemandForPMT(self, mode, demand, trip_distance):
        self._travel_demand.addModePMT(mode, demand, trip_distance)

    def setModeDemand(self, mode, demand, trip_distance):
        self._travel_demand.setSingleDemand(mode, demand, trip_distance)

    def resetDemand(self):
        self._travel_demand.resetDemand()

    def getModeCharacteristics(self, mode: str) -> ModeCharacteristics:
        return self._mode_characteristics[mode]

    def getStartAndEndRate(self, mode: str) -> (float, float):
        return self._travel_demand.getStartRate(mode), self._travel_demand.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self._travel_demand.getAverageDistance(mode)

    def setModeSupplyCharacteristics(self, mode: str, supply_characteristics: supply.SupplyCharacteristics):
        self.getModeCharacteristics(mode).setSupplyCharacteristics(supply_characteristics)

    def setModeDemandCharacteristics(self, mode: str, demand_characteristics: supply.DemandCharacteristics):
        self.getModeCharacteristics(mode).setDemandCharacteristics(demand_characteristics)

    def getThroughTimeCostWait(self, mode: str, distance: float) -> (float, float, float):
        speed = np.max([self.getModeSpeed(mode) , 0.01])
        time = distance / speed * self.costs[mode].vott_multiplier
        cost = distance * self.costs[mode].per_meter
        wait = 0.
        return time, cost, wait

    def getStartTimeCostWait(self, mode: str) -> (float, float, float):
        time = 0.
        cost = self.costs[mode].per_start
        if mode == 'bus':
            wait = self.getModeCharacteristics('bus').demand_characteristics.headway / 2.
        else:
            wait = 0.
        return time, cost, wait

    def getEndTimeCostWait(self, mode: str) -> (float, float, float):
        time = 0.
        cost = self.costs[mode].per_end
        wait = 0.
        return time, cost, wait

    def getModeDensity(self, mode):
        mc = self.getModeCharacteristics(mode)
        fixed_density = mc.params.getFixedDensity()
        mode_speed = mc.demand_characteristics.getSpeed()
        if mode_speed > 0:
            littles_law_density = self._travel_demand.getRateOfPMT(mode) / mode_speed
        else:
            littles_law_density = np.nan
        return fixed_density or littles_law_density

    def updateDemandCharacteristics(self):
        for mode in self.modes:
            self.setModeDemandCharacteristics(mode,
                                              copy.deepcopy(getModeDemandCharacteristics(self._baseSpeed,
                                                                                         self.getModeCharacteristics(
                                                                                             mode),
                                                                                         self._travel_demand)))

    def updateSupplyCharacteristics(self):
        for mode in self.modes:
            density = self.getModeDensity(mode)
            L_eq = getModeBlockedDistance(self, mode)
            N_eq = (self.getModeCharacteristics(mode).params.size or 1.0) * density
            supplyCharacteristics = supply.SupplyCharacteristics(density, N_eq, L_eq)
            self.setModeSupplyCharacteristics(mode, supplyCharacteristics)

    def getNewSpeedFromDensities(self):
        N_eq = np.sum([self.getModeCharacteristics(mode).supply_characteristics.getN() for mode in self.modes])
        L_eq = self.network_params.L - np.sum(
            [self.getModeCharacteristics(mode).supply_characteristics.getL() for mode in self.modes])
        return self.network_params.MFD(N_eq, L_eq)

    def setSpeed(self, speed):
        self._baseSpeed = speed
        self.updateDemandCharacteristics()
        self.updateSupplyCharacteristics()

    def findEquilibriumDensityAndSpeed(self, iter_max=20):
        newData = copy.deepcopy(self)
        oldData = copy.deepcopy(self)
        keepGoing = True
        ii = 1
        while keepGoing:
            newSpeed = newData.getNewSpeedFromDensities()
            if np.isnan(newSpeed):
                newSpeed = 0.0
            print('New Speed: ', newSpeed)
            newData.setSpeed(newSpeed)
            print('Diff: ', np.abs(newData._baseSpeed - oldData._baseSpeed))
            keepGoing = (np.abs(newData._baseSpeed - oldData._baseSpeed) > 0.001) & (ii < iter_max)
            oldData = copy.deepcopy(newData)
            if ii == 20:
                newSpeed = 0.0
        self.setSpeed(newSpeed)

    def getFlows(self):
        return [np.nan_to_num(np.max([self.getModeFlow(mode), 0.0])) for mode in
                self.modes]

    def getSpeeds(self):
        return [self.getModeSpeed(mode) for mode in self.modes]

    def getDemandsForPMT(self):
        return [self.getModeDemandForPMT(mode) for mode in self.modes]

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


def getBusDwellTime(v, params_bus, trip_start_rate, trip_end_rate):
    if v > 0:
        out = 1. / (params_bus.s_b * v) * (
                v * params_bus.k * params_bus.t_0 * params_bus.s_b +
                params_bus.gamma_s * 2 * (trip_start_rate + trip_end_rate)) / (
                      params_bus.k - params_bus.gamma_s * (trip_start_rate + trip_end_rate))
    else:
        out = np.nan
    return out


def getModeDemandCharacteristics(base_speed: float, mode_characteristics: ModeCharacteristics, td: TravelDemand):
    mode = mode_characteristics.mode_name
    mode_params = mode_characteristics.params
    if mode == 'car':
        return DemandCharacteristics(base_speed, td.getRateOfPMT(mode))
    elif mode == 'bus':
        assert (isinstance(mode_params, BusParams))
        dwellTime = getBusDwellTime(base_speed, mode_params, td.getStartRate(mode), td.getEndRate(mode))
        if dwellTime > 0:
            speed = base_speed / (1 + dwellTime * base_speed * mode_params.s_b)
            headway = mode_params.road_network_fraction / speed
        else:
            speed = 0.0
            headway = 60*60

        if (dwellTime > 0) & (base_speed > 0):
            passengerFlow: float = td.getRateOfPMT(mode)
            occupancy: float = passengerFlow / mode_params.k / speed
        else:
            passengerFlow: float = 0.0
            occupancy: float = np.nan

        return BusDemandCharacteristics(speed, passengerFlow, dwellTime, headway, occupancy)

    else:
        return DemandCharacteristics(base_speed, td.getRateOfPMT(mode))


def getModeBlockedDistance(microtype, mode):
    """

    :rtype: float
    :param microtype: Microtype
    :param mode: str
    :return: float
    """
    if mode == 'car':
        return 0.0
    elif mode == 'bus':
        modeParams = microtype.getModeCharacteristics(mode).params
        modeSpeed = microtype.getModeSpeed(mode)
        trip_start_rate, trip_end_rate = microtype.getStartAndEndRate(mode)
        dwellTime = getBusDwellTime(microtype.getBaseSpeed(), modeParams, trip_start_rate, trip_end_rate)
        return microtype.network_params.l * modeParams.road_network_fraction * modeParams.s_b * modeParams.k * dwellTime * modeSpeed / microtype.network_params.L
    else:
        return 0.0


if __name__ == "__main__":
    main()
