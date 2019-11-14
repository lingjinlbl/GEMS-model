#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy

from utils.Network import Network


class ModeParams:
    def __init__(self, mean_trip_distance, road_network_fraction=1.0, relative_length=1.0):
        self.mean_trip_distance = mean_trip_distance
        self.road_network_fraction = road_network_fraction
        self.size = relative_length

    def getSize(self):
        return self.size

    def getFixedDensity(self):
        return None


class BusParams(ModeParams):
    def __init__(self, mean_trip_distance: float, road_network_fraction: float, relative_length: float,
                 fixed_density: float, min_stop_time: float,
                 stop_spacing: float, passenger_wait: float) -> None:
        super().__init__(mean_trip_distance, road_network_fraction, relative_length)
        self.k = fixed_density
        self.t_0 = min_stop_time
        self.s_b = stop_spacing
        self.gamma_s = passenger_wait

    def getFixedDensity(self):
        return self.k


class SupplyCharacteristics:
    def __init__(self, density, N_eq, L_eq):
        self.density = density
        self.N_eq = N_eq
        self.L_eq = L_eq

    def getN(self):
        return self.N_eq

    def getL(self):
        return self.L_eq


class DemandCharacteristics:
    def __init__(self, speed, passenger_flow):
        self.speed = speed
        self.passenger_flow = passenger_flow

    def getSpeed(self):
        return self.speed

    def __str__(self):
        return 'Speed: ' + str(self.speed) + ' , Flow: ' + str(self.passenger_flow)

class TravelDemand:
    def __init__(self, modes):
        self._modes = modes
        self._tripStartRate = dict()
        self._tripEndRate = dict()
        self._rateOfPMT = dict()
        for mode in modes:
            self._tripStartRate[mode] = 0.0
            self._tripEndRate[mode] = 0.0
            self._rateOfPMT[mode] = 0.0

    def setEndRate(self, mode: str, rate: float):
        self._tripEndRate[mode] = rate

    def setStartRate(self, mode: str, rate: float):
        self._tripStartRate[mode] = rate

    def getEndRate(self, mode: str):
        return self._tripEndRate[mode]

    def getStartRate(self, mode: str):
        return self._tripStartRate[mode]

    def getRateOfPMT(self, mode: str):
        return self._rateOfPMT[mode]

    def setSingleDemand(self, mode, demand: float, trip_distance: float):
        self._tripStartRate[mode] = demand
        self._tripEndRate[mode] = demand
        self._rateOfPMT[mode] = demand * trip_distance


class BusDemandCharacteristics(DemandCharacteristics):
    def __init__(self, speed, passenger_flow, dwell_time, headway, occupancy):
        super().__init__(speed, passenger_flow)
        self.dwell_time = dwell_time
        self.headway = headway
        self.occupancy = occupancy


class ModeCharacteristics:
    def __init__(self, mode_name: str, params: ModeParams, demand: float):
        self.mode_name = mode_name
        self.params = params
        self.demand_characteristics = getDefaultDemandCharacteristics(mode_name)
        self.supply_characteristics = getDefaultSupplyCharacteristics()
        self.demand = demand

    def __str__(self):
        return self.mode_name.upper() + ': ' + str(self.demand_characteristics)

    def setSupplyCharacteristics(self, supply_characteristics: SupplyCharacteristics):
        self.supply_characteristics = supply_characteristics

    def setDemandCharacteristics(self, demand_characteristics: DemandCharacteristics):
        self.demand_characteristics = demand_characteristics

    def getSpeed(self):
        return self.demand_characteristics.speed

    def getFlow(self):
        return self.demand_characteristics.passenger_flow


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

    def setModeDemand(self, mode: str, new_demand: float):
        self._data[mode].demand = new_demand

    def addModeDemand(self, mode: str, demand: float):
        self._data[mode].demand += demand

    def getModeSpeed(self, mode: str) -> float:
        return self._data[mode].demand_characteristics.passenger_flow

class Microtype:
    def __init__(self, network_params: Network, mode_characteristics: CollectedModeCharacteristics):
        self.modes = mode_characteristics.getModes()
        self.network_params = network_params
        self._baseSpeed = network_params.getBaseSpeed()
        self._mode_characteristics = mode_characteristics
        self._travel_demand = TravelDemand(self.modes)
        self.updateSupplyCharacteristics()
        self.updateDemandCharacteristics()

    def getModeSpeed(self, mode) -> float:
        return self.getModeCharacteristics(mode).demand_characteristics.getSpeed()

    def getModeFlow(self, mode) -> float:
        return self.getModeCharacteristics(mode).demand_characteristics.passenger_flow

    def getModeDemand(self, mode):
        return self.getModeCharacteristics(mode).demand

    def addModeDemand(self, mode, demand):
        self._mode_characteristics.addModeDemand(mode, demand)

    def setModeDemand(self, mode, demand, trip_distance):
        self._travel_demand.setSingleDemand(mode, demand, trip_distance)

    def getModeCharacteristics(self, mode: str) -> ModeCharacteristics:
        return self._mode_characteristics[mode]

    def getModeMeanDistance(self, mode: str):
        return self.getModeCharacteristics(mode).params.mean_trip_distance

    def setModeSupplyCharacteristics(self, mode: str, supply_characteristics: SupplyCharacteristics):
        self.getModeCharacteristics(mode).setSupplyCharacteristics(supply_characteristics)

    def setModeDemandCharacteristics(self, mode: str, demand_characteristics: DemandCharacteristics):
        self.getModeCharacteristics(mode).setDemandCharacteristics(demand_characteristics)

    def getModeDensity(self, mode):
        mc = self.getModeCharacteristics(mode)
        fixed_density = mc.params.getFixedDensity()
        littles_law_density = self._travel_demand.getRateOfPMT(mode) / mc.demand_characteristics.getSpeed()
        return fixed_density or littles_law_density

    def updateDemandCharacteristics(self):
        for mode in self.modes:
            self.setModeDemandCharacteristics(mode,
                                              copy.deepcopy(getModeDemandCharacteristics(self._baseSpeed, mode,
                                                                                                self.getModeCharacteristics(
                                                                                                    mode))))

    def updateSupplyCharacteristics(self):
        for mode in self.modes:
            density = self.getModeDensity(mode)
            L_eq = getModeBlockedDistance(self, mode)
            N_eq = (self.getModeCharacteristics(mode).params.size or 1.0) * density
            supplyCharacteristics = SupplyCharacteristics(density, N_eq, L_eq)
            self.setModeSupplyCharacteristics(mode, supplyCharacteristics)

    def getNewSpeedFromDensities(self):
        N_eq = np.sum([self.getModeCharacteristics(mode).supply_characteristics.getN() for mode in self.modes])
        L_eq = self.network_params.L - np.sum(
            [self.getModeCharacteristics(mode).supply_characteristics.getL() for mode in self.modes])
        return self.network_params.MFD(N_eq, L_eq)

    def setSpeed(self, speed):
        self._baseSpeed = speed
        self.updateDemandCharacteristics()

    def findEquilibriumDensityAndSpeed(self):
        newData = copy.deepcopy(self)
        oldData = copy.deepcopy(self)
        keepGoing = True
        ii = 0
        while keepGoing:
            newSpeed = newData.getNewSpeedFromDensities()
            #print(str(newData._mode_characteristics))
            #print('New Speed: ', newSpeed)
            newData.setSpeed(newSpeed)
            newData.updateSupplyCharacteristics()
            keepGoing = (np.abs(newData._baseSpeed - oldData._baseSpeed) > 0.001) & (ii < 20)
            oldData = copy.deepcopy(newData)
            if ii == 20:
                newSpeed = 0.0
        self.setSpeed(newSpeed)

    def getFlows(self):
        return [np.nan_to_num(np.max([self.getModeFlow(mode), 0.0])) for mode in
                self.modes]

    def getSpeeds(self):
        return [self.getModeSpeed(mode) for mode in self.modes]

    def getDemands(self):
        return [self.getModeDemand(mode) for mode in self.modes]

    def getTravelTimes(self):
        speeds = np.array(self.getSpeeds())
        speeds[~(speeds > 0)] = np.nan
        distances = np.array([self.getModeMeanDistance(mode) for mode in self.modes])
        return distances / speeds

    def getTotalTimes(self):
        tts = self.getTravelTimes()
        demands = self.getDemands()
        return np.array(tts) * np.array(demands)

    def print(self):
        print('------------')
        print('Modes:')
        print(self.modes)
        print('Supply Characteristics:')
        print(self._modeSupplyCharacteristics)
        print('Demand Characteristics:')
        print(self._modeDemandCharacteristics)
        print('Demand Density:')
        print(self._demands)
        print('------------')


def main():
    network_params_default = {'lambda': 0.068,
                              'u_f': 15.42,
                              'w': 1.88,
                              'kappa': 0.145,
                              'Q': 0.177,
                              'L': 100,
                              'l': 50}
    bus_params_default = {'k': 1. / 100.,
                          't_0': 10,
                          's_b': 1. / 250.,
                          'gamma_s': 5.,
                          'size': 3.0,
                          'meanTripDistance': 1000,
                          'L_mode': 25
                          }
    car_params_default = {'meanTripDistance': 1000, 'size': 1.0}
    modes = {'car', 'bus'}
    mode_params_default = {'car': car_params_default, 'bus': bus_params_default}
    demands = {'car': 5. / (10 * 60), 'bus': 1. / (100 * 60)}
    m = Microtype(modes, mode_params_default, network_params_default, demands)
    m.print()




def getDefaultDemandCharacteristics(mode):
    """

    :param mode: str
    :return: DemandCharacteristics
    """
    if mode == 'car':
        return DemandCharacteristics(15., 0.0)
    elif mode == 'bus':
        return BusDemandCharacteristics(15., 0.0, 0.0, 0.0, 0.0)
    else:
        return DemandCharacteristics(15., 0.0)


def getDefaultSupplyCharacteristics():
    return SupplyCharacteristics(0.0, 0.0, 0.0)

def getBusdwellTime(v, params_bus, modeDemand):
    if v > 0:
        out = 1. / (params_bus.s_b * v) * (
                v * params_bus.k * params_bus.t_0 * params_bus.s_b +
                params_bus.gamma_s * 2 * modeDemand) / (
                      params_bus.k - params_bus.gamma_s * 2 * modeDemand)
    else:
        out = np.nan
    return out


def getModeDemandCharacteristics(baseSpeed, mode, modeCharacteristics: ModeCharacteristics):
    """

    :param modeCharacteristics:
    :param baseSpeed: float
    :type mode: str
    :return: DemandCharacteristics
    """
    modeParams = modeCharacteristics.params
    modeDemand = modeCharacteristics.demand
    if mode == 'car':
        return DemandCharacteristics(baseSpeed, modeDemand * modeParams.mean_trip_distance)
    elif mode == 'bus':
        assert (isinstance(modeParams, BusParams))
        dwellTime = getBusdwellTime(baseSpeed, modeParams, modeDemand)
        if dwellTime > 0:
            speed = baseSpeed / (1 + dwellTime * baseSpeed * modeParams.s_b)
            headway = modeParams.road_network_fraction / speed
        else:
            speed = 0.0
            headway = np.nan

        if (dwellTime > 0) & (baseSpeed > 0):
            passengerFlow: float = modeDemand * modeParams.mean_trip_distance
            occupancy: float = passengerFlow / modeParams.k / speed
        else:
            passengerFlow: float = 0.0
            occupancy: float = np.nan

        return BusDemandCharacteristics(speed, passengerFlow, dwellTime, headway, occupancy)

    else:
        return DemandCharacteristics(baseSpeed, modeDemand * modeParams.mean_trip_distance)


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
        modeDemand = microtype.getModeDemand(mode)
        dwellTime = getBusdwellTime(microtype._baseSpeed, modeParams, modeDemand)
        return microtype.network_params.l * modeParams.road_network_fraction * modeParams.s_b * modeParams.k * dwellTime * modeSpeed /microtype.network_params.L
    else:
        return 0.0




if __name__ == "__main__":
    main()
