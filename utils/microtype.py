#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import utils.supply as supply
import utils.IO as io
import copy


class Microtype:
    def __init__(self, network_params: io.Network, mode_characteristics: io.CollectedModeCharacteristics):
        self.modes = mode_characteristics.getModes()
        self.network_params = network_params
        self._baseSpeed = network_params.getBaseSpeed()
        self._mode_characteristics = mode_characteristics
        self.updateSupplyCharacteristics()
        self.updateDemandCharacteristics()

    def getModeSpeed(self, mode) -> float:
        return self.getModeCharacteristics(mode).demand_characteristics.getSpeed()

    def getModeFlow(self, mode) -> float:
        return self.getModeCharacteristics(mode).demand_characteristics.passenger_flow

    def getModeDemand(self, mode):
        return self.getModeCharacteristics(mode).demand

    def getModeCharacteristics(self, mode: str) -> io.ModeCharacteristics:
        return self._mode_characteristics[mode]

    def getModeMeanDistance(self, mode: str):
        return self.getModeCharacteristics(mode).params.mean_trip_distance

    def setModeSupplyCharacteristics(self, mode: str, supply_characteristics: io.SupplyCharacteristics):
        self.getModeCharacteristics(mode).setSupplyCharacteristics(supply_characteristics)

    def setModeDemandCharacteristics(self, mode: str, demand_characteristics: io.DemandCharacteristics):
        self.getModeCharacteristics(mode).setDemandCharacteristics(demand_characteristics)

    def getModeDensity(self, mode):
        mc = self.getModeCharacteristics(mode)
        fixed_density = mc.params.getFixedDensity()
        littles_law_density = mc.demand * mc.params.mean_trip_distance / mc.demand_characteristics.getSpeed()
        return fixed_density or littles_law_density

    def updateDemandCharacteristics(self):
        for mode in self.modes:
            self.setModeDemandCharacteristics(mode,
                                              copy.deepcopy(supply.getModeDemandCharacteristics(self._baseSpeed, mode,
                                                                                                self.getModeCharacteristics(
                                                                                                    mode))))

    def updateSupplyCharacteristics(self):
        for mode in self.modes:
            density = self.getModeDensity(mode)
            L_eq = supply.getModeBlockedDistance(self, mode)
            N_eq = (self.getModeCharacteristics(mode).params.size or 1.0) * density
            supplyCharacteristics = io.SupplyCharacteristics(density, N_eq, L_eq)
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
            print(str(newData._mode_characteristics))
            print('New Speed: ', newSpeed)
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
        speeds = self.getSpeeds()
        distances = [self.getModeMeanDistance(mode) for mode in self.modes]
        return np.array(distances) / np.array(speeds)

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


if __name__ == "__main__":
    main()
