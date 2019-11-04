#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import utils.supply as supply


# def


def getDefaultDemandCharacteristics(mode, network_params):
    if mode == 'car':
        return {'speed': network_params['u_f'], 'passengerFlow': 0}
    elif mode == 'bus':
        return {'speed': network_params['u_f'],
                'dwellTime': 0,
                'headway': 0,
                'occupancy': 0,
                'passengerFlow': 0}
    else:
        return {'speed': network_params['u_f'], 'passengerFlow': 0}


def getDefaultSupplyCharacteristics():
    return {'density': 0.0, 'N_eq': 0.0, 'L_eq': 0.0}


def giveAverageDensities(d1, d2):
    return {mode: (d1[mode] + d2[mode]) / 2. for mode in d1.keys()}


class Microtype:
    def __init__(self, modes, mode_params, network_params, demands):
        self.modes = modes
        self.network_params = network_params
        self.mode_params = mode_params
        self._baseSpeed = network_params['u_f']
        self._modeDemandCharacteristics = {mode: getDefaultDemandCharacteristics(mode, network_params) for mode in
                                           modes}
        self._modeSupplyCharacteristics = {mode: getDefaultSupplyCharacteristics() for mode in modes}
        self._demands = {mode: 0.0 for mode in modes}
        self.setDemand(demands)

    def copy(self):
        out = Microtype(self.modes, self.mode_params, self.network_params, self._demands)
        out.setSpeed(self._baseSpeed)
        return out

    def updateDemandCharacteristics(self):
        for mode in self.modes:
            self._modeDemandCharacteristics[mode] = supply.getModeDemandCharacteristics(self._baseSpeed, mode,
                                                                                        self.mode_params[mode],
                                                                                        self._demands[mode])

    def updateSupplyCharacteristics(self):
        for mode in self.modes:
            k = self.mode_params[mode].get('k')
            littlesLawDensity = self._demands[mode] * self.mode_params[mode].get('meanTripDistance') / (
                    self._modeDemandCharacteristics[mode].get('speed') or np.nan)
            density = k or littlesLawDensity
            L_eq = supply.getModeBlockedDistance(self, mode)
            N_eq = (self.mode_params[mode].get('size') or 1.0) * density
            supplyCharacteristics = {'density': density, 'N_eq': N_eq, 'L_eq': L_eq}
            self._modeSupplyCharacteristics[mode] = supplyCharacteristics

    def getNewSpeedFromDensities(self):
        N_eq = np.sum([self._modeSupplyCharacteristics[mode].get('N_eq') for mode in self.modes])
        L_eq = self.network_params['L'] - np.sum(
            [self._modeSupplyCharacteristics[mode].get('N_eq') for mode in self.modes])
        print('MFD: ', N_eq, L_eq)
        return supply.MFD(N_eq, L_eq, self.network_params)

    def setDemand(self, demands):
        self._demands = demands
        self.updateSupplyCharacteristics()
        self.updateDemandCharacteristics()

    def setSpeed(self, speed):
        self._baseSpeed = speed
        self.updateDemandCharacteristics()

    def findEquilibriumDensityAndSpeed(self):
        newData = self.copy()
        oldData = self.copy()
        keepGoing = True
        ii = 0
        while keepGoing:
            newSpeed = newData.getNewSpeedFromDensities()
            print(newData._modeDemandCharacteristics)
            print(newData._modeSupplyCharacteristics)
            print('New Speed: ', newSpeed)
            newData.setSpeed(newSpeed)
            newData.updateSupplyCharacteristics()
            keepGoing = (np.abs(newData._baseSpeed - oldData._baseSpeed) > 0.001) & (ii < 20)
            oldData = newData.copy()
            if ii == 20:
                newSpeed = 0.0
        self.setSpeed(newSpeed)

    def getFlows(self):
        return [np.nan_to_num(np.max([self._modeDemandCharacteristics[mode].get('passengerFlow'), 0.0])) for mode in
                self.modes]

    def getSpeeds(self):
        return [self._modeDemandCharacteristics[mode].get('speed') for mode in self.modes]

    def getDemands(self):
        return [self._demands[mode] for mode in self.modes]

    def getTravelTimes(self):
        speeds = self.getSpeeds()
        distances = [self.mode_params[mode].get('meanTripDistance') for mode in self.modes]
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
