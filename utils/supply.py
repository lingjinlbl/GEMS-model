# -*- coding: utf-8 -*-


class TravelDemand:
    def __init__(self):
        self.tripStartRate = 0.0
        self.tripEndRate = 0.0
        self.rateOfPMT = 0.0
        self.averageDistanceInSystem = 100.0

    def reset(self):
        self.tripStartRate = 0.0
        self.tripEndRate = 0.0
        self.rateOfPMT = 0.0
        self.averageDistanceInSystem = 100.0


class TravelDemands:
    def __init__(self, modes: list):
        self._modes = modes
        self._demands = dict()
        for mode in modes:
            self._demands[mode] = TravelDemand()

    def __setitem__(self, key: str, value: TravelDemand):
        self._demands[key] = value
        if key not in self._modes:
            self._modes.append(key)

    def __getitem__(self, item: str) -> TravelDemand:
        return self._demands[item]

    def setEndRate(self, mode: str, rate: float):
        self._demands[mode].tripEndRate = rate

    def setStartRate(self, mode: str, rate: float):
        self._demands[mode].tripStartRate = rate

    def getEndRate(self, mode: str):
        return self._demands[mode].tripEndRate

    def getStartRate(self, mode: str):
        return self._demands[mode].tripStartRate

    def getRateOfPMT(self, mode: str):
        return self._demands[mode].rateOfPMT

    def getAverageDistance(self, mode: str):
        return self._demands[mode].averageDistanceInSystem

    def resetDemand(self):
        for mode in self._modes:
            self._demands[mode].reset()

    def setSingleDemand(self, mode, demand: float, trip_distance: float):
        self._demands[mode].tripStartRate = demand
        self._demands[mode].tripEndRate = demand
        self._demands[mode].rateOfPMT = demand * trip_distance
        self._demands[mode].averageDistanceInSystem = trip_distance

    def addSingleDemand(self, mode, demand: float, trip_distance: float):
        self._demands[mode].tripStartRate += demand
        self._demands[mode].tripEndRate += demand
        self._demands[mode].rateOfPMT += demand * trip_distance
        self._demands[mode].averageDistanceInSystem += trip_distance

    def addModeStarts(self, mode: str, demand: float):
        self._demands[mode].tripStartRate += demand

    def addModeEnds(self, mode: str, demand: float):
        self._demands[mode].tripEndRate += demand

    def addModeThroughTrips(self, mode: str, demand: float, trip_distance: float):
        if demand > 0:
            try:
                current_demand = self._demands[mode].rateOfPMT
            except:
                current_demand = 0.0
            try:
                current_distance = self._demands[mode].averageDistanceInSystem
            except:
                current_distance = 0.0
            self._demands[mode].rateOfPMT += demand * trip_distance
            self._demands[mode].averageDistanceInSystem = (
                                                                      current_demand * current_distance + demand * trip_distance) / (
                                                                  current_demand + demand)

    def __str__(self):
        return 'Start Rate: ' + str(
            [mode + ' ' + str(self.getStartRate(mode)) for mode in self._modes]) + '; Dist: ' + str(
            [mode + ' ' + str(self.getAverageDistance(mode)) for mode in self._modes])
