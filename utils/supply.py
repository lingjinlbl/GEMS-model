# -*- coding: utf-8 -*-


class ModeParams:
    def __init__(self, road_network_fraction=1.0, relative_length=1.0):
        self.road_network_fraction = road_network_fraction
        self.size = relative_length

    def getSize(self):
        return self.size

    def getFixedDensity(self):
        return None


class BusParams(ModeParams):
    def __init__(self, road_network_fraction: float, relative_length: float,
                 fixed_density: float, min_stop_time: float,
                 stop_spacing: float, passenger_wait: float) -> None:
        super().__init__(road_network_fraction, relative_length)
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

    def __str__(self):
        return 'N_eq: ' + str(self.N_eq) + ' , L_eq: ' + str(self.L_eq)


class DemandCharacteristics:
    def __init__(self, speed, passenger_flow):
        self.speed = speed
        self.passenger_flow = passenger_flow

    def getSpeed(self):
        return self.speed

    def __str__(self):
        return 'Speed: ' + str(self.speed) + ' , Flow: ' + str(self.passenger_flow)


class BusDemandCharacteristics(DemandCharacteristics):
    def __init__(self, speed, passenger_flow, dwell_time, headway, occupancy):
        super().__init__(speed, passenger_flow)
        self.dwell_time = dwell_time
        self.headway = headway
        self.occupancy = occupancy


class TravelDemand:
    def __init__(self, modes):
        self._modes = modes
        self._tripStartRate = dict()
        self._tripEndRate = dict()
        self._rateOfPMT = dict()
        self._averageDistanceInSystem = dict()
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

    def getAverageDistance(self, mode: str):
        return self._averageDistanceInSystem[mode]

    def resetDemand(self):
        for mode in self._modes:
            self._tripStartRate[mode] = 0.
            self._tripEndRate[mode] = 0.
            self._rateOfPMT[mode] = 0.

    def setSingleDemand(self, mode, demand: float, trip_distance: float):
        self._tripStartRate[mode] = demand
        self._tripEndRate[mode] = demand
        self._rateOfPMT[mode] = demand * trip_distance
        self._averageDistanceInSystem[mode] = trip_distance

    def addModeStarts(self, mode: str, demand: float):
        self._tripStartRate[mode] += demand

    def addModeEnds(self, mode: str, demand: float):
        self._tripEndRate[mode] += demand

    def addModePMT(self, mode: str, demand: float, trip_distance: float):
        try:
            current_demand = self._rateOfPMT[mode]
        except:
            current_demand = 0.0
        try:
            current_distance = self._averageDistanceInSystem[mode]
        except:
            current_distance = 0.0
        self._rateOfPMT[mode] += demand * trip_distance
        self._averageDistanceInSystem[mode] = (current_demand * current_distance + demand * trip_distance) / (
                current_demand + demand)

    def __str__(self):
        return 'Start Rate: ' + str(
            [mode + ' ' + str(self.getStartRate(mode)) for mode in self._modes]) + '; Dist: ' + str(
            [mode + ' ' + str(self.getAverageDistance(mode)) for mode in self._modes])
