import numpy as np


class Network:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, network_length,
                 avg_link_length):
        self.lam = smoothing
        self.u_f = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.L = network_length
        self.l = avg_link_length

    def getBaseSpeed(self):
        return self.u_f

    def MFD(self, N_eq, L_eq):
        maxDensity = 0.25
        if (N_eq / L_eq < maxDensity) & (N_eq / L_eq > 0.0):
            noCongestionN = (self.kappa * L_eq * self.w - L_eq * self.Q) / (self.u_f + self.w)
            N_eq = np.clip(N_eq, noCongestionN, None)
            v = - L_eq * self.lam / N_eq * np.log(
                np.exp(- self.u_f * N_eq / (L_eq * self.lam)) +
                np.exp(- self.Q / self.lam) +
                np.exp(-(self.kappa - N_eq / L_eq) * self.w / self.lam))
            return np.maximum(v, 0.0)
        else:
            return np.nan


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


class CollectedModeCharacteristics:
    def __init__(self):
        self._data = dict()

    def __setitem__(self, mode_name: str, mode_info: ModeCharacteristics):
        self._data[mode_name] = mode_info

    def __getitem__(self, mode_name):
        return self._data[mode_name]

    def getModes(self):
        return list(self._data.keys())

    def __str__(self):
        return str([str(self._data[key]) for key in self._data])


def getDefaultDemandCharacteristics(mode):
    """

    :param mode: str
    :return: io.DemandCharacteristics
    """
    if mode == 'car':
        return DemandCharacteristics(15., 0.0)
    elif mode == 'bus':
        return BusDemandCharacteristics(15., 0.0, 0.0, 0.0, 0.0)
    else:
        return DemandCharacteristics(15., 0.0)


def getDefaultSupplyCharacteristics():
    return SupplyCharacteristics(0.0, 0.0, 0.0)
