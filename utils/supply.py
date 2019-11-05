# -*- coding: utf-8 -*-

import numpy as np
import utils.IO as io



def getBusdwellTime(v, params_bus, modeDemand):
    if v > 0:
        out = 1. / (params_bus.s_b * v) * (
                v * params_bus.k * params_bus.t_0 * params_bus.s_b +
                params_bus.gamma_s * 2 * modeDemand) / (
                      params_bus.k - params_bus.gamma_s * 2 * modeDemand)
    else:
        out = np.nan
    return out


def getModeDemandCharacteristics(baseSpeed, mode, modeCharacteristics: io.ModeCharacteristics):
    """

    :param modeCharacteristics:
    :param baseSpeed: float
    :type mode: str
    :return: io.DemandCharacteristics
    """
    modeParams = modeCharacteristics.params
    modeDemand = modeCharacteristics.demand
    if mode == 'car':
        return io.DemandCharacteristics(baseSpeed, modeDemand * modeParams.mean_trip_distance)
    elif mode == 'bus':
        assert (isinstance(modeParams, io.BusParams))
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

        return io.BusDemandCharacteristics(speed, passengerFlow, dwellTime, headway, occupancy)

    else:
        return io.DemandCharacteristics(baseSpeed, modeDemand * modeParams.mean_trip_distance)


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
