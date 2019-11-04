# -*- coding: utf-8 -*-

import numpy as np


def MFD(N_eq, L_eq, params):
    maxDensity = 0.25
    if (N_eq/L_eq < maxDensity) & (N_eq/L_eq > 0.0):
        l = params['lambda']
        u = params['u_f']
        Q = params['Q']
        w = params['w']
        k = params['kappa']
        noCongestionN = (k * L_eq * w - L_eq * Q) / (u + w)
        N_eq = np.clip(N_eq, noCongestionN, None)
        v = - L_eq * l / N_eq * np.log(
            np.exp(- u * N_eq / (L_eq * l)) +
            np.exp(- Q / l) +
            np.exp(-(k - N_eq / L_eq) * w / l))
        return np.maximum(v, 0.0)
    else:
        return np.nan


def getBusdwellTime(v, params_bus, modeDemand):
    if v > 0:
        out = 1. / (params_bus['s_b'] * v) * (
                v * params_bus['k'] * params_bus['t_0'] * params_bus['s_b'] +
                params_bus['gamma_s'] * 2 * modeDemand) / (
                      params_bus['k'] - params_bus['gamma_s'] * 2 * modeDemand)
    else:
        out = np.nan
    return out


def getModeDemandCharacteristics(baseSpeed, mode, modeParams, modeDemand):
    """

    :param baseSpeed: float
    :type modeDemand: float
    :type mode: str
    :type modeParams: dict
    :return: float
    """
    if mode == 'car':
        return {'speed': baseSpeed,
                'passengerFlow': modeDemand * modeParams['meanTripDistance']}
    elif mode == 'bus':
        dwellTime = getBusdwellTime(baseSpeed, modeParams, modeDemand)
        if dwellTime > 0:
            speed = baseSpeed / (1 + dwellTime * baseSpeed * modeParams['s_b'])
            headway = modeParams['L_mode'] / speed
        else:
            speed = 0.0
            headway = np.nan

        if (dwellTime > 0) & (baseSpeed > 0):
            passengerFlow: float = modeDemand * modeParams['meanTripDistance']
            occupancy: float = passengerFlow / modeParams['k'] / speed
        else:
            passengerFlow: float = 0.0
            occupancy: float = np.nan

        return {'speed': speed,
                'dwellTime': dwellTime,
                'headway': headway,
                'occupancy': occupancy,
                'passengerFlow': passengerFlow}
    else:
        return baseSpeed


def getModeBlockedDistance(microtype, mode):
    """

    :param microtype: Microtype
    :param mode: str
    :return: float
    """
    if mode == 'car':
        return 0.0
    elif mode == 'bus':
        modeParams = microtype.mode_params[mode]
        modeSpeed = microtype._modeDemandCharacteristics[mode].get('speed')
        modeDemand = microtype._demands[mode]
        dwellTime = getBusdwellTime(microtype._baseSpeed, modeParams, modeDemand)
        return microtype.network_params['l'] * modeParams['L_mode'] * modeParams['s_b'] * modeParams[
            'k'] * dwellTime * modeSpeed
    else:
        return 0.0
