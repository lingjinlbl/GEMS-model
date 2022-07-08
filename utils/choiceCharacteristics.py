# from .microtype import MicrotypeCollection
from itertools import product

import numpy as np
import pandas as pd

from .misc import DistanceBins


class ChoiceCharacteristics:
    """
    UNITS ARE IN HOURS
    """

    def __init__(self, travel_time=0., cost=0., wait_time=0., access_time=0, unprotected_travel_time=0, distance=0,
                 data=None):
        self.__parameterToIdx = {'intercept': 0, 'travel_time': 1, 'cost': 2, 'wait_time': 3, 'access_time': 4,
                                 'unprotected_travel_time': 5, 'distance': 6}
        if data is None:
            self.__numpy = np.array([1.0, travel_time, cost, wait_time, access_time, unprotected_travel_time, distance],
                                    dtype=float)
        else:
            self.__numpy = data

    @property
    def travel_time(self):
        return self.__numpy[self.__parameterToIdx['travel_time']]

    @travel_time.setter
    def travel_time(self, val: float):
        self.__numpy[self.__parameterToIdx['travel_time']] = val

    @property
    def cost(self):
        return self.__numpy[self.__parameterToIdx['cost']]

    @cost.setter
    def cost(self, val: float):
        self.__numpy[self.__parameterToIdx['cost']] = val

    @property
    def wait_time(self):
        return self.__numpy[self.__parameterToIdx['wait_time']]

    @wait_time.setter
    def wait_time(self, val: float):
        self.__numpy[self.__parameterToIdx['wait_time']] = val

    # @property
    # def wait_time_squared(self):
    #     return self.__numpy[self.__parameterToIdx['wait_time_squared']]
    #
    # @wait_time_squared.setter
    # def wait_time_squared(self, val: float):
    #     self.__numpy[self.__parameterToIdx['wait_time_squared']] = val

    @property
    def access_time(self):
        return self.__numpy[self.__parameterToIdx['access_time']]

    @access_time.setter
    def access_time(self, val: float):
        self.__numpy[self.__parameterToIdx['access_time']] = val

    @property
    def unprotected_travel_time(self):
        return self.__numpy[self.__parameterToIdx['unprotected_travel_time']]

    @unprotected_travel_time.setter
    def unprotected_travel_time(self, val: float):
        self.__numpy[self.__parameterToIdx['unprotected_travel_time']] = val

    @property
    def distance(self):
        return self.__numpy[self.__parameterToIdx['distance']]

    @distance.setter
    def distance(self, val: float):
        self.__numpy[self.__parameterToIdx['distance']] = val

    @property
    def data(self):
        return self.__numpy

    def __len__(self):
        return len(self.__numpy)

    def idx(self):
        return self.__parameterToIdx

    def __add__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            self.wait_time_squared = self.wait_time ** 2.0
            self.access_time += other.access_time
            self.protected_distance += other.protected_distance
            self.distance += other.distance
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

    def __iadd__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            self.wait_time_squared = self.wait_time ** 2.0
            self.access_time += other.access_time
            self.protected_distance += other.protected_distance
            self.distance += other.distance
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self


class ModalChoiceCharacteristics:
    def __init__(self, modeToIdx=None, distanceInMiles=0.0, data=None):
        if modeToIdx is None:
            modeToIdx = dict()
        self.__modalChoiceCharacteristics = dict()
        if data is None:
            self.__numpy = np.zeros((len(modeToIdx), len(ChoiceCharacteristics())))
        else:
            self.__numpy = data
        self.distanceInMiles = distanceInMiles
        self.__modeToIdx = modeToIdx
        for mode, idx in modeToIdx.items():
            self.__modalChoiceCharacteristics[mode] = ChoiceCharacteristics(data=self.__numpy[idx, :])

    def __getitem__(self, item: str) -> ChoiceCharacteristics:
        return self.__modalChoiceCharacteristics[item]  # .setdefault(item, ChoiceCharacteristics())

    def __setitem__(self, key: str, value: ChoiceCharacteristics):
        self.__modalChoiceCharacteristics[key] = value

    def modes(self):
        return list(self.__modalChoiceCharacteristics.keys())

    def reset(self):
        for mode in self.modes():
            self[mode] = ChoiceCharacteristics()

    def __contains__(self, item):
        return item in self.__modalChoiceCharacteristics


class CollectedChoiceCharacteristics:
    def __init__(self, scenarioData, demand, numpyData, fixedData):
        self.__scenarioData = scenarioData
        self.__demand = demand
        self.modes = scenarioData.getPassengerModes()
        self.__choiceCharacteristics = dict()
        self.__distanceBins = DistanceBins()
        self.__numpy = numpyData['choiceCharacteristics']
        self.__fleetSize = numpyData['fleetSize']
        self.__fixedData = fixedData
        self.__broken = False
        self.__cache = dict()

    def clearCache(self, category=None):
        if category is None:
            self.__cache.clear()
        elif category in self.__cache:
            self.__cache.pop(category)

    @property
    def costTypeToIdx(self):
        return self.__scenarioData.costTypeToIdx

    @property
    def odiToIdx(self):
        return self.__scenarioData.odiToIdx

    @property
    def passengerModeToIdx(self):
        return self.__scenarioData.passengerModeToIdx

    @property
    def dataToIdx(self):
        return self.__scenarioData.demandDataTypeToIdx

    @property
    def paramToIdx(self):
        return self.__scenarioData.paramToIdx

    @property
    def diToIdx(self):
        return self.__scenarioData.diToIdx

    @property
    def numpy(self) -> np.ndarray:
        return self.__numpy

    @property
    def broken(self) -> bool:
        return self.__broken

    def __setitem__(self, key, value: ModalChoiceCharacteristics):
        self.__choiceCharacteristics[key] = value

    def __getitem__(self, item) -> ModalChoiceCharacteristics:
        di, odi, mode = item
        return self.__numpy[self.diToIdx[di], self.odiToIdx[odi], self.passengerModeToIdx[mode], :]
        # return self.__choiceCharacteristics[item]

    def toDataFrame(self):  # self.homeMicrotype, self.populationGroupType, self.tripPurpose
        odis = [odi.toTuple() for odi in self.odiToIdx.keys()]
        dis = [di.toTuple() for di in self.diToIdx.keys()]
        modes = self.passengerModeToIdx.keys()
        params = self.paramToIdx.keys()
        tuples = [(f, g, h, a, b, c, d, e) for (f, g, h), (a, b, c), d, e in product(dis, odis, modes, params)]
        mi = pd.MultiIndex.from_tuples(tuples, names=('homeMicrotype', 'populationGroup', 'tripPurpose',
                                                      'originMicrotype', 'destinationMicrotype', 'distanceBin', 'mode',
                                                      'parameter'))
        return pd.DataFrame({"Value": self.__numpy.flatten()}, index=mi).unstack()

    def initializeChoiceCharacteristics(self, microtypes, distanceBins: DistanceBins):
        self.__distanceBins = distanceBins
        self.__numpy[:, :, :, self.paramToIdx['intercept']] = 1
        for odIndex, odIndexIdx in self.odiToIdx.items():
            self.__numpy[:, odIndexIdx, :, self.paramToIdx['distance']] = distanceBins[odIndex.distBin]

    def resetChoiceCharacteristics(self):
        # {'intercept': 0, 'travel_time': 1, 'cost': 2, 'wait_time': 3, 'access_time': 4,
        #  'unprotected_travel_time': 5, 'distance': 6}
        for param in ['travel_time', 'cost', 'wait_time', 'access_time', 'unprotected_travel_time']: # TODO: vectorize
            self.__numpy[:, :, :, self.paramToIdx[param]] = 0.0
        # self.__numpy[~np.isnan(self.__numpy)] *= 0.0
        self.__numpy[:, :, :, self.paramToIdx['intercept']] = 1

    def updateChoiceCharacteristics(self, microtypes) -> np.ndarray:
        endIdx = self.__scenarioData.firstFreightIdx
        self.resetChoiceCharacteristics()
        travelTimeInHours, broken = speedToTravelTime(microtypes.numpySpeed[:, :endIdx],
                                                      self.__demand.toThroughDistance)
        mixedTravelPortion = mixedPortion(microtypes.numpyMixedTrafficDistance[:, :endIdx],
                                          self.__demand.toThroughDistance)  # Right now it's a waste to recalculate it every time but it might come in handy at some point?
        self.__broken = broken

        allCosts = self.__fixedData['microtypeCosts'][:, :, :endIdx, :]  # only grab this for passenger modes

        accessDistance = self.__fixedData['accessDistance'][:, :endIdx]
        # microtypeAccessSeconds = (1 / microtypes.numpySpeed[:, self.passengerModeToIdx['walk'], None]) * accessDistance
        microtypeAccessSeconds = accessDistance / 1.5

        accessSeconds = self.__cache.setdefault(
            'accessDistance', np.einsum('im,ki->km', microtypeAccessSeconds,
                                        self.__fixedData['toStarts'] + self.__fixedData['toEnds']))

        bikeFleetDensity = self.__cache.setdefault(
            'bikeFleetDensity', np.einsum('im,ki->km', self.__fleetSize[:, :endIdx], self.__fixedData['toStarts']))
        startCosts = self.__cache.setdefault(
            'startCosts', np.einsum('ijmc,ki->jkm', allCosts[:, :, :, [self.costTypeToIdx['perStartPrivateCost'],
                                                                       self.costTypeToIdx['perStartPublicCost']]],
                                    self.__fixedData['toStarts']))
        endCosts = self.__cache.setdefault(
            'endCosts', np.einsum('ijmc,ki->jkm', allCosts[:, :, :, [self.costTypeToIdx['perEndPrivateCost'],
                                                                     self.costTypeToIdx['perEndPublicCost']]],
                                  self.__fixedData['toEnds']))
        throughCosts = self.__cache.setdefault(
            'throughCosts', np.einsum('ijmc,ki->jkm', allCosts[:, :, :, [self.costTypeToIdx['perMilePrivateCost'],
                                                                         self.costTypeToIdx['perMilePublicCost']]],
                                      self.__fixedData['toThroughDistance']))

        self.__numpy[:, :, :, self.paramToIdx['cost']] = startCosts + endCosts + throughCosts
        self.__numpy[:, :, :, self.paramToIdx['travel_time']] = travelTimeInHours[None, :]
        self.__numpy[:, :, :, self.paramToIdx['unprotected_travel_time']] = (travelTimeInHours * mixedTravelPortion)[
                                                                            None, :]
        self.__numpy[:, :, :, self.paramToIdx['mode_density']] = bikeFleetDensity[None, :, :]
        self.__numpy[:, :, :, self.paramToIdx['access_time']] = accessSeconds[None, :] / 3600.
        return self.__numpy

    def isBroken(self):
        return self.__broken


def speedToTravelTime(modeSpeed: np.ndarray, toThroughDistance: np.ndarray) -> (np.ndarray, bool):
    if np.any(np.isnan(modeSpeed) | (modeSpeed <= 0.001)):
        broken = True
    else:
        broken = False
    modeSpeed[(modeSpeed < 0.001) | np.isnan(modeSpeed)] = 0.001
    modeSecondsPerMeter = (1 / modeSpeed)
    modeSecondsPerMeter[np.isinf(modeSecondsPerMeter)] = 0.0
    assignmentMatrix = toThroughDistance * 1609.34
    throughTravelTimeInSeconds = assignmentMatrix @ modeSecondsPerMeter
    return throughTravelTimeInSeconds / 3600.0, broken


def mixedPortion(mixedPortionByMicrotype: np.ndarray, toThroughDistance: np.ndarray):
    assignmentMatrix = toThroughDistance * 1609.34
    mixedTrafficDistance = assignmentMatrix @ mixedPortionByMicrotype
    full = np.ones_like(mixedPortionByMicrotype)
    out = mixedTrafficDistance / (assignmentMatrix @ full)
    out[np.isnan(out)] = 1.0
    return out


def filterAllocation(mode: str, inputAllocation, microtypes):
    through_microtypes = []
    allocation = []
    tot = 0.0
    for m, a in inputAllocation:
        if (a > 0) & (mode in microtypes[m].mode_names):
            through_microtypes.append(m)
            allocation.append(a)
            tot += a
    return {m: a / tot for m, a in zip(through_microtypes, allocation)}  # dict(zip(through_microtypes, allocation))
