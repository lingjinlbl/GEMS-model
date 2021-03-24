import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

# from utils.microtype import Microtype
from .choiceCharacteristics import ChoiceCharacteristics

warnings.filterwarnings("ignore")


class Allocation:
    def __init__(self, mapping=None):
        if mapping is None:
            self._mapping = Dict[str, float]
        else:
            assert (isinstance(mapping, Dict))
            self._mapping = mapping

    def __setitem__(self, key, value):
        self._mapping[key] = value

    def __getitem__(self, item):
        return self._mapping[item]

    def keys(self):
        return self._mapping.keys()

    def __iter__(self):
        return iter(self._mapping.items())

    def pop(self, item):
        self._mapping.pop(item)


class ModeSplit:
    """
    Class for storing mode splits and respective properties
    """

    def __init__(self, mapping=None, demandForTrips=0, demandForPMT=0):
        self.demandForTripsPerHour = demandForTrips
        self.demandForPmtPerHour = demandForPMT
        if mapping is None:
            self._mapping = dict()
        else:
            assert (isinstance(mapping, Dict))
            self._mapping = mapping
        self.__counter = 1.0

    def updateMapping(self, mapping: Dict[str, float]):
        if self._mapping.keys() == mapping.keys():
            self._mapping = mapping
        else:
            print("OH NO BAD MAPPING")

    def copy(self):
        return ModeSplit(self._mapping.copy(), self.demandForTripsPerHour, self.demandForPmtPerHour)

    def __sub__(self, other):
        output = []
        for key in self._mapping.keys():
            output.append(self[key] - other[key])
        return np.linalg.norm(output)

    def __rsub__(self, other):
        output = []
        for key in self._mapping.keys():
            output.append(self[key] - other[key])
        return np.linalg.norm(output)

    def __mul__(self, other):
        out = self.copy()
        portion = 1. / self.__counter  # uniform(0.5 / self.__counter, 1. / self.__counter)
        for key in out._mapping.keys():
            out[key] = out[key] * portion + other[key] * (1.0 - portion)
        self.__counter += 0.2
        return out

    def __imul__(self, other):
        portion = 1. / self.__counter  # uniform(0.5 / self.__counter, 1. / self.__counter)
        for key in self._mapping.keys():
            self[key] = self[key] * portion + other[key] * (1.0 - portion)
        self.__counter += 0.2
        return self

    def __add__(self, other):
        out = self.copy()
        for key in self._mapping.keys():
            out[key] = (self[key] * self.demandForTripsPerHour + other[key] * other.demandForTripsPerHour) / (
                    self.demandForTripsPerHour + other.demandForTripsPerHour)
        out.demandForTripsPerHour = (self.demandForTripsPerHour + other.demandForTripsPerHour)
        out.demandForPmtPerHour = (self.demandForPmtPerHour + other.demandForPmtPerHour)
        return out

    def __iadd__(self, other):
        if self._mapping:
            for key in self._mapping.keys():
                self[key] = (self[key] * self.demandForTripsPerHour + other[key] * other.demandForTripsPerHour) / (
                        self.demandForTripsPerHour + other.demandForTripsPerHour)
        else:
            self._mapping = other._mapping.copy()
        self.demandForTripsPerHour = (self.demandForTripsPerHour + other.demandForTripsPerHour)
        self.demandForPmtPerHour = (self.demandForPmtPerHour + other.demandForPmtPerHour)
        return self

    def toDict(self):
        out = self._mapping.copy()
        out["PMT"] = self.__demandForPmtPerHour
        out["Trips"] = self.__demandForTripsPerHour
        return out

    def keys(self):
        return list(self._mapping.keys())

    @property
    def demandForPmtPerHour(self):
        return self.__demandForPmtPerHour

    @demandForPmtPerHour.setter
    def demandForPmtPerHour(self, demandForPMT):
        if demandForPMT < 0:
            self.__demandForPmtPerHour = 0
            print("OH NO NEGATIVE PMT ")
        elif demandForPMT >= 0:
            self.__demandForPmtPerHour = demandForPMT
        else:
            self.__demandForPmtPerHour = 0
            print("OH NO BAD PMT ")

    @property
    def demandForTripsPerHour(self):
        return self.__demandForTripsPerHour

    @demandForTripsPerHour.setter
    def demandForTripsPerHour(self, demandForTrips):
        if demandForTrips < 0:
            self.__demandForTripsPerHour = 0
            print("OH NO NEGATIVE DEMAND FOR TRIPS")
        elif demandForTrips >= 0:
            self.__demandForTripsPerHour = demandForTrips
        else:
            self.__demandForTripsPerHour = 0
            print("OH NO BAD TRIP")

    def __setitem__(self, key, value):
        self._mapping[key] = value

    def __getitem__(self, item):
        if item in self._mapping.keys():
            return self._mapping[item]
        else:
            return 0.0

    def keys(self) -> List:
        return list(self._mapping.keys())

    def __str__(self):
        return str([mode + ': ' + str(self[mode]) for mode in self.keys()])

    def __iter__(self):
        return iter(self._mapping.items())


class ModeCharacteristics:
    def __init__(self, modes: List[str]):
        self._modes = modes
        self.characteristics = dict()
        for mode in modes:
            self.characteristics[mode] = ChoiceCharacteristics()

    def __getitem__(self, item):
        if item in self._modes:
            return self.characteristics[item]

    def __setitem__(self, key, value):
        self.characteristics[key] = value

    def keys(self) -> List:
        return list(self._modes)


class DemandUnit:
    def __init__(self, distance: float, demand: float, allocation=None, mode_split=None):
        if allocation is None:
            allocation = Allocation()
        if mode_split is None:
            mode_split = ModeSplit({'car': 1.0})
        self.distance = distance
        self.demand = demand
        self.allocation = allocation
        self.mode_split = mode_split

    def __setitem__(self, key, value: Dict[str, float]):
        self.allocation[key] = value

    def __getitem__(self, item):
        return self.allocation[item]

    def getChoiceCharacteristics(self) -> ModeCharacteristics:
        mode_characteristics = ModeCharacteristics(list(self.mode_split.keys()))
        for mode in self.mode_split.keys():
            choice_characteristics = ChoiceCharacteristics()
            for microtype in self.allocation.keys():
                choice_characteristics += microtype.getThroughTimeCostWait(mode,
                                                                           self.distance * self.allocation[microtype])
                choice_characteristics += microtype.getStartTimeCostWait(mode)
                choice_characteristics += microtype.getEndTimeCostWait(mode)
            mode_characteristics[mode] = choice_characteristics
        return mode_characteristics

    def updateModeSplit(self, mode_split: ModeSplit):
        for key in self.mode_split.keys():
            self.mode_split[key] = (mode_split[key] + self.mode_split[key]) / 2.0


class DemandIndex:
    def __init__(self, homeMicrotypeID, populationGroupTypeID, tripPurposeID):
        self.homeMicrotype = homeMicrotypeID
        self.populationGroupType = populationGroupTypeID
        self.tripPurpose = tripPurposeID

    def __eq__(self, other):
        if (self.homeMicrotype == other.homeMicrotype) & (self.populationGroupType == other.populationGroupType) & (
                self.tripPurpose == other.tripPurpose):
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.homeMicrotype, self.populationGroupType, self.tripPurpose))

    def __str__(self):
        return "Home: " + self.homeMicrotype + ", type: " + self.populationGroupType + ", purpose: " + self.tripPurpose

    def toIndex(self):
        return pd.MultiIndex.from_tuples([(self.homeMicrotype, self.populationGroupType, self.tripPurpose)],
                                         names=('homeMicrotype', 'populationGroupType', 'tripPurpose'))


class ODindex:
    def __init__(self, o, d, distBin: int):
        if isinstance(o, str):
            self.o = o
        else:
            self.o = o.microtypeID
        if isinstance(d, str):
            self.d = d
        else:
            self.d = d.microtypeID
        self.distBin = distBin
        self.__hash = hash((self.o, self.d, self.distBin))

    def __eq__(self, other):
        if isinstance(other, ODindex):
            if (self.o == other.o) & (self.distBin == other.distBin) & (self.d == other.d):
                return True
            else:
                return False
        else:
            return False

    def __hash__(self):
        return self.__hash

    def __str__(self):
        return str(self.distBin) + " trip from " + self.o + " to " + self.d


class Trip:
    def __init__(self, odIndex: ODindex, allocation: Allocation):
        self.odIndex = odIndex
        self.allocation = allocation


class TripCollection:
    """
    Class to store trips, their microtypes, and the distance it belongs to as well as other aspects.
    """

    def __init__(self):
        self.__trips = dict()

    def __setitem__(self, key: ODindex, value: Trip):
        self.__trips[key] = value

    def __getitem__(self, item: ODindex) -> Trip:
        # assert isinstance(item, ODindex)
        if item in self.__trips:
            return self.__trips[item]
        else:
            # print("Not in database! for " + str(item))
            if item.o == item.d:
                allocation = Allocation({item.o: 1.0})
            else:
                allocation = Allocation({item.o: 0.5, item.d: 0.5})
            self[item] = Trip(item, allocation)
            return self[item]

    def importTrips(self, df: pd.DataFrame):
        for row in df.itertuples():
            odi = ODindex(row.FromMicrotypeID, row.ToMicrotypeID, row.DistanceBinID)
            if odi in self.__trips:
                self[odi].allocation[row.ThroughMicrotypeID] = row.Portion
            else:
                self[odi] = Trip(odi, Allocation({row.ThroughMicrotypeID: row.Portion}))
        print("-------------------------------")
        print("|  Loaded ", len(df), " trips")

    def __iter__(self):
        return iter(self.__trips.items())


class TripGeneration:
    """
    Class to import and initialize trips from data.
    """

    def __init__(self):
        self.__data = pd.DataFrame()
        self.__tripClasses = dict()
        self.__currentTimePeriod = "BAD"

    @property
    def tripClasses(self):
        if self.__currentTimePeriod not in self.__tripClasses:
            self.__tripClasses[self.__currentTimePeriod] = dict()
        return self.__tripClasses[self.__currentTimePeriod]

    def setTimePeriod(self, timePeriod: str):
        self.__currentTimePeriod = timePeriod

    def __setitem__(self, key: (str, str), value: float):
        self.tripClasses[key] = value

    def __getitem__(self, item: (str, str)):
        return self.tripClasses[item]

    def __contains__(self, item):
        return item in self.__tripClasses

    def importTripGeneration(self, df: pd.DataFrame):
        self.__data = df
        print("|  Loaded ", len(df), " trip generation rates")
        print("-------------------------------")

    def initializeTimePeriod(self, timePeriod: str):
        # self.__tripClasses = dict()
        self.__currentTimePeriod = timePeriod
        if timePeriod not in self:
            relevantDemand = self.__data.loc[self.__data["TimePeriodID"] == timePeriod]
            for row in relevantDemand.itertuples():
                self[row.PopulationGroupTypeID, row.TripPurposeID] = row.TripGenerationRatePerHour
            print("|  Loaded ", len(relevantDemand), " demand classes")

    def __iter__(self):
        return iter(self.tripClasses.items())


class OriginDestination:
    """
    A class to import and store the origin and destination of trips.
    """

    def __init__(self):
        self.__ods = pd.DataFrame()
        self.__distances = pd.DataFrame()
        self.__originDestination = dict()
        self.__currentTimePeriod = "BAD"

    def setTimePeriod(self, timePeriod: str):
        self.__currentTimePeriod = timePeriod

    @property
    def originDestination(self):
        if self.__currentTimePeriod not in self.__originDestination:
            self.__originDestination[self.__currentTimePeriod] = dict()
        return self.__originDestination[self.__currentTimePeriod]

    def importOriginDestination(self, ods: pd.DataFrame, distances: pd.DataFrame):
        self.__ods = ods
        self.__distances = distances
        print("|  Loaded ", len(ods), " ODs and ", len(distances), "unique distance bins")

    def __setitem__(self, key: DemandIndex, value: dict):
        self.originDestination[key] = value

    def __getitem__(self, item: DemandIndex):
        if item not in self.originDestination:
            print("OH NO, no origin destination defined for ", str(item), " in ", self.__currentTimePeriod)
            subitem = self.__distances.loc[(self.__distances["OriginMicrotypeID"] == item.homeMicrotype) & (
                    self.__distances["DestinationMicrotypeID"] == item.homeMicrotype) & (
                                                   self.__distances["TripPurposeID"] == item.tripPurpose)]
            out = dict()
            for row in subitem.itertuples():
                out[ODindex(row.OriginMicrotypeID, row.DestinationMicrotypeID, row.DistanceBinID)] = row.Portion
            self.originDestination[item] = out
            return out
        else:
            return self.originDestination[item]

    def __contains__(self, item):
        return item in self.originDestination

    def initializeTimePeriod(self, timePeriod: str):
        self.__currentTimePeriod = timePeriod
        if timePeriod not in self.__originDestination:
            print("|  Loaded ", len(self.__ods.loc[self.__ods["TimePeriodID"] == timePeriod]), " distance bins")
            relevantODs = self.__ods.loc[self.__ods["TimePeriodID"] == timePeriod]
            merged = relevantODs.merge(self.__distances,
                                       on=["TripPurposeID", "OriginMicrotypeID", "DestinationMicrotypeID"],
                                       suffixes=("_OD", "_Dist"),
                                       how="inner")
            for tripClass, grouped in merged.groupby(["HomeMicrotypeID", "PopulationGroupTypeID", "TripPurposeID"]):
                grouped["tot"] = grouped["Portion_OD"] * grouped["Portion_Dist"]
                tot = np.sum(grouped["tot"])
                grouped["tot"] = grouped["tot"] / tot
                if abs(tot - 1) > 0.0001:  # TODO: FIX
                    print(f"Oops, totals for {tripClass} add up to {tot}")
                distribution = dict()
                for row in grouped.itertuples():
                    distribution[
                        ODindex(row.OriginMicrotypeID, row.DestinationMicrotypeID, row.DistanceBinID)] = row.tot
                self[DemandIndex(*tripClass)] = distribution
        # for row in relevantDemand.itertuples():
        #     self[row.PopulationGroupTypeID, row.TripPurposeID] = row.TripGenerationRatePerHour


class TransitionMatrix:
    def __init__(self, microtypes: list, matrix=None):
        self.__names = microtypes
        self.__nameToIdx = {val: idx for idx, val in enumerate(microtypes)}
        self.__averageSpeeds = np.zeros((len(microtypes), 1))
        if isinstance(matrix, pd.DataFrame):
            self.__matrix = pd.DataFrame(0.0, index=microtypes, columns=microtypes).add(matrix, fill_value=0.0)
        elif matrix is None:
            self.__matrix = pd.DataFrame(0.0, index=microtypes, columns=microtypes)
        else:
            print("ERROR INITIALIZING TRANSITION MATRIX")

    @property
    def averageSpeeds(self) -> np.ndarray:
        return self.__averageSpeeds

    def setAverageSpeeds(self, averageSpeeds: np.ndarray):
        self.__averageSpeeds = averageSpeeds

    @property
    def names(self) -> list:
        return self.__names

    @property
    def matrix(self) -> pd.DataFrame:
        return self.__matrix

    def __getitem__(self, item):
        return dict(zip(self.__names, self.__matrix[self.__nameToIdx[item], :].values))

    def __add__(self, other):
        if isinstance(other, TransitionMatrix):
            return TransitionMatrix(self.__names, self.matrix + other.matrix)
        else:
            print("ERROR ADDING TRANSITION MATRIX")
            return self

    def __radd__(self, other):
        if isinstance(other, TransitionMatrix):
            return TransitionMatrix(self.__names, self.matrix + other.matrix)
        else:
            print("ERROR ADDING TRANSITION MATRIX")
            return self

    def __mul__(self, other):
        try:
            return TransitionMatrix(self.__names, self.matrix * other)
        except Exception as err:
            print("ERROR multiplying TRANSITION MATRIX")
            print(err)
            return self

    def idx(self, idx):
        return self.__nameToIdx[idx]

    def fillZeros(self):
        self.__matrix += 1. / (len(self.__names)**2)
        return self


class TransitionMatrices:
    def __init__(self, microtypes=None):
        if microtypes is None:
            microtypes = []
        self.__names = microtypes
        self.__data = pd.DataFrame()

    def __getitem__(self, item: ODindex):
        try:
            return TransitionMatrix(self.__names, self.__data.loc[item.o, item.d, item.distBin])
        except Exception as err:
            print(f"No transition matrix found for {err}")
            out = TransitionMatrix(self.__names).fillZeros()
            return out

    def setNames(self, names: list):
        self.__names = names

    def importTransitionMatrices(self, df: pd.DataFrame):
        self.__data = df
        print("|  Loaded ", len(df), " transition probabilities")
        print("-------------------------------")
