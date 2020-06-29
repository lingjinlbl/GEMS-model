import numpy as np
from utils.microtype import Microtype
from typing import Dict, List
import pandas as pd


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


class ModeSplit:
    def __init__(self, mapping=None):
        if mapping is None:
            self._mapping = Dict[str, float]
        else:
            assert (isinstance(mapping, Dict))
            self._mapping = mapping

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
        return str([mode + ': ' + str(self[mode]) + '| ' for mode in self.keys()])


class ChoiceCharacteristics:
    def __init__(self, travel_time=0., cost=0., wait_time=0.):
        self.travel_time = travel_time
        self.cost = cost
        self.wait_time = wait_time

    def __add__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

    def __iadd__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self


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

    def __setitem__(self, key: Microtype, value: Dict[str, float]):
        self.allocation[key] = value

    def __getitem__(self, item: Microtype):
        return self.allocation[item]

    def getChoiceCharacteristics(self) -> ModeCharacteristics:
        mode_characteristics = ModeCharacteristics(list(self.mode_split.keys()))
        for mode in self.mode_split.keys():
            choice_characteristics = ChoiceCharacteristics()
            for microtype in self.allocation.keys():
                time, cost, wait = microtype.getThroughTimeCostWait(mode, self.distance * self.allocation[microtype])
                choice_characteristics += ChoiceCharacteristics(time, cost, wait)
                time, cost, wait = microtype.getStartTimeCostWait(mode)
                choice_characteristics += ChoiceCharacteristics(time, cost, wait)
                time, cost, wait = microtype.getEndTimeCostWait(mode)
                choice_characteristics += ChoiceCharacteristics(time, cost, wait)
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


class ODindex:
    def __init__(self, o, d, distBin: int):
        if isinstance(o, Microtype):
            self.o = o.microtypeID
        elif isinstance(o, str):
            self.o = o
        else:
            print("AAAH")
        if isinstance(d, Microtype):
            self.d = d.microtypeID
        elif isinstance(d, str):
            self.d = d
        else:
            print("AAAAH")
        self.distBin = distBin

    def __eq__(self, other):
        if isinstance(other, ODindex):
            if (self.o == other.o) & (self.distBin == other.distBin) & (self.d == other.d):
                return True
            else:
                return False
        else:
            return False

    def __hash__(self):
        return hash((self.o, self.d, self.distBin))

    def __str__(self):
        return str(self.distBin) + " trip from " + self.o + " to " + self.d


class Trip:
    def __init__(self, odIndex: ODindex, allocation: Allocation):
        self.odIndex = odIndex
        self.allocation = allocation


class TripCollection:
    def __init__(self):
        self.__trips = dict()

    def __setitem__(self, key: ODindex, value: Trip):
        self.__trips[key] = value

    def __getitem__(self, item: ODindex) -> Trip:
        assert isinstance(item, ODindex)
        if item in self.__trips:
            return self.__trips[item]
        else:
            print("Not in database!")
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


class TripGeneration:
    def __init__(self):
        self.__data = pd.DataFrame()
        self.__tripClasses = dict()

    def __setitem__(self, key: (str, str), value: float):
        self.__tripClasses[key] = value

    def __getitem__(self, item: (str, str)):
        return self.__tripClasses[item]

    def importTripGeneration(self, df: pd.DataFrame):
        self.__data = df

    def initializeTimePeriod(self, timePeriod: str):
        self.__tripClasses = dict()
        relevantDemand = self.__data.loc[self.__data["TimePeriodID"] == timePeriod]
        for row in relevantDemand.itertuples():
            self[row.PopulationGroupTypeID, row.TripPurposeID] = row.TripGenerationRatePerHour


class OriginDestination:
    def __init__(self):
        self.__ods = pd.DataFrame()
        self.__distances = pd.DataFrame()
        self.__originDestination = dict()

    def importOriginDestination(self, ods: pd.DataFrame, distances: pd.DataFrame):
        self.__ods = ods
        self.__distances = distances

    def __setitem__(self, key: DemandIndex, value: dict):
        self.__originDestination[key] = value

    def __getitem__(self, item: DemandIndex):
        return self.__originDestination[item]

    def initializeTimePeriod(self, timePeriod: str):
        self.__originDestination = dict()
        relevantODs = self.__ods.loc[self.__ods["TimePeriodID"] == timePeriod]
        merged = relevantODs.merge(self.__distances,
                                   on=["TripPurposeID", "OriginMicrotypeID", "DestinationMicrotypeID"],
                                   suffixes=("_OD", "_Dist"),
                                   how="inner")
        for tripClass, grouped in merged.groupby(["HomeMicrotypeID", "PopulationGroupTypeID", "TripPurposeID"]):
            grouped["tot"] = grouped["Portion_OD"] * grouped["Portion_Dist"]
            tot = np.sum(grouped["tot"])
            assert tot == 1.0
            distribution = dict()
            for row in grouped.itertuples():
                distribution[ODindex(row.OriginMicrotypeID, row.DestinationMicrotypeID, row.DistanceBinID)] = row.tot
            self[tripClass] = distribution
        # for row in relevantDemand.itertuples():
        #     self[row.PopulationGroupTypeID, row.TripPurposeID] = row.TripGenerationRatePerHour


class Demand:
    def __init__(self):
        self.__demand = dict()
        self.__modeSplit = dict()

    def __setitem__(self, key: (DemandIndex, ODindex), value: float):
        self.__demand[key] = value

    def __getitem__(self, item):
        return self.__demand[item]

#    def initializeDemand(self, originDestination: OriginDestination):
