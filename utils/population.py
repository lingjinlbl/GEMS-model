import pandas as pd
import numpy as np

from utils.OD import DemandIndex, TripCollection, Trip, ModeSplit, OriginDestination
from utils.choiceCharacteristics import ModalChoiceCharacteristics, CollectedChoiceCharacteristics
from utils.microtype import MicrotypeCollection


class PopulationGroup:
    def __init__(self, homeLocation: str, populationGroupType: str, population: float):
        self.homeLocation = homeLocation
        self.populationGroupType = populationGroupType
        self.population = population


class DemandClass:
    def __init__(self):
        self.__params = {"ASC": 0.0, "VOT": 15.0, "VOM": 1.0}

    def __iadd__(self, other: dict):
        self.__params.update(other)
        print("UPDATED")

    def __add__(self, other: dict):
        self.__params.update(other)
        print("UPDATED")
        return self

    def __getitem__(self, item):
        return self.__params[item]

    def updateModeSplit(self, mcc: ModalChoiceCharacteristics) -> dict:
        utils = np.array([])
        k = 1.0
        modes = mcc.modes()
        for mode in modes:
            util = 0.
            util += -mcc[mode].travel_time * self["VOT"]
            util += -mcc[mode].wait_time * self["VOT"]
            util += -mcc[mode].cost * self["VOM"]
            utils = np.append(utils, util)
        exp_utils = np.exp(utils * k)
        probabilities = exp_utils / np.sum(exp_utils)
        mode_split = dict()
        for ind in range(np.size(probabilities)):
            mode_split[modes[ind]] = probabilities[ind]
        return mode_split


class Population:
    def __init__(self):
        self.__populationGroups = dict()
        self.__demandClasses = dict()
        self.totalPopulation = 0

    def __setitem__(self, key: DemandIndex, value: DemandClass):
        self.__demandClasses[key] = value

    def __getitem__(self, item: DemandIndex) -> DemandClass:
        return self.__demandClasses[item]

    def getPopulation(self, homeMicrotypeID: str, populationGroupType: str):
        return self.__populationGroups[homeMicrotypeID, populationGroupType].population

    def importPopulation(self, populations: pd.DataFrame, populationGroups: pd.DataFrame):
        for row in populations.itertuples():
            homeMicrotypeID = row.MicrotypeID
            populationGroupType = row.PopulationGroupTypeID
            self.__populationGroups[homeMicrotypeID, populationGroupType] = PopulationGroup(homeMicrotypeID,
                                                                                            populationGroupType,
                                                                                            row.Population)
            self.totalPopulation += row.Population
        for homeMicrotypeID in populations["MicrotypeID"].unique():
            for row in populationGroups.iterrows():
                demandIndex = DemandIndex(homeMicrotypeID, row[1].PopulationGroupTypeID, row[1].TripPurposeID)
                params = row[1][2:].to_dict()
                out = DemandClass() + params
                self[demandIndex] = DemandClass() + params  # TODO: disaggregate by mode

    def __iter__(self):
        return iter(self.__demandClasses.items())
