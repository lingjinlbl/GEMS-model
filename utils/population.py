from typing import Dict

import numpy as np
import pandas as pd

from utils.OD import DemandIndex, OriginDestination
from utils.choiceCharacteristics import ModalChoiceCharacteristics, CollectedChoiceCharacteristics



class PopulationGroup:
    def __init__(self, homeLocation: str, populationGroupType: str, population: float):
        self.homeLocation = homeLocation
        self.populationGroupType = populationGroupType
        self.population = population


class DemandClass:
    def __init__(self):
        self.__params = {"ASC": 0.0, "VOT": 15.0, "VOM": 1.0}

    def __iadd__(self, other: Dict[str, float]):
        self.__params.update(other)

    def __add__(self, other: Dict[str, float]):
        self.__params.update(other)
        return self

    def __getitem__(self, item) -> float:
        return self.__params[item]

    def updateModeSplit(self, mcc: ModalChoiceCharacteristics) -> Dict[str, float]:
        utils = np.array([], dtype=float)
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

    def getCostPerCapita(self, mcc: ModalChoiceCharacteristics, modeSplit, params=None) -> float:
        if params is None:
            params = self.__params
        costPerCapita = 0.
        for mode, split in modeSplit:
            costPerCapita += mcc[mode].travel_time * params["VOT"] * split
            costPerCapita += mcc[mode].wait_time * params["VOT"] * split
            costPerCapita += mcc[mode].cost * params["VOM"] * split
        return costPerCapita


class Population:
    def __init__(self):
        self.__populationGroups = dict()
        self.__demandClasses = dict()
        self.__totalCosts = dict()
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

    # def getCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
    #              originDestination: OriginDestination) -> CollectedTotalCosts:
    #     collectedTotalCosts = CollectedTotalCosts()
    #     for di, dc in self.__demandClasses.items():
    #         od = originDestination[di]
    #
    #         for odi, portion in od.items():
    #             costs = dc.getCostPerCapita(collectedChoiceCharacteristics[od])
    #         collectedTotalCosts[di] = TotalCosts
    #         print("aah")
    #     return collectedTotalCosts
