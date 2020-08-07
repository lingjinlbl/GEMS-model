from typing import Dict

import numpy as np
import pandas as pd

from utils.OD import DemandIndex
from utils.choiceCharacteristics import ModalChoiceCharacteristics


class PopulationGroup:
    def __init__(self, homeLocation: str, populationGroupType: str, population: float):
        self.homeLocation = homeLocation
        self.populationGroupType = populationGroupType
        self.population = population


class DemandClass:
    def __init__(self, params: pd.DataFrame):
        self.__params = params

    # def __iadd__(self, other: Dict[str, float]):
    #     self.__params.update(other)
    #
    # def __add__(self, other: Dict[str, float]):
    #     self.__params.update(other)
    #     return self

    def __getitem__(self, item) -> float:
        item1, item2 = item
        if item1 in self.__params.index and item2 in self.__params.columns:
            return self.__params.loc[item1, item2]
        else:
            return 0.0

    def updateModeSplit(self, mcc: ModalChoiceCharacteristics) -> Dict[str, float]:
        utils = np.array([], dtype=float)
        k = 1.0
        modes = mcc.modes()
        for mode in modes:
            util = 0.
            util += self[mode, "Intercept"]
            util += mcc[mode].travel_time * self[mode, "BetaTravelTime"]
            util += mcc[mode].wait_time * self[mode, "BetaWaitTime"]
            util += mcc[mode].cost * self[mode, "VOM"]
            utils = np.append(utils, util)
        exp_utils = np.exp(utils * k)
        probabilities = exp_utils / np.sum(exp_utils)
        mode_split = dict()
        for ind in range(np.size(probabilities)):
            mode_split[modes[ind]] = probabilities[ind]
        return mode_split

    def getCostPerCapita(self, mcc: ModalChoiceCharacteristics, modeSplit, params=None) -> float:
        if params is not None:
            params = DemandClass(params)
        else:
            params = self
        costPerCapita = 0.
        for mode, split in modeSplit:
            costPerCapita += params[mode, "Intercept"] * split
            costPerCapita += mcc[mode].travel_time * params[mode, "BetaTravelTime"] * split
            costPerCapita += mcc[mode].wait_time * params[mode, "BetaWaitTime"] * split
            costPerCapita += mcc[mode].cost * params[mode, "VOM"] * split
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
            for (groupId, tripPurpose), group in populationGroups.groupby(['PopulationGroupTypeID', 'TripPurposeID']):
                demandIndex = DemandIndex(homeMicrotypeID, groupId, tripPurpose)
                out = DemandClass(group.set_index("mode").drop(columns=['PopulationGroupTypeID','TripPurposeID']))
                self[demandIndex] = out

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
