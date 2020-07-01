import pandas as pd
import os
from utils.microtype import Microtype
from utils.network import Network, NetworkCollection, NetworkFlowParams, BusModeParams, \
    AutoModeParams, Costs
from utils.OD import Trip, TripCollection, DemandIndex
from utils.OD import Trip, TripCollection, OriginDestination, TripGeneration


class PopulationGroup:
    def __init__(self, homeLocation: str, populationGroupType: str, population: float):
        self.homeLocation = homeLocation
        self.populationGroupType = populationGroupType
        self.population = population


class Population:
    def __init__(self):
        self.__populationGroups = dict()
        self.__demandClasses = dict()
        self.totalPopulation = 0

    def __setitem__(self, key: DemandIndex, value: dict):
        self.__demandClasses[key] = value

    def __getitem__(self, item: DemandIndex) -> dict:
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
                self[demandIndex] = params

    def __iter__(self):
        return iter(self.__demandClasses.items())


