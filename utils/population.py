import pandas as pd
import os
from utils.microtype import Microtype
from utils.network import Network, NetworkCollection, NetworkFlowParams, BusModeParams, \
    AutoModeParams, Costs
from utils.OD import Trip, TripCollection


class PopulationGroup:
    def __init__(self, homeLocation: str, populationGroupType: str, population: float):
        self.homeLocation = homeLocation
        self.populationGroupType = populationGroupType
        self.population = population


class Population:
    def __init__(self):
        self.__populationGroups = dict()
        self.totalPopulation = 0

    def __setitem__(self, key: (str, str), value: PopulationGroup):
        self.__populationGroups[key] = value

    def __getitem__(self, item: (str, str)) -> PopulationGroup:
        return self.__populationGroups[item]

    def importPopulation(self, df: pd.DataFrame):
        for row in df.itertuples():
            homeMicrotypeID = row.MicrotypeID
            populationGroupType = row.PopulationGroupTypeID
            self[homeMicrotypeID, populationGroupType] = PopulationGroup(homeMicrotypeID, populationGroupType,
                                                                         row.Population)
            self.totalPopulation += row.Population
