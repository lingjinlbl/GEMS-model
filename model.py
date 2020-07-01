import pandas as pd
import os
from utils.microtype import MicrotypeCollection
from utils.network import Network, NetworkCollection, NetworkFlowParams, BusModeParams, \
    AutoModeParams, Costs
from utils.OD import Trip, TripCollection, OriginDestination, TripGeneration
from utils.population import PopulationGroup, Population
from utils.demand import Demand
from typing import Dict, List


class TimePeriods:
    def __init__(self):
        self.__timePeriods = dict()

    def __setitem__(self, key: str, value: float):
        self.__timePeriods[key] = value

    def __getitem__(self, item) -> float:
        return self.__timePeriods[item]

    def importTimePeriods(self, df: pd.DataFrame):
        for row in df.itertuples():
            self[row.TimePeriodID] = row.DurationInHours


class DistanceBins:
    def __init__(self):
        self.__distanceBins = dict()

    def __setitem__(self, key: str, value: float):
        self.__distanceBins[key] = value

    def __getitem__(self, item) -> float:
        return self.__distanceBins[item]

    def importDistanceBins(self, df: pd.DataFrame):
        for row in df.itertuples():
            self[row.DistanceBinID] = row.MeanDistanceInMiles


class Model:
    def __init__(self, path: str):
        self.__path = path
        self.microtypes = MicrotypeCollection(path)
        self.population = Population()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__timePeriods = TimePeriods()
        self.__tripGeneration = TripGeneration()
        self.__originDestination = OriginDestination()
        self.__demand = Demand()
        self.readFiles()

    def readFiles(self):
        subNetworkData = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"))
        modeToSubNetworkData = pd.read_csv(os.path.join(self.__path, "ModeToSubNetwork.csv"))
        self.microtypes.importMicrotypes(subNetworkData, modeToSubNetworkData)

        self.__trips.importTrips(pd.read_csv(os.path.join(self.__path, "MicrotypeAssignment.csv")))

        populations = pd.read_csv(os.path.join(self.__path, "Population.csv"))
        populationGroups = pd.read_csv(os.path.join(self.__path, "PopulationGroups.csv"))
        self.population.importPopulation(populations, populationGroups)

        self.__timePeriods.importTimePeriods(pd.read_csv(os.path.join(self.__path, "TimePeriods.csv")))

        self.__distanceBins.importDistanceBins(pd.read_csv(os.path.join(self.__path, "DistanceBins.csv")))

        originDestinations = pd.read_csv(os.path.join(self.__path, "OriginDestination.csv"))
        distanceDistribution = pd.read_csv(os.path.join(self.__path, "DistanceDistribution.csv"))
        self.__originDestination.importOriginDestination(originDestinations, distanceDistribution)

        tripGeneration = pd.read_csv(os.path.join(self.__path, "TripGeneration.csv"))
        self.__tripGeneration.importTripGeneration(tripGeneration)

    def initializeTimePeriod(self, timePeriod: str):
        self.__originDestination.initializeTimePeriod(timePeriod)
        self.__tripGeneration.initializeTimePeriod(timePeriod)

    def initializeDemand(self):
        self.__demand.initializeDemand(self.__originDestination)


if __name__ == "__main__":
    a = Model("input-data")
    a.initializeTimePeriod("AM-Peak")
    print("aah")
