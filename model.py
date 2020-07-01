import pandas as pd
import os
from utils.microtype import MicrotypeCollection
from utils.network import Network, NetworkCollection, NetworkFlowParams, BusModeParams, \
    AutoModeParams, Costs
from utils.OD import Trip, TripCollection, OriginDestination, TripGeneration
from utils.population import PopulationGroup, Population
from utils.demand import Demand
from utils.misc import TimePeriods, DistanceBins
from typing import Dict, List


class Model:
    def __init__(self, path: str):
        self.__path = path
        self.microtypes = MicrotypeCollection(path)
        self.population = Population()
        self.demand = Demand()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__timePeriods = TimePeriods()
        self.__tripGeneration = TripGeneration()
        self.__originDestination = OriginDestination()
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
        self.demand.initializeDemand(self.population, self.__originDestination, self.__tripGeneration, self.__trips,
                                     self.microtypes, self.__distanceBins)


if __name__ == "__main__":
    a = Model("input-data")
    a.initializeTimePeriod("AM-Peak")
    a.initializeDemand()
    print("aah")
