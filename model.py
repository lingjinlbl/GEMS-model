import os
import pandas as pd
from utils.OD import TripCollection, OriginDestination, TripGeneration
from utils.choiceCharacteristics import CollectedChoiceCharacteristics
from utils.demand import Demand
from utils.microtype import MicrotypeCollection
from utils.misc import TimePeriods, DistanceBins
from utils.population import Population


class Model:
    def __init__(self, path: str):
        self.__path = path
        self.microtypes = MicrotypeCollection(path)
        self.demand = Demand()
        self.choice = CollectedChoiceCharacteristics()
        self.__population = Population()
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
        self.__population.importPopulation(populations, populationGroups)

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
        self.demand.initializeDemand(self.__population, self.__originDestination, self.__tripGeneration, self.__trips,
                                     self.microtypes, self.__distanceBins)
        self.choice.initializeChoiceCharacteristics(self.__trips, self.microtypes, self.__distanceBins)

    def findEquilibrium(self):
        for i in range(20):
            self.demand.updateMFD(self.microtypes)
            self.choice.updateChoiceCharacteristics(self.microtypes, self.__trips)
            self.demand.updateModeSplit(self.choice, self.__originDestination)


if __name__ == "__main__":
    a = Model("input-data")
    a.initializeTimePeriod("AM-Peak")
    a.findEquilibrium()
    print("aah")
