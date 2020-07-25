import os

import pandas as pd

from utils.OD import TripCollection, OriginDestination, TripGeneration
from utils.choiceCharacteristics import CollectedChoiceCharacteristics
from utils.demand import Demand
from utils.microtype import MicrotypeCollection
from utils.misc import TimePeriods, DistanceBins
from utils.population import Population


class ModeData:
    def __init__(self, path: str):
        self.__path = path
        self.data = dict()
        self.loadData()

    def __setitem__(self, key: str, value: pd.DataFrame):
        self.data[key] = value

    def __getitem__(self, item: str) -> pd.DataFrame:
        return self.data[item]

    def loadData(self):
        (_, _, fileNames) = next(os.walk(os.path.join(self.__path, "modes")))
        for file in fileNames:
            self[file.split(".")[0]] = pd.read_csv(os.path.join(self.__path, "modes", file))


class ScenarioData:
    def __init__(self, path: str):
        self.__path = path
        self.data = dict()
        self.loadData()

    def __setitem__(self, key: str, value: pd.DataFrame):
        self.data[key] = value

    def __getitem__(self, item: str) -> pd.DataFrame:
        return self.data[item]

    def loadData(self):
        self["subNetworkData"] = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"))
        self["modeToSubNetworkData"] = pd.read_csv(os.path.join(self.__path, "ModeToSubNetwork.csv"))
        self["microtypeAssignment"] = pd.read_csv(os.path.join(self.__path, "MicrotypeAssignment.csv"))
        self["populations"] = pd.read_csv(os.path.join(self.__path, "Population.csv"))
        self["populationGroups"] = pd.read_csv(os.path.join(self.__path, "PopulationGroups.csv"))
        self["timePeriods"] = pd.read_csv(os.path.join(self.__path, "TimePeriods.csv"))
        self["distanceBins"] = pd.read_csv(os.path.join(self.__path, "DistanceBins.csv"))
        self["originDestinations"] = pd.read_csv(os.path.join(self.__path, "OriginDestination.csv"))
        self["distanceDistribution"] = pd.read_csv(os.path.join(self.__path, "DistanceDistribution.csv"))
        self["tripGeneration"] = pd.read_csv(os.path.join(self.__path, "TripGeneration.csv"))


class Model:
    def __init__(self, path: str):
        self.__path = path
        self.scenarioData = ScenarioData(path)
        self.modeData = ModeData(path)
        self.microtypes = MicrotypeCollection(self.modeData.data)
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
        self.microtypes.importMicrotypes(self.scenarioData["subNetworkData"], self.scenarioData["modeToSubNetworkData"])
        self.__trips.importTrips(self.scenarioData["microtypeAssignment"])
        self.__population.importPopulation(self.scenarioData["populations"], self.scenarioData["populationGroups"])
        self.__timePeriods.importTimePeriods(self.scenarioData["timePeriods"])
        self.__distanceBins.importDistanceBins(self.scenarioData["distanceBins"])
        self.__originDestination.importOriginDestination(self.scenarioData["originDestinations"],
                                                         self.scenarioData["distanceDistribution"])
        self.__tripGeneration.importTripGeneration(self.scenarioData["tripGeneration"])

    def initializeTimePeriod(self, timePeriod: str):
        self.__originDestination.initializeTimePeriod(timePeriod)
        self.__tripGeneration.initializeTimePeriod(timePeriod)
        self.demand.initializeDemand(self.__population, self.__originDestination, self.__tripGeneration, self.__trips,
                                     self.microtypes, self.__distanceBins, 0.075)
        self.choice.initializeChoiceCharacteristics(self.__trips, self.microtypes, self.__distanceBins)

    def findEquilibrium(self):
        for i in range(10):
            self.demand.updateMFD(self.microtypes)
            self.choice.updateChoiceCharacteristics(self.microtypes, self.__trips)
            self.demand.updateModeSplit(self.choice, self.__originDestination)

    def getModeSplit(self):
        mode_split = self.demand.getTotalModeSplit()
        return mode_split

    def getCosts(self):
        return self.demand.getUserCosts(self.choice, self.__originDestination)


if __name__ == "__main__":
    a = Model("input-data")
    a.initializeTimePeriod("AM-Peak")
    a.findEquilibrium()
    ms = a.getModeSplit()
    speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
    print(speeds)
    costs = a.getCosts()
    print("aah")
