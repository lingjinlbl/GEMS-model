import os

import numpy as np
import pandas as pd
from scipy.optimize import shgo

from utils.OD import TripCollection, OriginDestination, TripGeneration
from utils.choiceCharacteristics import CollectedChoiceCharacteristics
from utils.demand import Demand, CollectedTotalUserCosts
from utils.microtype import MicrotypeCollection, CollectedTotalOperatorCosts
from utils.misc import TimePeriods, DistanceBins
from utils.population import Population


class Optimizer:
    def __init__(self, path: str, fromSubNetworkIDs: list, toSubNetworkIDs: list):
        self.__path = path
        self.__fromSubNetworkIDs = fromSubNetworkIDs
        self.__toSubNetworkIDs = toSubNetworkIDs
        self.model = Model(path)

    def getDedicationCost(self, reallocations: np.ndarray) -> float:
        microtypes = self.model.scenarioData["subNetworkData"].loc[self.__toSubNetworkIDs, "MicrotypeID"]
        modes = self.model.scenarioData["modeToSubNetworkData"].loc[
            self.model.scenarioData["modeToSubNetworkData"]["SubnetworkID"].isin(self.__toSubNetworkIDs), "ModeTypeID"]
        perMeterCosts = self.model.scenarioData["laneDedicationCost"].loc[
            pd.MultiIndex.from_arrays([microtypes, modes]), "CostPerMeter"].values
        cost = np.sum(reallocations * perMeterCosts)
        if np.isnan(cost):
            return np.inf
        else:
            return cost

    def evaluate(self, reallocations: np.ndarray) -> float:
        modification = NetworkModification(reallocations, self.__fromSubNetworkIDs, self.__toSubNetworkIDs)
        self.model.modifyNetworks(modification)
        userCosts, operatorCosts = self.model.collectAllCosts()
        dedicationCosts = self.getDedicationCost(reallocations)
        print(reallocations)
        print(userCosts.total + operatorCosts.total + dedicationCosts)
        return userCosts.total + operatorCosts.total + dedicationCosts

    def getBounds(self):
        upperBounds = self.model.scenarioData["subNetworkData"].loc[self.__fromSubNetworkIDs, "Length"].values
        lowerBounds = [0.0] * len(self.__fromSubNetworkIDs)
        return list(zip(lowerBounds, upperBounds))
        # return Bounds(lowerBounds, upperBounds)

    def x0(self) -> list:
        return self.model.scenarioData["subNetworkData"].loc[self.__fromSubNetworkIDs, "Length"].values / 4.

    def minimize(self):
        return shgo(self.evaluate, self.getBounds())
        # return dual_annealing(self.evaluate, self.getBounds(), no_local_search=False, initial_temp=150.)
        # return minimize(self.evaluate, self.x0(), method='trust-constr', bounds=self.getBounds(),
        #                 options={'verbose': 3, 'xtol': 10.0, 'gtol': 1e-4, 'maxiter': 15, 'initial_tr_radius': 10.})


class NetworkModification:
    def __init__(self, reallocations: np.ndarray, fromSubNetworkIDs: list, toSubNetworkIDs: list):
        self.reallocations = reallocations
        self.fromSubNetworkIDs = fromSubNetworkIDs
        self.toSubNetworkIDs = toSubNetworkIDs

    def __iter__(self):
        for i in range(len(self.reallocations)):
            yield (self.fromSubNetworkIDs[i], self.toSubNetworkIDs[i]), self.reallocations[i]


class ScenarioData:
    def __init__(self, path: str, data=None):
        self.__path = path
        if data is None:
            self.data = dict()
            self.loadData()
        else:
            self.data = data

    def __setitem__(self, key: str, value):
        self.data[key] = value

    def __getitem__(self, item: str):
        return self.data[item]

    def loadModeData(self):
        collected = dict()
        (_, _, fileNames) = next(os.walk(os.path.join(self.__path, "modes")))
        for file in fileNames:
            collected[file.split(".")[0]] = pd.read_csv(os.path.join(self.__path, "modes", file),
                                                        index_col="MicrotypeID")
        return collected

    def loadData(self):
        self["subNetworkData"] = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"),
                                             index_col="SubnetworkID")
        self["modeToSubNetworkData"] = pd.read_csv(os.path.join(self.__path, "ModeToSubNetwork.csv"))
        self["microtypeAssignment"] = pd.read_csv(os.path.join(self.__path, "MicrotypeAssignment.csv"))
        self["populations"] = pd.read_csv(os.path.join(self.__path, "Population.csv"))
        self["populationGroups"] = pd.read_csv(os.path.join(self.__path, "PopulationGroups.csv"))
        self["timePeriods"] = pd.read_csv(os.path.join(self.__path, "TimePeriods.csv"))
        self["distanceBins"] = pd.read_csv(os.path.join(self.__path, "DistanceBins.csv"))
        self["originDestinations"] = pd.read_csv(os.path.join(self.__path, "OriginDestination.csv"))
        self["distanceDistribution"] = pd.read_csv(os.path.join(self.__path, "DistanceDistribution.csv"))
        self["tripGeneration"] = pd.read_csv(os.path.join(self.__path, "TripGeneration.csv"))
        self["laneDedicationCost"] = pd.read_csv(os.path.join(self.__path, "LaneDedicationCost.csv"),
                                                 index_col=["MicrotypeID", "ModeTypeID"])
        self["modeData"] = self.loadModeData()

    def copy(self):
        return ScenarioData(self.__path, self.data.copy())

    # def reallocate(self, fromSubNetwork, toSubNetwork, dist):


class Model:
    def __init__(self, path: str):
        self.__path = path
        self.scenarioData = ScenarioData(path)
        self.__initialScenarioData = ScenarioData(path)
        self.__currentTimePeriod = None
        self.__microtypes = dict()  # MicrotypeCollection(self.modeData.data)
        self.__demand = dict()  # Demand()
        self.__choice = dict()  # CollectedChoiceCharacteristics()
        self.__population = Population()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__timePeriods = TimePeriods()
        self.__tripGeneration = TripGeneration()
        self.__originDestination = OriginDestination()
        self.readFiles()

    @property
    def microtypes(self):
        if self.__currentTimePeriod not in self.__microtypes:
            self.__microtypes[self.__currentTimePeriod] = MicrotypeCollection(self.scenarioData["modeData"])
        return self.__microtypes[self.__currentTimePeriod]

    @property
    def demand(self):
        if self.__currentTimePeriod not in self.__demand:
            self.__demand[self.__currentTimePeriod] = Demand()
        return self.__demand[self.__currentTimePeriod]

    @property
    def choice(self):
        if self.__currentTimePeriod not in self.__choice:
            self.__choice[self.__currentTimePeriod] = CollectedChoiceCharacteristics()
        return self.__choice[self.__currentTimePeriod]

    def readFiles(self):
        self.__trips.importTrips(self.scenarioData["microtypeAssignment"])
        self.__population.importPopulation(self.scenarioData["populations"], self.scenarioData["populationGroups"])
        self.__timePeriods.importTimePeriods(self.scenarioData["timePeriods"])
        self.__distanceBins.importDistanceBins(self.scenarioData["distanceBins"])
        self.__originDestination.importOriginDestination(self.scenarioData["originDestinations"],
                                                         self.scenarioData["distanceDistribution"])
        self.__tripGeneration.importTripGeneration(self.scenarioData["tripGeneration"])

    def initializeTimePeriod(self, timePeriod: str):
        self.__currentTimePeriod = timePeriod
        self.microtypes.importMicrotypes(self.scenarioData["subNetworkData"], self.scenarioData["modeToSubNetworkData"])
        self.__originDestination.initializeTimePeriod(timePeriod)
        self.__tripGeneration.initializeTimePeriod(timePeriod)
        self.demand.initializeDemand(self.__population, self.__originDestination, self.__tripGeneration, self.__trips,
                                     self.microtypes, self.__distanceBins, 0.2)
        self.choice.initializeChoiceCharacteristics(self.__trips, self.microtypes, self.__distanceBins)

    def findEquilibrium(self):
        diff = 1000.
        while diff > 0.00001:
            ms = self.getModeSplit()
            self.demand.updateMFD(self.microtypes, 5)
            self.choice.updateChoiceCharacteristics(self.microtypes, self.__trips)
            diff = self.demand.updateModeSplit(self.choice, self.__originDestination, ms)

    def getModeSplit(self):
        mode_split = self.demand.getTotalModeSplit()
        return mode_split

    def getUserCosts(self):
        return self.demand.getUserCosts(self.choice, self.__originDestination)

    def getOperatorCosts(self):
        return self.microtypes.getOperatorCosts()

    def modifyNetworks(self, modification: NetworkModification):
        originalScenarioData = self.__initialScenarioData.copy()
        for ((fromNetwork, toNetwork), laneDistance) in modification:
            oldFromLaneDistance = originalScenarioData["subNetworkData"].loc[fromNetwork, "Length"]
            self.scenarioData["subNetworkData"].loc[fromNetwork, "Length"] = oldFromLaneDistance - laneDistance
            oldToLaneDistance = originalScenarioData["subNetworkData"].loc[toNetwork, "Length"]
            self.scenarioData["subNetworkData"].loc[toNetwork, "Length"] = oldToLaneDistance + laneDistance

    def collectAllCosts(self):
        userCosts = CollectedTotalUserCosts()
        operatorCosts = CollectedTotalOperatorCosts()
        for timePeriod, durationInHours in self.__timePeriods:
            self.initializeTimePeriod(timePeriod)
            self.findEquilibrium()
            userCosts += self.getUserCosts() * durationInHours
            operatorCosts += self.getOperatorCosts() * durationInHours
            # print(self.getModeSplit())
        return userCosts, operatorCosts


if __name__ == "__main__":
    o = Optimizer("input-data", [2, 4, 6, 8], [13, 14, 15, 16])
    output = o.minimize()
    print("DONE")
    print(output.x)
    print(output.fun)
    print(output.message)

    print("DONE")
    # cost2 = o.evaluate(np.array([10., 1.]))
    # print(cost2)
    # cost3 = o.evaluate(np.array([500., 10.]))
    # print(cost3)
    # a = Model("input-data")
    # a.initializeTimePeriod("AM-Peak")
    # a.findEquilibrium()
    # ms = a.getModeSplit()
    # speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
    # print(speeds)
    # userCosts = a.getUserCosts()
    # opCosts = a.getOperatorCosts()
    # mod = NetworkModification(np.array([200., 100.]), [2, 4], [13, 14])
    # a.modifyNetworks(mod)
    # userCosts2 = userCosts * 2
    # print("aah")
