# from line_profiler_pycharm import profile
from sys import stdout

import ipywidgets as widgets
import numpy as np
import pandas as pd
from noisyopt import minimizeCompass, minimizeSPSA
from scipy.optimize import root, minimize, Bounds, shgo
from mock import Mock
# from skopt import gp_minimize, forest_minimize

from utils.OD import TripCollection, OriginDestination, TripGeneration, TransitionMatrices
from utils.choiceCharacteristics import CollectedChoiceCharacteristics
from utils.demand import Demand, Externalities, modeSplitFromUtilsWithExcludedModes
from utils.interact import Interact
from utils.microtype import MicrotypeCollection, CollectedTotalOperatorCosts
from utils.misc import TimePeriods, DistanceBins
from utils.population import Population
from utils.data import Data, ScenarioData


class TransitScheduleModification:
    def __init__(self, headways: np.ndarray, modesAndMicrotypes: list):
        self.headways = headways
        self.modesAndMicrotypes = modesAndMicrotypes

    def __iter__(self):
        for i in range(len(self.headways)):
            yield (self.modesAndMicrotypes[i]), self.headways[i]


class NetworkModification:
    def __init__(self, reallocations: np.ndarray, modesAndMicrotypeIDs: list):
        self.reallocations = reallocations
        self.modesAndMicrotypeIDs = modesAndMicrotypeIDs

    def __iter__(self):
        for i in range(len(self.reallocations)):
            yield (self.modesAndMicrotypeIDs[i]), self.reallocations[i]


class Model:
    """
    A class representing the GEMS Model.

    ...

    Attributes
    ----------
    path : str
        File path to input data
    scenarioData : ScenarioData
        Class object to fetch and store mode and parameter data
    initialScenarioData : ScenarioData
        Initial state of the scenario
    currentTimePeriod : str
        Description of the current time period (e.g. 'AM-Peak')
    microtypes : dict(str, MicrotypeCollection)
        Stores currentTimePeriod to MicrotypeCollection object
    demand : dict(str, Demand)
        Stores currentTimePeriod to Demand object
    choice : dict(str, CollectedChoiceCharacteristics)
        Stores currentTimePeriod to CollectedChoiceCharacteristics object
    population : Population
        Stores demandClasses, populationGroups, and totalCosts
    trips : TripCollection
        Stores a collection of trips through microtypes
    distanceBins : DistanceBins
        Set of distance bins for different distances of trips
    timePeriods  : TimePeriods
        Contains amount of time in each time period
    tripGeneration : TripGeneration
        Contains initialized trips with all the sets and classes
    originDestination : OriginDestination
        Stores origin/destination form of trips

    Methods
    -------
    microtypes:
        Returns the microtypes loaded into the model
    demand:
        Returns demand of the trips
    choice:
        Returns a choiceCharacteristic object
    readFiles():
        Initializes the model with file data
    initializeTimePeriod:
        Initializes the model with time periods
    findEquilibrium():
        Finds the ideal mode splits for the model
    getModeSplit(timePeriod=None, userClass=None, microtypeID=None, distanceBin=None):
        Returns the optimal mode splits
    getUserCosts(mode=None):
        Returns total user costs
    getModeUserCosts():
        Returns costs per mode found by the model
    getOperatorCosts():
        Returns total costs to the operator
    modifyNetworks(networkModification=None, scheduleModification=None):
        Used in the optimizer class to edit network after initialization
    resetNetworks():
        Reset network to original initialization
    setTimePeriod(timePeriod: str):

    getModeSpeeds(timePeriod=None):
        Returns speeds for each mode in each microtype
    """

    def __init__(self, path: str, nSubBins=2, interactive=False):
        self.__path = path
        self.__nSubBins = nSubBins
        self.__timeStepInSeconds = 15.0
        self.scenarioData = ScenarioData(path, self.__timeStepInSeconds)
        self.data = Data(self.scenarioData, self.__nSubBins, self.__timeStepInSeconds)
        self.__fixedData = self.data.getInvariants()

        self.__initialScenarioData = ScenarioData(path, self.__timeStepInSeconds)
        self.__currentTimePeriod = None
        self.__microtypes = dict()  # MicrotypeCollection(self.modeData.data)
        self.__demand = dict()  # Demand()
        self.__choice = dict()  # CollectedChoiceCharacteristics()
        self.__population = Population(self.scenarioData, self.__fixedData)
        self.__distanceBins = DistanceBins()
        self.__timePeriods = TimePeriods()
        self.__tripGeneration = TripGeneration()
        self.__transitionMatrices = TransitionMatrices(self.scenarioData)
        self.__originDestination = OriginDestination(self.__timePeriods, self.__distanceBins, self.__population,
                                                     self.__transitionMatrices, self.__fixedData,
                                                     self.scenarioData)
        self.__externalities = Externalities(self.scenarioData)
        self.__printLoc = stdout
        self.__interactive = interactive
        self.__tolerance = 2e-11
        self.interact = Interact(self, figure=interactive)
        self.readFiles()
        self.initializeAllTimePeriods()
        self.__successful = True
        if interactive:
            self.interact.init()

    @property
    def nSubBins(self):
        return self.__nSubBins

    @property
    def diToIdx(self):
        return self.scenarioData.diToIdx

    @property
    def odiToIdx(self):
        return self.scenarioData.odiToIdx

    @property
    def modeToIdx(self):
        return self.scenarioData.modeToIdx

    @property
    def passengerModeToIdx(self):
        return self.scenarioData.passengerModeToIdx

    @property
    def dataToIdx(self):
        return self.scenarioData.demandDataTypeToIdx

    @property
    def microtypeIdToIdx(self):
        return self.scenarioData.microtypeIdToIdx

    @property
    def currentTimePeriod(self):
        return self.__currentTimePeriod

    @property
    def microtypes(self):
        if self.__currentTimePeriod not in self.__microtypes:
            self.__microtypes[self.__currentTimePeriod] = MicrotypeCollection(self.scenarioData, self.data.getSupply(
                self.__currentTimePeriod))
        return self.__microtypes[self.__currentTimePeriod]

    def getMicrotypeCollection(self, timePeriod) -> MicrotypeCollection:
        return self.__microtypes[timePeriod]

    @property
    def demand(self):
        if self.__currentTimePeriod not in self.__demand:
            self.__demand[self.__currentTimePeriod] = Demand(self.scenarioData,
                                                             self.data.getDemand(self.__currentTimePeriod))
        return self.__demand[self.__currentTimePeriod]

    @property
    def choice(self):
        if self.__currentTimePeriod not in self.__choice:
            self.__choice[self.__currentTimePeriod] = CollectedChoiceCharacteristics(self.scenarioData, self.demand,
                                                                                     self.data.getDemand(
                                                                                         self.__currentTimePeriod),
                                                                                     self.__fixedData)
        return self.__choice[self.__currentTimePeriod]

    @property
    def successful(self):
        return self.__successful

    def getCurrentTimePeriodDuration(self):
        return self.__timePeriods[self.currentTimePeriod]

    def readFiles(self):
        # self.__trips.importTrips(self.scenarioData["microtypeAssignment"])
        self.__population.importPopulation(self.scenarioData["populations"], self.scenarioData["populationGroups"])
        self.__timePeriods.importTimePeriods(self.scenarioData["timePeriods"], nSubBins=self.__nSubBins)
        self.__distanceBins.importDistanceBins(self.scenarioData["distanceBins"])
        self.__transitionMatrices.importTransitionMatrices(self.scenarioData["transitionMatrices"],
                                                           self.scenarioData["microtypeIDs"],
                                                           self.scenarioData["distanceBins"])
        self.__originDestination.importOriginDestination(self.scenarioData["originDestinations"],
                                                         self.scenarioData["distanceDistribution"],
                                                         self.scenarioData["modeAvailability"])
        self.__tripGeneration.importTripGeneration(self.scenarioData["tripGeneration"])

    def initializeTimePeriod(self, timePeriod: int, override=False):
        self.__currentTimePeriod = timePeriod
        # if timePeriod not in self.__microtypes:
        #     print("-------------------------------")
        #     print("|  Loading time period ", timePeriod, " ", self.__timePeriods.getTimePeriodName(timePeriod))
        self.microtypes.importMicrotypes(override)
        self.__originDestination.initializeTimePeriod(timePeriod, self.__timePeriods.getTimePeriodName(timePeriod))
        self.__tripGeneration.initializeTimePeriod(timePeriod, self.__timePeriods.getTimePeriodName(timePeriod))
        self.demand.initializeDemand(self.__population, self.__originDestination, self.__tripGeneration,
                                     self.microtypes, self.__distanceBins, self.__transitionMatrices,
                                     self.__timePeriods, self.__currentTimePeriod, dict(), 1.0)
        self.choice.initializeChoiceCharacteristics(self.microtypes, self.__distanceBins)

    def initializeAllTimePeriods(self, override=False):
        self.__externalities.init()
        self.__successful = True
        for timePeriod, durationInHours in self.__timePeriods:
            self.initializeTimePeriod(timePeriod, override)
        self.data.updateMicrotypeNetworkLength()

    def fromVector(self, flatUtilitiesArray):
        return np.reshape(flatUtilitiesArray, self.demand.currentUtilities().shape)

    def supplySide(self, modeSplitArray):
        self.demand.updateMFD(self.microtypes, modeSplitArray=modeSplitArray)
        choiceCharacteristicsArray = self.choice.updateChoiceCharacteristics(self.microtypes)
        return choiceCharacteristicsArray

    def demandSide(self, choiceCharacteristicsArray):
        utilitiesArray = self.demand.calculateUtilities(choiceCharacteristicsArray)
        modeSplitArray = modeSplitFromUtilsWithExcludedModes(utilitiesArray, self.__fixedData['toTransitLayer'])
        return modeSplitArray

    def toObjectiveFunction(self, modeSplitArray):
        return modeSplitArray[:, :, :-1].reshape(-1)

    def fromObjectiveFunction(self, flatModeSplitArray):
        modeSplitArray = np.zeros_like(self.demand.modeSplitData)
        modeSplitArray[:, :, :-1] = flatModeSplitArray.reshape(list(self.demand.modeSplitData.shape[:-1]) + [-1]).clip(
            0, 1)
        modeSplitArray[:, :, -1] = 1 - modeSplitArray.sum(axis=2)
        return modeSplitArray

    def f(self, flatModeSplitArray):
        choiceCharacteristicsArray = self.supplySide(self.fromObjectiveFunction(flatModeSplitArray))
        modeSplitArray = self.demandSide(choiceCharacteristicsArray)
        return self.toObjectiveFunction(modeSplitArray)

    def g(self, flatModeSplitArray):
        output = self.f(flatModeSplitArray)
        diff = output - flatModeSplitArray
        diff[np.isnan(diff)] = 0.0
        diff += flatModeSplitArray.clip(None, 0) * 1e2
        modeSplitError = self.fromObjectiveFunction(flatModeSplitArray)
        modeSplitError[:, :, :-1] = flatModeSplitArray.reshape(list(self.demand.modeSplitData.shape[:-1]) + [-1]).clip(
            None, 0)
        diff += self.toObjectiveFunction(modeSplitError) * 1e1
        # diff[(flatModeSplitArray < 0) | (output < 0)] *= 100.
        return diff

    def timePeriods(self):
        return {a: b for a, b in self.__timePeriods}

    def timePeriodNames(self):
        return {a: self.__timePeriods.getTimePeriodName(a) for a, _ in self.__timePeriods}

    def findEquilibrium(self):

        """
        Initial conditions
        """

        # self.demand.updateMFD(self.microtypes)
        # self.choice.updateChoiceCharacteristics(self.microtypes, self.__trips)
        # self.demand.updateModeSplit(self.choice, self.__originDestination)

        """
        Optimization loop
        """

        startingPoint = self.toObjectiveFunction(self.demand.modeSplitData)

        if np.linalg.norm(self.g(startingPoint)) < self.__tolerance * 10.:
            fixedPointModeSplit = self.fromObjectiveFunction(startingPoint)
            success = True
        else:
            sol = root(self.g, startingPoint, method='df-sane', tol=self.__tolerance,
                       options={'maxfev': 1000, 'maxiter': 500, 'line_search': 'cheng', 'sigma_0': -0.8})
            self.g(sol.x)
            fixedPointModeSplit = self.fromObjectiveFunction(sol.x)
            success = sol.success
            if not success:
                print("Convergence didn't finish")
        self.__successful = self.__successful & success

        """
        Finalize
        """
        self.demand.updateMFD(self.microtypes, modeSplitArray=fixedPointModeSplit)
        self.choice.updateChoiceCharacteristics(self.microtypes)

    def getModeSplit(self, timePeriod=None, userClass=None, microtypeID=None, distanceBin=None, weighted=False):
        # TODO: allow subset of modesplit by userclass, microtype, distance, etc.
        if timePeriod is None:
            tripRate = self.data.tripRate()
            modeSplits = self.data.modeSplit()
            toStarts = self.data.toStarts()
            if microtypeID is None:
                modeSplit = np.einsum('tijk,tij->k', modeSplits, tripRate)
            else:
                mask = (toStarts[:, self.microtypeIdToIdx[microtypeID]] == 1)
                modeSplit = np.einsum('tijk,tij->k',
                                      modeSplits[:, :, mask, :],
                                      tripRate[:, :, mask])
        else:
            tripRate = self.data.tripRate(timePeriod)
            modeSplits = self.data.modeSplit(timePeriod)
            toStarts = self.data.toStarts()
            if microtypeID is None:
                modeSplit = np.einsum('ijk,ij->k', modeSplits, tripRate)
            else:
                modeSplit = np.einsum('ijk,ij->k',
                                      modeSplits[:, (toStarts[:, self.microtypeIdToIdx[microtypeID]] == 1)],
                                      tripRate[:, (toStarts[:, self.microtypeIdToIdx[microtypeID]] == 1)])
        return modeSplit / np.sum(modeSplit)

    def getModePMT(self, timePeriod=None, userClass=None, microtypeID=None, distanceBin=None, weighted=False):
        # TODO: allow subset of modesplit by userclass, microtype, distance, etc.
        if timePeriod is None:
            modeSplit = np.zeros(len(self.modeToIdx))
            for tp, weight in self.__timePeriods:
                if tp in self.__demand:
                    if microtypeID is None:
                        modeSplit += self.__microtypes[tp].passengerDistanceByMode.sum(axis=0) * weight
                    else:
                        modeSplit += self.__microtypes[tp].passengerDistanceByMode[self.microtypeIdToIdx[microtypeID],
                                     :] * weight
        else:
            if microtypeID is None:
                modeSplit = self.__microtypes[timePeriod].passengerDistanceByMode.sum(axis=0) * self.__timePeriods[
                    timePeriod]
            else:
                modeSplit = self.__microtypes[timePeriod].passengerDistanceByMode[self.microtypeIdToIdx[microtypeID],
                            :] * self.__timePeriods[timePeriod]
        return modeSplit

    """
    DEPRECIATED
    def getUserCosts(self, mode=None):
        return self.demand.getUserCosts(self.choice, self.__originDestination)
    """

    def getMatrixUserCosts(self):
        return self.demand.getMatrixUserCosts(self.choice)

    def getMatrixSummedCharacteristics(self):
        return self.demand.getSummedCharacteristics(self.choice)

    def getModeUserCosts(self):
        out = dict()
        for mode in self.scenarioData['modeData'].keys():
            out[mode] = self.getUserCosts([mode]).toDataFrame()
        return pd.concat(out)

    def getOperatorCosts(self):
        # d = self.data.getDemand()
        # tripRateByMode = d['tripRate'][:, :, :, None] * d['modeSplit']
        # costPerTrip = d['choiceCharacteristics'][:, :, :, :, self.scenarioData.paramToIdx['cost']]
        # totalCosts = tripRateByMode * costPerTrip
        # costsByStartMicrotype = np.einsum('ijkl,km->iml', totalCosts, d['toStarts'])
        # timePeriods = model.scenarioData['timePeriods']['TimePeriodID'].values
        # a1, a2, a3 = np.meshgrid(timePeriods, list(self.microtypeIdToIdx.keys()), list(self.modeToIdx.keys()))
        # revenues = pd.DataFrame(costsByStartMicrotype.flatten(), index=[a1.flatten(), a2.flatten(), a3.flatten()],
        #                         columns=['cost'])
        # s = self.data.getSupply()
        # modeMicrotypeAccumulation = np.einsum('ijk,mj->ikm', s['subNetworkAccumulation'], s['subNetworkToMicrotype'])
        return self.microtypes.getOperatorCosts()

    def getFreightOperatorCosts(self):
        return self.microtypes.getFreightOperatorCosts()

    def modifyNetworks(self, networkModification=None,
                       scheduleModification=None):
        # originalScenarioData = self.__initialScenarioData.copy()
        if networkModification is not None:
            for ((microtypeID, modeName), laneDistance) in networkModification:
                self.interact.modifyModel(changeType=('dedicated', (microtypeID, modeName)), value=laneDistance)
                # oldFromLaneDistance = originalScenarioData["subNetworkData"].loc[fromNetwork, "Length"]
                # self.scenarioData["subNetworkData"].loc[fromNetwork, "Length"] = oldFromLaneDistance - laneDistance
                # oldToLaneDistance = originalScenarioData["subNetworkData"].loc[toNetwork, "Length"]
                # self.scenarioData["subNetworkData"].loc[toNetwork, "Length"] = oldToLaneDistance + laneDistance

        if scheduleModification is not None:
            for ((microtypeID, modeName), newHeadway) in scheduleModification:
                self.interact.modifyModel(changeType=('headway', (microtypeID, modeName)), value=newHeadway)
                # self.scenarioData["modeData"][modeName].loc[microtypeID, "Headway"] = newHeadway

    def resetNetworks(self):
        self.scenarioData = self.__initialScenarioData.copy()

    def updateTimePeriodDemand(self, timePeriodId, newTripStartRate):
        self.__demand[timePeriodId].updateTripStartRate(newTripStartRate)

    def setTimePeriod(self, timePeriod: int, init=False, preserve=False):
        """Note: Are we always going to go through them in order? Should maybe just store time periods
        as a dataframe and go by index. But, we're not keeping track of all accumulations so in that sense
        we always need to go in order."""
        self.__currentTimePeriod = timePeriod
        self.__originDestination.setTimePeriod(timePeriod)
        self.__tripGeneration.setTimePeriod(timePeriod)

    def collectAllCosts(self, event=None):
        operatorCosts = CollectedTotalOperatorCosts()
        freightOperatorCosts = CollectedTotalOperatorCosts()
        externalities = dict()
        vectorUserCosts = dict()
        for timePeriod, durationInHours in self.__timePeriods:
            self.setTimePeriod(timePeriod, preserve=True)
            matCosts = self.getMatrixUserCosts() * durationInHours
            vectorUserCosts[timePeriod] = matCosts
            operatorCosts += self.getOperatorCosts() * durationInHours
            freightOperatorCosts += self.getFreightOperatorCosts() * durationInHours
            externalities[timePeriod] = self.__externalities.calcuate(self.microtypes) * durationInHours
        return operatorCosts, freightOperatorCosts, vectorUserCosts, externalities

    def updatePopulation(self):
        for timePeriod, durationInHours in self.__timePeriods:
            self.setTimePeriod(timePeriod, preserve=True)
            self.demand.updateTripGeneration(self.microtypes)

    def collectAllCharacteristics(self):
        vectorUserCosts = 0.0
        init = True
        utilities = []
        keepGoing = True
        self.__successful = True
        for timePeriod, durationInHours in self.__timePeriods:
            self.setTimePeriod(timePeriod, init)
            if keepGoing:
                self.microtypes.updateNetworkData()
                init = False
                self.findEquilibrium()
                matCosts = self.getMatrixSummedCharacteristics() * durationInHours
                vectorUserCosts += matCosts
                # self.__networkStateData[timePeriod] = self.microtypes.getStateData()
                utility = self.demand.utility(self.choice)
                utilities.append(utility)
                if not self.successful:
                    keepGoing = False
                    print('SHOULD I BE BROKEN?')
            else:
                # self.__networkStateData[timePeriod] = self.microtypes.getStateData().resetAll()
                vectorUserCosts *= np.nan
                utilities.append(utility * np.nan)
        return vectorUserCosts, np.stack(utilities)

    def toPandas(self):
        modeSplitData = dict()
        speedData = dict()
        utilityData = dict()
        for timePeriod, durationInHours in self.__timePeriods:
            msd = self.__demand[timePeriod].modeSplitDataFrame()
            modeSplitData[timePeriod] = msd

            sd = self.__microtypes[timePeriod].dataByModeDataFrame()
            sd['TotalTripStarts'] = sd['TripStartsPerHour'] * durationInHours
            sd['TotalTripEnds'] = sd['TripEndsPerHour'] * durationInHours
            sd['TotalPassengerDistance'] = sd['PassengerDistancePerHour'] * durationInHours
            sd['TotalVehicleDistance'] = sd['VehicleDistancePerHour'] * durationInHours
            speedData[timePeriod] = sd

            ud = self.__choice[timePeriod].toDataFrame()
            utilityData[timePeriod] = ud

        modeSplitData = pd.concat(modeSplitData, axis=1)
        speedData = pd.concat(speedData, axis=1)
        utilityData = pd.concat(utilityData, axis=1)

        modeSplitData.columns.set_names(['Time Period', 'Value'], inplace=True)
        speedData.columns.set_names(['Time Period', 'Value'], inplace=True)
        utilityData.columns.set_names(['Time Period', 'Value', 'Parameter'], inplace=True)

        return modeSplitData, speedData, utilityData

    def getModeSpeeds(self, timePeriod=None):
        if timePeriod is None:
            timePeriod = self.__currentTimePeriod
        return pd.DataFrame(self.__microtypes[timePeriod].getModeSpeeds()) * 3600. / 1609.34

    def plotAllDynamicStats(self, type, microtype=None):
        ts = []
        vs = []
        ns = []
        prods = []
        inflows = []
        outflows = []
        matrices = []
        for id, dur in self.__timePeriods:
            # out = self.getMicrotypeCollection(id).transitionMatrixMFD(dur, self.getNetworkStateData(id),
            #                                                           self.getMicrotypeCollection(
            #                                                               id).getModeStartRatePerSecond("auto"))
            # sd = self.getNetworkStateData(id)
            pass
            # t, v, n, inflow, outflow, matrix, label = sd.getAutoSpeeds()
            # prods.append(sd.getAutoProduction())
            # inflows.append(inflow)
            # outflows.append(outflow)
            # ts.append(t)
            # vs.append(v)
            # ns.append(n)
            # matrices.append(matrix)

        if type.lower() == "n":
            # x = np.concatenate(ts) / 3600
            # y = np.concatenate(ns)
            # plt.plot(x, y)
            x = self.data.t
            y = self.data.n
            return x, y.transpose()
        elif type.lower().startswith('mat'):
            x = np.concatenate(ts) / 3600
            y = np.swapaxes(np.concatenate(matrices, axis=1), 0, 1)
            return x, y
        elif type.lower() == "v":
            # x = np.concatenate(ts) / 3600.
            # y = np.concatenate(vs) * 3600. / 1609.34
            x = self.data.t / 3600.
            y = self.data.v * 3600. / 1609.34
            # plt.plot(x, y)
            return x, y.transpose()
        elif type.lower() == "delay":
            y1 = np.cumsum(np.concatenate(inflows, axis=0), axis=0)
            y2 = np.cumsum(np.concatenate(outflows, axis=0), axis=0)
            x = np.concatenate(ts) / 3600.
            # plt.plot(x, y1)
            # plt.plot(x, y2, linestyle='--')
            # colors = ['C0', 'C1','C2','C3','C0', 'C1','C2','C3']
            # for i, j in enumerate(plt.gca().lines):
            #     j.set_color(colors[i])
            return x, (np.stack([y1, y2]))
        # elif type.lower() == "density":
        #     x = np.concatenate(ts)
        #     y = np.concatenate(reldensity)
        #     # plt.plot(x, y)
        #     return x, y
        elif type.lower() == "modes":
            if microtype is None:
                x = np.cumsum([0] + [val for val in self.timePeriods().values()])
                y = np.vstack([self.getModeSplit(0)] + [self.getModeSplit(p) for p in self.timePeriods().keys()])
                return x, y
            else:
                x = np.cumsum([0] + [val for val in self.timePeriods().values()])
                y = np.vstack(
                    [self.getModeSplit(0, microtypeID=microtype)] + [self.getModeSplit(p, microtypeID=microtype) for p
                                                                     in self.timePeriods().keys()])
                return x, y
        # elif type.lower() == "cartrips":
        #     autoProdsMFD = [0.]
        #     autoProdsDemand = [0.]
        #     for p in self.timePeriods().keys():
        #         sd = self.getNetworkStateData(p)
        #         autoProdsMFD.append(sd.getAutoProduction().sum(axis=1).sum() / 1609.34)
        #         autoProdsDemand.append(self.getModePMT(p)[self.modeToIdx['auto']])
        #
        #     x = np.cumsum([0] + [val for val in self.timePeriods().values()])
        #     y = np.vstack([autoProdsMFD, autoProdsDemand]).transpose()
        #     return x, y
        elif type.lower() == "modespeeds":
            x = np.cumsum([0] + [val for val in self.timePeriods().values()])
            y = pd.concat([self.getModeSpeeds(0).stack()] + [self.getModeSpeeds(val).stack() for val in
                                                             self.timePeriods().keys()], axis=1)
            return x, y.transpose()
        elif type.lower() == "costs":
            operatorCosts, freightOperatorCosts, vectorUserCosts, externalities = self.collectAllCosts()
            x = list(self.microtypeIdToIdx.keys())
            userCostsByMicrotype = self.userCostDataFrame(vectorUserCosts).stack().stack().stack().unstack(
                level='homeMicrotype').sum(axis=0)
            operatorCostsByMicrotype = operatorCosts.toDataFrame().sum(axis=1)
            laneDedicationCostsByMicrotype = self.getDedicationCostByMicrotype()
            externalityCostByMicrotype = pd.Series(sum([e.sum(axis=1) for e in externalities.values()]),
                                                   index=self.microtypeIdToIdx.keys())
            y = pd.concat({'User': userCostsByMicrotype, 'Operator': operatorCostsByMicrotype,
                           'Lane Dedication': laneDedicationCostsByMicrotype['cost']}).unstack()
            return x, y
        else:
            print('WTF')
        print('AA')

    def diTuples(self):
        return [(di.homeMicrotype, di.populationGroupType, di.tripPurpose) for di in self.diToIdx.keys()]

    def odiTuples(self):
        return [(odi.o, odi.d, odi.distBin) for odi in self.odiToIdx.keys()]

    def userCostDataFrame(self, userCostMatrixByTimePeriod):
        dfByTimePeriod = {}
        for timePeriodID, timePeriodName in self.timePeriodNames().items():
            dfs = {}
            userCostMatrix = userCostMatrixByTimePeriod[timePeriodID]
            for mode, idx in self.modeToIdx.items():
                data = userCostMatrix[:, :, idx]
                df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(self.diTuples(),
                                                                        names=['homeMicrotype', 'populationGroupType',
                                                                               'tripPurpose']),
                                  columns=pd.MultiIndex.from_tuples(self.odiTuples(),
                                                                    names=['originMicrotype', 'destinationMicrotype',
                                                                           'distanceBin']))
                dfs[mode] = df
            dfByTimePeriod[(timePeriodID, timePeriodName)] = pd.concat(dfs)
        return pd.concat(dfByTimePeriod)

    def getDedicationCostByMicrotype(self):
        updatedNetworkData = self.scenarioData["subNetworkData"].merge(
            self.scenarioData["subNetworkDataFull"][['Dedicated', 'ModesAllowed', 'MicrotypeID']], left_index=True,
            right_index=True)
        dedicationCosts = self.scenarioData["laneDedicationCost"]
        dedicationCostsByMicrotype = pd.DataFrame(0, index=list(self.microtypeIdToIdx.keys()), columns=['cost'])
        for row in updatedNetworkData.itertuples():
            if row.Dedicated:
                if (row.MicrotypeID, row.ModesAllowed) in dedicationCosts.index:
                    dedicationCostsByMicrotype.loc[row.MicrotypeID, 'cost'] += row.Length * dedicationCosts.loc[
                        (row.MicrotypeID, row.ModesAllowed), 'CostPerMeter']
        return dedicationCostsByMicrotype

    def updateUtilityParam(self, value: float, param: str, populationGroupTypeID=None, tripPurposeID=None, mode=None):
        self.__population.setUtilityParam(value, param, populationGroupTypeID, tripPurposeID, mode)

    def getUtilityParam(self, param: str, populationGroupTypeID: str, tripPurposeID: str, mode: str, mID=None):
        return self.__population.getUtilityParam(param, populationGroupTypeID, tripPurposeID, mode, mID)

    def emptyOptimizer(self):
        return Optimizer(model=self)


class Optimizer:
    """
    Wrapper for the Model opject that allows model inputs to be optimized over.

    Attributes
    ----------
    model : Model
        model object
    fromToSubNetworkIDs : dict | None
        1:1 mapping of subnetworks between which ROW can be reassigned, e.g. mixed traffic -> bus only
    modesAndMicrotypes : dict | None
        List of tuples of mode/microtype pairs for which we will optimize headways
        e.g. [('A', 'bus'), ('B','rail')]
    method : str
        Optimization method

    Methods
    ---------
    evaluate(reallocations):
        Evaluate the objective function given a set of modifications to the transportation system
    minimize():
        Minimize the objective function using the set method
    """

    def __init__(self, model: Model, fromToSubNetworkIDs=None, modesAndMicrotypes=None, method="shgo"):
        self.__fromToSubNetworkIDs = fromToSubNetworkIDs
        self.__modesAndMicrotypes = modesAndMicrotypes
        self.__method = method
        self.__alphas = {"User": np.ones(len(model.microtypeIdToIdx)) * 20.,
                         "Operator": np.ones(len(model.microtypeIdToIdx)),
                         "Externality": np.ones(len(model.microtypeIdToIdx)),
                         "Dedication": np.ones(len(model.microtypeIdToIdx))}
        self.__trialParams = []
        self.__objectiveFunctionValues = []
        self.__isImprovement = []
        self.model = model

    def updateAlpha(self, costType, newValue, mID=None):
        if mID is None:
            if costType in self.__alphas:
                self.__alphas[costType][:] = newValue
            else:
                print("BAD INPUT ALPHA")
        else:
            if costType in self.__alphas:
                self.__alphas[costType][self.model.microtypeIdToIdx[mID]] = newValue
            else:
                print("BAD INPUT ALPHA")

    def nSubNetworks(self):
        if self.__fromToSubNetworkIDs is not None:
            return len(self.__fromToSubNetworkIDs)
        else:
            return 0

    def nModes(self):
        if self.__modesAndMicrotypes is not None:
            return len(self.__modesAndMicrotypes)
        else:
            return 0

    def toSubNetworkIDs(self):  # RENAME TO Microtypes
        return [toID for fromID, toID in self.__fromToSubNetworkIDs]

    def fromSubNetworkIDs(self):  # Rename to modes
        return [fromID for fromID, toID in self.__fromToSubNetworkIDs]

    def getDedicationCost(self, reallocations: np.ndarray) -> float:
        if self.nSubNetworks() > 0:
            microtypes = self.fromSubNetworkIDs()  # self.model.scenarioData["subNetworkData"].loc[self.toSubNetworkIDs(), "MicrotypeID"]
            modes = self.toSubNetworkIDs()
            # self.model.scenarioData["modeToSubNetworkData"].loc[
            # self.model.scenarioData["modeToSubNetworkData"]["SubnetworkID"].isin(
            #     self.toSubNetworkIDs()), "Mode"]
            perMeterCosts = self.model.scenarioData["laneDedicationCost"].loc[
                pd.MultiIndex.from_arrays([microtypes, modes]), "CostPerMeter"].values
            cost = np.sum(reallocations[:self.nSubNetworks()] * perMeterCosts)  # TODO: Convert back to real numbers
            if np.isnan(cost):
                return np.inf
            else:
                return cost
        else:
            return 0.0

    def updateAndRunModel(self, reallocations=None):
        if reallocations is not None:
            if self.__fromToSubNetworkIDs is not None:
                networkModification = NetworkModification(reallocations[:self.nSubNetworks()],
                                                          self.__fromToSubNetworkIDs)
            else:
                networkModification = None
            if self.__modesAndMicrotypes is not None:
                transitModification = TransitScheduleModification(reallocations[-self.nModes():],
                                                                  self.__modesAndMicrotypes)
            else:
                transitModification = None
            if self.model.choice.broken | (not self.model.successful):
                print("Starting from a bad place so I'll reset")
                self.model.initializeAllTimePeriods(True)
            self.model.modifyNetworks(networkModification, transitModification)
        self.model.collectAllCharacteristics()

    def scaling(self):
        return np.array([1.0] * self.nSubNetworks() + [0.001] * self.nModes())

    def sumAllCosts(self):
        operatorCosts, freightOperatorCosts, vectorUserCosts, externalities = self.model.collectAllCosts()
        if self.model.choice.broken | (not self.model.successful):
            return np.nan
        operatorCostsByMicrotype = operatorCosts.toDataFrame()['Cost'].unstack().sum(axis=1)
        operatorRevenuesByMicrotype = operatorCosts.toDataFrame()['Revenue'].unstack().sum(axis=1)
        freightCostsByMicrotype = freightOperatorCosts.toDataFrame()['Cost'].unstack().sum(axis=1)
        if len(freightCostsByMicrotype) == 0:
            freightCostsByMicrotype = operatorCostsByMicrotype * 0.0
        userCostsByMicrotype = self.model.userCostDataFrame(vectorUserCosts).stack().stack().stack().unstack(
            level='homeMicrotype').sum(axis=0)
        externalityCostsByMicrotype = pd.Series(sum([e.sum(axis=1) for e in externalities.values()]),
                                                index=sorted(self.model.microtypeIdToIdx))
        dedication = self.model.scenarioData['subNetworkDataFull'].loc[
            self.model.scenarioData['subNetworkDataFull'].Dedicated & (
                    self.model.scenarioData['subNetworkDataFull'].Type == "Road"), ["ModesAllowed", "MicrotypeID"]]
        dedication['Distance'] = self.model.scenarioData['subNetworkData'].loc[
            self.model.scenarioData['subNetworkDataFull'].Dedicated & (
                    self.model.scenarioData['subNetworkDataFull'].Type == "Road"), "Length"]
        dedicationCostsByMicrotype = pd.Series(0.0, index=sorted(self.model.microtypeIdToIdx))
        for _, val in dedication.iterrows():
            costPerMeter = self.model.scenarioData['laneDedicationCost']['CostPerMeter'].get(
                (val.MicrotypeID, val.ModesAllowed.lower()), 0.0)
            dedicationCostsByMicrotype[val.MicrotypeID] += costPerMeter * val.Distance
        output = dict()
        # {"User":1.0, "Operator":1.0, "Externality":1.0, "Dedication":1.0}
        output['User'] = userCostsByMicrotype * self.__alphas['User']
        output['Freight'] = freightCostsByMicrotype * self.__alphas['User']
        output['Operator'] = operatorCostsByMicrotype * self.__alphas['Operator']
        output['Revenue'] = - operatorRevenuesByMicrotype * self.__alphas['Operator']
        output['Externality'] = externalityCostsByMicrotype * self.__alphas['Externality']
        output['Dedication'] = dedicationCostsByMicrotype * self.__alphas['Dedication']
        allCosts = pd.concat(output, axis=1)
        allCosts.index.set_names(['Microtype'], inplace=True)
        return allCosts

    def evaluate(self, reallocations: np.ndarray) -> float:
        if np.any(np.isnan(reallocations)):
            print('SOMETHING WENT WRONG')
        scaling = self.scaling()
        if np.any([np.allclose(reallocations, trial) for trial in self.__trialParams]):
            outcome = self.__objectiveFunctionValues[
                [ind for ind, val in enumerate(self.__trialParams) if np.allclose(val, reallocations)][0]]
            print([str(reallocations), outcome])
            return outcome
        self.updateAndRunModel(reallocations / scaling)
        # operatorCosts, vectorUserCosts, externalities = self.model.collectAllCosts()
        if self.model.choice.broken | (not self.model.successful):
            print('SKIPPING!')
            print(reallocations)
            return 1e10
        # dedicationCosts = self.getDedicationCost(reallocations / scaling)
        # print(reallocations / scaling, self.model.sumAllCosts(operatorCosts, vectorUserCosts, externalities) + dedicationCosts)
        # print({a: b.sum() for a, b in vectorUserCosts.items()})
        # print(operatorCosts)
        # print({a: b.sum() for a, b in externalities.items()})
        allCosts = self.sumAllCosts()
        outcome = allCosts.to_numpy().sum()
        if self.__objectiveFunctionValues:
            self.__isImprovement.append(outcome < min(self.__objectiveFunctionValues))
        else:
            self.__isImprovement.append(True)
        self.__trialParams.append(reallocations)
        self.__objectiveFunctionValues.append(outcome)
        if np.isnan(outcome):
            self.model.interact.hardReset()
        print([str(reallocations), outcome])
        return outcome

    def getBounds(self):
        if self.__fromToSubNetworkIDs is not None:
            upperBoundsROW = [0.6] * len(self.fromSubNetworkIDs())
            lowerBoundsROW = [0.0] * len(self.fromSubNetworkIDs())
        else:
            upperBoundsROW = []
            lowerBoundsROW = []
        upperBoundsHeadway = [3.600] * self.nModes()
        lowerBoundsHeadway = [0.06] * self.nModes()
        defaultAllocation = [0.01] * len(self.fromSubNetworkIDs())
        defaultHeadway = [0.300] * self.nModes()
        bounds = list(zip(lowerBoundsROW + lowerBoundsHeadway, upperBoundsROW + upperBoundsHeadway,
                          defaultAllocation + defaultHeadway))
        if self.__method == "shgo":
            return list(zip(lowerBoundsROW + lowerBoundsHeadway, upperBoundsROW + upperBoundsHeadway))
        elif self.__method == "sklearn":
            return list(zip(lowerBoundsROW + lowerBoundsHeadway, upperBoundsROW + upperBoundsHeadway))
        elif (self.__method == "noisy") | (self.__method == "SPSA"):
            return bounds
        else:
            return Bounds(lowerBoundsROW + lowerBoundsHeadway, upperBoundsROW + upperBoundsHeadway)

    def x0(self) -> np.ndarray:
        network = [0.0] * self.nSubNetworks()
        headways = [300.0] * self.nModes()
        return np.array(network + headways) * self.scaling()

    def minimize(self, x0=None):
        self.__objectiveFunctionValues = []
        self.__trialParams = []
        self.__isImprovement = []
        if x0 is None:
            x0 = self.x0()
        if self.__method == "shgo":
            return shgo(self.evaluate, self.getBounds(), sampling_method="simplicial",
                        options={'disp': True, 'maxiter': 100})
        # elif self.__method == "sklearn":
        #     b = self.getBounds()
        #     return gp_minimize(self.evaluate, self.getBounds(), n_calls=1000, verbose=True)
        elif self.__method == "noisy":
            # scaling = [1.0] * self.nSubNetworks() + [1000.0] * self.nModes()
            return minimizeCompass(self.evaluate, x0, bounds=self.getBounds(), paired=False, deltainit=1.0,
                                   errorcontrol=False, disp=True, deltatol=1e-4)
        elif self.__method == "SPSA":
            # scaling = [1.0] * self.nSubNetworks() + [1000.0] * self.nModes()
            return minimizeSPSA(self.evaluate, x0, bounds=self.getBounds(), paired=False, disp=True, a=0.02, c=0.02)

        else:
            # return minimize(self.evaluate, self.x0(), bounds=self.getBounds(), options={'eps':1e-1})
            # return dual_annealing(self.evaluate, self.getBounds(), no_local_search=False, initial_temp=150.)
            # return minimize(self.evaluate, x0, method='L-BFGS-B', bounds=self.getBounds(),
            #                 options={'eps': 0.002, 'iprint': 1})
            return minimize(self.evaluate, x0, method='TNC', bounds=self.getBounds(),
                            options={'eps': 0.0005, 'eta': 0.025, 'disp': True, 'maxiter': 500, 'maxfev': 2000})
            # options={'initial_tr_radius': 0.6, 'finite_diff_rel_step': 0.002, 'maxiter': 2000,
            #          'xtol': 0.002, 'barrier_tol': 0.002, 'verbose': 3})

    def plotConvergence(self):
        params = np.array(self.__trialParams)
        outcomes = np.array(self.__objectiveFunctionValues)
        mask = np.array(self.__isImprovement)
        return params, outcomes, mask


def startBar():
    modelInput = widgets.Dropdown(
        options=['One microtype toy model', '4 microtype toy model', 'Los Angeles (National params)',
                 'California A', 'California B', 'California C', 'California D', 'California E', 'California F',
                 'Geotype A', 'Geotype B', 'Geotype C', 'Geotype D', 'Geotype E', 'Geotype F'],
        value='Los Angeles (National params)',
        description='Input data:',
        disabled=False,
    )
    lookup = {'One microtype toy model': 'input-data-simpler',
              '4 microtype toy model': 'input-data',
              'Los Angeles (National params)': 'input-data-losangeles-national-params',
              'California A': 'input-data-california-A',
              'California B': 'input-data-california-B',
              'California C': 'input-data-california-C',
              'California D': 'input-data-california-D',
              'California E': 'input-data-california-E',
              'California F': 'input-data-california-F',
              'Geotype A': 'input-data-geotype-A',
              'Geotype B': 'input-data-geotype-B',
              'Geotype C': 'input-data-geotype-C',
              'Geotype D': 'input-data-geotype-D',
              'Geotype E': 'input-data-geotype-E',
              'Geotype F': 'input-data-geotype-F'}
    return modelInput, lookup


if __name__ == "__main__":
    model = Model("input-data", 1, False)
    optimizer = Optimizer(model, modesAndMicrotypes=None,
                          fromToSubNetworkIDs=[('A', 'Bike')], method="opt")
    # optimizer.evaluate([0.1])
    #
    # allCosts = optimizer.sumAllCosts()
    # #
    # # print(model.getModeSpeeds())
    # model.collectAllCharacteristics()

    # print(model.getModeSpeeds())
    obj = Mock()
    obj.new = 0.5

    model.interact.modifyModel(('networkLength', 'A'), obj)
    optimizer.updateAndRunModel()
    allCosts = optimizer.sumAllCosts()
    # model.collectAllCharacteristics()
    # print(model.getModeSpeeds())
    # display(model.interact.grid)
    # operatorCosts, vectorUserCosts, externalities = model.collectAllCosts()
    # a, b = model.collectAllCharacteristics()
    # a, b = model.collectAllCharacteristics()
    # optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'bus'), ('B', 'bus')],
    #                       fromToSubNetworkIDs=[('A', 'Bus'), ('A', 'Bike'), ('B', 'Bus'), ('B', 'Bike')],
    #                       method="min")

    optimizer = Optimizer(model, modesAndMicrotypes=[('1', 'Bus')],
                          fromToSubNetworkIDs=[('1', 'Bus'), ('1', 'Bike')],
                          method="min")

    optimizer.evaluate(optimizer.x0())
    optimizer.evaluate([0.05, 0.05, 0.05])
    # model.data.updateMicrotypeNetworkLength('1', 0.75)
    # model.data.updateMicrotypeNetworkLength('2', 0.75)
    # optimizer.updateAlpha("Operator", 5.0)
    # optimizer.updateAlpha("Externality", 5.0)
    # optimizer.minimize()
    print('-----0.0------')

    # optimizer.evaluate([0.0])

    # model.collectAllCharacteristics()
    # x, y = model.plotAllDynamicStats('n')
    # # model.interact.updatePlots()
    # print('-----0.15------')
    # optimizer.evaluate([0.15])
    # print('-----0.0------')
    # optimizer.evaluate([0.0])
    # print('-----0.0------')
    # optimizer.evaluate([0.0])
    # # print('-----0.0------')
    # # optimizer.evaluate([0.0])
    # print('done')

    # # model.collectAllCharacteristics()
    # # userCostDf = model.userCostDataFrame(vectorUserCosts)
    # outcome = optimizer.minimize()
    # params, outcomes, mask = optimizer.plotConvergence()
    # print(outcome)
