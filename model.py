# from line_profiler_pycharm import profile
from sys import stdout

import ipywidgets as widgets
import numpy as np
import pandas as pd
# from noisyopt import minimizeCompass, minimizeSPSA
from scipy.optimize import root, minimize, Bounds, shgo
# from mock import Mock
# from skopt import gp_minimize, forest_minimize

from utils.OD import OriginDestination, TripGeneration, TransitionMatrices
from utils.choiceCharacteristics import CollectedChoiceCharacteristics
from utils.demand import Demand, Externalities, modeSplitFromUtilsWithExcludedModes, Accessibility
from utils.interact import Interact
from utils.microtype import MicrotypeCollection
from utils.misc import TimePeriods, DistanceBins
from utils.population import Population
from utils.data import Data, ScenarioData


class OptimizationDomain:
    def __init__(self, modifications: list):
        self.headways = []
        self.modesAndMicrotypes = []
        self.modificationToIdx = dict()
        self.__scalingDefaults = {"headway": 0.01, "dedicated": 1.0, "fare": 0.1, "fareSenior": 0.1, "coverage": 1.0,
                                  "fleetSize": 1.0}
        for idx, (changeType, changeDetails) in enumerate(modifications):
            if changeType == "dedicated":
                self.modesAndMicrotypes.append(changeDetails)
            elif changeType == "headway":
                self.headways.append(changeDetails)
            elif (changeType == "fare") | (changeType == "fareSenior"):
                pass
            elif (changeType == "coverage") & (changeDetails[1].lower() == "bus"):
                pass
            elif (changeType == "fleetSize") & (changeDetails[1].lower() == "bike"):
                pass
            else:
                raise NotImplementedError("Optimization value {0} not yet included".format(changeType))
            self.modificationToIdx[(changeType, changeDetails)] = idx

    @property
    def modifications(self):
        return list(self.modificationToIdx.keys())

    def getBounds(self):
        lowerBounds = []
        upperBounds = []
        defaults = []

        for (changeType, changeDetails) in self.modifications:
            if changeType == "dedicated":
                lowerBounds.append(0.0 * self.__scalingDefaults[changeType])
                upperBounds.append(0.6 * self.__scalingDefaults[changeType])
                defaults.append(0.0 * self.__scalingDefaults[changeType])
            elif changeType == "headway":
                lowerBounds.append(60 * self.__scalingDefaults[changeType])
                upperBounds.append(3600 * self.__scalingDefaults[changeType])
                defaults.append(300 * self.__scalingDefaults[changeType])
            elif (changeType == "fare") | (changeType == "fareSenior"):
                lowerBounds.append(0.0 * self.__scalingDefaults[changeType])
                upperBounds.append(20.0 * self.__scalingDefaults[changeType])
                defaults.append(2.0 * self.__scalingDefaults[changeType])
            elif (changeType == "coverage") & (changeDetails[1].lower() == "bus"):
                lowerBounds.append(0.001 * self.__scalingDefaults[changeType])
                upperBounds.append(1.0 * self.__scalingDefaults[changeType])
                defaults.append(0.2 * self.__scalingDefaults[changeType])
            elif (changeType == "fleetSize") & (changeDetails[1].lower() == "bike"):
                lowerBounds.append(0.001 * self.__scalingDefaults[changeType])
                upperBounds.append(1.2 * self.__scalingDefaults[changeType])
                defaults.append(0.5 * self.__scalingDefaults[changeType])
            else:
                raise NotImplementedError("Optimization value {0} not yet included".format(changeType))

        return lowerBounds, upperBounds, defaults

    def scaling(self):
        return np.array([self.__scalingDefaults[changeType] for (changeType, changeDetails) in self.modifications])

    def nSubNetworks(self):
        return len(self.modesAndMicrotypes)


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
        self.__timeStepInSeconds = 10.0
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
        self.__transitionMatrices = TransitionMatrices(self.scenarioData, self.data.getSupply())
        self.__originDestination = OriginDestination(self.__timePeriods, self.__distanceBins, self.__population,
                                                     self.__transitionMatrices, self.__fixedData,
                                                     self.scenarioData)
        self.__externalities = Externalities(self.scenarioData)
        self.__accessibility = Accessibility(self.scenarioData, self.data)
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
                self.__currentTimePeriod), self.data.getDemand(self.__currentTimePeriod))
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

    def clearCostCache(self, type=None):
        for cc in self.__choice.values():
            cc.clearCache(type)

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
        self.__accessibility.init()
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
        utilitiesArray = self.demand.calculateUtilities(choiceCharacteristicsArray, True)
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
        choiceCharacteristicsArray = self.choice.updateChoiceCharacteristics(self.microtypes)
        utilities = self.demand.calculateUtilities(choiceCharacteristicsArray, False)
        self.demand.utilities = utilities

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
        self.__transitionMatrices.setTimePeriod(timePeriod)

    def collectAllCosts(self, event=None):
        operatorCosts = np.zeros((len(self.microtypeIdToIdx), len(self.modeToIdx)))
        externalities = dict()
        vectorUserCosts = dict()
        policyRevenues = np.zeros((len(self.microtypeIdToIdx), len(self.modeToIdx)))
        for timePeriod, durationInHours in self.__timePeriods:
            self.setTimePeriod(timePeriod, preserve=True)
            matCosts = self.getMatrixUserCosts() * durationInHours
            vectorUserCosts[timePeriod] = matCosts
            operatorCosts += self.getOperatorCosts() * durationInHours + self.getFreightOperatorCosts() * durationInHours
            externalities[timePeriod] = self.__externalities.calcuate(self.microtypes) * durationInHours
            policyRevenues[:, :len(self.passengerModeToIdx)] += self.demand.getMatrixPolicyRevenues() * durationInHours
        accessibility = self.calculateAccessibility()
        return operatorCosts, policyRevenues, vectorUserCosts, externalities, accessibility

    def updatePopulation(self):
        for timePeriod, durationInHours in self.__timePeriods:
            self.setTimePeriod(timePeriod, preserve=True)
            self.demand.updateTripGeneration(self.microtypes)

    def calculateAccessibility(self, normalize=False):
        acc = self.__accessibility.calculateByDI()
        if normalize:
            for (row, val) in self.scenarioData['populations'].iterrows():
                acc.loc[pd.IndexSlice[val.MicrotypeID, val.PopulationGroupTypeID]] /= val.Population
        return acc

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

        continuousSpeed = pd.DataFrame(self.data.v.T, columns=self.microtypeIdToIdx.keys(), index=self.data.t)
        continuousAccumulation = pd.DataFrame(self.data.n.T, columns=self.microtypeIdToIdx.keys(), index=self.data.t)
        continuousData = pd.concat({"SpeedInMetersPerSecond": continuousSpeed, "Accumulation": continuousAccumulation},
                                   axis=1)

        return modeSplitData, speedData, utilityData, continuousData

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
        elif type.lower() == "production":
            t = self.data.t / 3600.
            actualProduction = np.cumsum(
                (self.data.n * self.data.v).sum(axis=0)) * self.__timeStepInSeconds / 1609.34  # ** 2
            demandedProduction = self.data.getDemand()['demandData'][:, :, self.modeToIdx['auto'],
                                 self.scenarioData.demandDataTypeToIdx['vehicleDistance']].sum(axis=1)
            demandedProduction = np.cumsum(np.hstack([[0], demandedProduction])) / self.data.params.nSubBins * \
                                 self.scenarioData['timePeriods']['DurationInHours'].mean()
            t2 = np.linspace(0, self.scenarioData['timePeriods']['DurationInHours'].sum(),
                             self.data.params.nTimePeriods + 1)
            interpDemandedProduction = np.interp(t, t2, demandedProduction)
            y = np.vstack([actualProduction, interpDemandedProduction])
            # LOOKS LIKE WHEN WE DO IT THIS WAY the actual (MFD) production is 30x higher than it should be, doesn't depend on number of time bins or time step. THIS IS FOR 200m steps, try 400 next
            return t, y
        elif type.lower() == "modespeeds":
            x = np.cumsum([0] + [val for val in self.timePeriods().values()])
            y = pd.concat([self.getModeSpeeds(0).stack()] + [self.getModeSpeeds(val).stack() for val in
                                                             self.timePeriods().keys()], axis=1)
            return x, y.transpose()
        elif type.lower() == "costs":
            operatorCosts, policyRevenues, vectorUserCosts, externalities, accessibility = self.collectAllCosts()
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
            for mode, idx in self.passengerModeToIdx.items():
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
        return Optimizer(model=self, domain=OptimizationDomain([]))


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

    def __init__(self, model: Model, domain: OptimizationDomain, method="shgo"):
        self.__domain = domain
        self.__method = method
        self.__alphas = {"User": np.ones(len(model.microtypeIdToIdx)) * 20.,
                         "Operator": np.ones(len(model.microtypeIdToIdx)),
                         "Externality": np.ones(len(model.microtypeIdToIdx)),
                         "Dedication": np.ones(len(model.microtypeIdToIdx)),
                         "Accessibility": np.ones(len(model.microtypeIdToIdx))}
        self.__accessibilityMultipliers = pd.Series(0.0, index=pd.MultiIndex.from_tuples(
            [(odi.homeMicrotype, odi.populationGroupType, odi.tripPurpose) for odi in model.diToIdx.keys()],
            names=["Microtype ID", "Population Group", "Trip Purpose"]))
        self.__trialParams = []
        self.__objectiveFunctionValues = []
        self.__isImprovement = []
        self.__initializeAccessibilityMultipliers()
        self.model = model

    def __initializeAccessibilityMultipliers(self):
        relevantTripTypes = set(self.__accessibilityMultipliers.index.levels[-1]).difference({'home', 'work'})
        self.__accessibilityMultipliers.loc[pd.IndexSlice[:, :, list(relevantTripTypes)]] = 1.0

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

    def updateAndRunModel(self, xUnScaled=None):
        if xUnScaled is not None:
            if self.model.choice.broken | (not self.model.successful):
                print("Starting from a bad place so I'll reset")
                self.model.initializeAllTimePeriods(True)
            for mod, val in zip(self.__domain.modifications, xUnScaled):
                self.model.interact.modifyModel(changeType=mod, value=val)
        self.model.collectAllCharacteristics()

    def costDataFrame(self, vals) -> pd.DataFrame:
        return pd.DataFrame(vals, index=self.model.microtypeIdToIdx.keys(), columns=self.model.modeToIdx.keys())

    def sumAllCosts(self):
        operatorCosts, policyRevenues, vectorUserCosts, externalities, accessibility = self.model.collectAllCosts()
        # if self.model.choice.broken | (not self.model.successful):
        #     return np.nan
        operatorCostsByMicrotype = self.costDataFrame(operatorCosts)
        operatorRevenuesByMicrotype = self.costDataFrame(policyRevenues)

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
            if val.MicrotypeID in self.model.microtypeIdToIdx:
                costPerMeter = self.model.scenarioData['laneDedicationCost']['CostPerMeter'].get(
                    (val.MicrotypeID, val.ModesAllowed.lower()), 0.0)
                dedicationCostsByMicrotype[val.MicrotypeID] += costPerMeter * val.Distance
        am = self.__accessibilityMultipliers.unstack()

        accessibilityByMicrotype = (accessibility.loc[am.index, am.columns] * am).unstack().sum(axis=1)
        output = dict()
        # {"User":1.0, "Operator":1.0, "Externality":1.0, "Dedication":1.0}
        output['User'] = userCostsByMicrotype * self.__alphas['User']
        output['Freight'] = operatorCostsByMicrotype.iloc[:, len(self.model.passengerModeToIdx):].sum(axis=1) * \
                            self.__alphas['Operator']
        output['Operator'] = operatorCostsByMicrotype.iloc[:, :len(self.model.passengerModeToIdx)].sum(axis=1) * \
                             self.__alphas['Operator']
        output['Revenue'] = - operatorRevenuesByMicrotype.sum(axis=1) * self.__alphas['Operator']
        output['Externality'] = externalityCostsByMicrotype * self.__alphas['Externality']
        output['Dedication'] = dedicationCostsByMicrotype * self.__alphas['Dedication']
        output['Accessibility'] = accessibilityByMicrotype * self.__alphas['Accessibility']
        allCosts = pd.concat(output, axis=1)
        allCosts.index.set_names(['Microtype'], inplace=True)
        return allCosts

    def evaluate(self, reallocations: np.ndarray) -> float:
        if np.any(np.isnan(reallocations)):
            print('SOMETHING WENT WRONG')
        scaling = self.__domain.scaling()
        if np.any([np.allclose(reallocations, trial) for trial in self.__trialParams]):
            outcome = self.__objectiveFunctionValues[
                [ind for ind, val in enumerate(self.__trialParams) if np.allclose(val, reallocations)][0]]
            print([str(reallocations), outcome])
            return outcome
        self.updateAndRunModel(reallocations / scaling)
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

    def x0(self) -> np.ndarray:
        lb, ub, x0 = self.__domain.getBounds()
        return x0

    def minimize(self, x0=None):
        self.__objectiveFunctionValues = []
        self.__trialParams = []
        self.__isImprovement = []
        lowerBounds, upperBounds, maybeX = self.__domain.getBounds()
        if x0 is None:
            x0 = maybeX
        if self.__method == "shgo":
            bounds = list(zip(lowerBounds, upperBounds))
            return shgo(self.evaluate, bounds, sampling_method="simplicial",
                        options={'disp': True, 'maxiter': 100})
        # elif self.__method == "sklearn":
        #     b = self.getBounds()
        #     return gp_minimize(self.evaluate, self.getBounds(), n_calls=1000, verbose=True)
        elif self.__method == "noisy":
            bounds = list(zip(lowerBounds, upperBounds, x0))
            return minimizeCompass(self.evaluate, x0, bounds=bounds, paired=False, deltainit=1.0,
                                   errorcontrol=False, disp=True, deltatol=1e-4)
        elif self.__method == "SPSA":
            bounds = list(zip(lowerBounds, upperBounds, x0))
            return minimizeSPSA(self.evaluate, x0, bounds=bounds, paired=False, disp=True, a=0.02, c=0.02)

        else:
            # return minimize(self.evaluate, self.x0(), bounds=self.getBounds(), options={'eps':1e-1})
            # return dual_annealing(self.evaluate, self.getBounds(), no_local_search=False, initial_temp=150.)
            # return minimize(self.evaluate, x0, method='L-BFGS-B', bounds=self.getBounds(),
            #                 options={'eps': 0.002, 'iprint': 1})
            bounds = Bounds(lowerBounds, upperBounds)
            return minimize(self.evaluate, x0, method='TNC', bounds=bounds,
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
        options=['One microtype toy model', '4 microtype toy model',  # 'Los Angeles (National params)',
                 'California A', 'California B', 'California C', 'California D', 'California E', 'California F'],
        # 'Geotype A', 'Geotype B', 'Geotype C', 'Geotype D', 'Geotype E', 'Geotype F'],
        value='4 microtype toy model',
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
    model = Model("input-data", 1, True)
    optimizer = Optimizer(model, domain=OptimizationDomain(
        [('dedicated', ('A', 'Bus')),
         ('headway', ('A', 'Bus')),
         ('fare', ('A', 'Bus'))
         ]),
                          method="opt")
    # optimizer.updateAndRunModel(np.array([0.05, 250, 1.25]))
    # x, y = model.plotAllDynamicStats("production")
    model.interact.modifyModel(('maxInflowPerMeterPerHour', 1), 1.5)
    outcome = optimizer.minimize()
    print(outcome)
