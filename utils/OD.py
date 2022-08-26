import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from timebudget import timebudget

# from utils.microtype import Microtype
from .choiceCharacteristics import ChoiceCharacteristics

warnings.filterwarnings("ignore")


class ModeCharacteristics:
    def __init__(self, modes: List[str]):
        self._modes = modes
        self.characteristics = dict()
        for mode in modes:
            self.characteristics[mode] = ChoiceCharacteristics()

    def __getitem__(self, item):
        if item in self._modes:
            return self.characteristics[item]

    def __setitem__(self, key, value):
        self.characteristics[key] = value

    def keys(self) -> List:
        return list(self._modes)


class DemandIndex:
    def __init__(self, homeMicrotypeID, populationGroupTypeID, tripPurposeID):
        self.homeMicrotype = homeMicrotypeID
        self.populationGroupType = populationGroupTypeID
        self.tripPurpose = tripPurposeID
        self.__hash = hash((self.homeMicrotype.lower(), self.populationGroupType.lower(), self.tripPurpose.lower()))

    def __eq__(self, other):
        if (self.homeMicrotype.lower() == other.homeMicrotype.lower()) & (
                self.populationGroupType.lower() == other.populationGroupType.lower()) & (
                self.tripPurpose.lower() == other.tripPurpose.lower()):
            return True
        else:
            return False

    def __hash__(self):
        return self.__hash

    def isSenior(self):
        return "senior" in self.populationGroupType.lower()

    def __str__(self):
        return "Home: " + self.homeMicrotype + ", type: " + self.populationGroupType + ", purpose: " + self.tripPurpose

    def toTupleWith(self, other):
        return (self.homeMicrotype, self.populationGroupType, self.tripPurpose) + tuple([other])

    def toTuple(self):
        return self.homeMicrotype, self.populationGroupType, self.tripPurpose

    def toIndex(self):
        return pd.MultiIndex.from_tuples([(self.homeMicrotype, self.populationGroupType, self.tripPurpose)],
                                         names=('homeMicrotype', 'populationGroupType', 'tripPurpose'))


class ODindex:
    def __init__(self, o, d, distBin: int):
        if isinstance(o, str):
            self.o = o
        else:
            self.o = o.microtypeID
        if isinstance(d, str):
            self.d = d
        else:
            self.d = d.microtypeID
        self.distBin = distBin
        self.__hash = hash((self.o, self.d, self.distBin))

    def __eq__(self, other):
        if isinstance(other, ODindex):
            if (self.o == other.o) & (self.distBin == other.distBin) & (self.d == other.d):
                return True
            else:
                return False
        else:
            return False

    def __hash__(self):
        return self.__hash

    def __str__(self):
        return str(self.distBin) + " trip from " + self.o + " to " + self.d

    def toTuple(self):
        return self.o, self.d, self.distBin

    def toIndex(self):
        return pd.MultiIndex.from_tuples([(self.o, self.d, self.distBin)],
                                         names=('originMicrotype', 'destinationMicrotype', 'distanceBin'))


class TripGeneration:
    """
    Class to import and initialize trips from data.
    """

    def __init__(self):
        self.__data = pd.DataFrame()
        self.__tripClasses = dict()
        self.__currentTimePeriod = "BAD"

    @property
    def tripClasses(self):
        if self.__currentTimePeriod not in self.__tripClasses:
            self.__tripClasses[self.__currentTimePeriod] = dict()
        return self.__tripClasses[self.__currentTimePeriod]

    def setTimePeriod(self, timePeriod: int):
        self.__currentTimePeriod = timePeriod

    def __setitem__(self, key: (str, str), value: float):
        self.tripClasses[key] = value

    def __getitem__(self, item: (str, str)):
        return self.tripClasses.get(item, 0.0)

    def __contains__(self, item):
        return item in self.__tripClasses

    def importTripGeneration(self, df: pd.DataFrame):
        self.__data = df
        print("|  Loaded ", len(df), " trip generation rates")
        print("-------------------------------")

    def initializeTimePeriod(self, timePeriod, timePeriodID):
        # self.__tripClasses = dict()
        self.__currentTimePeriod = timePeriod
        if timePeriod not in self:
            relevantDemand = self.__data.loc[self.__data["TimePeriodID"] == timePeriodID]
            for row in relevantDemand.itertuples():
                self[row.PopulationGroupTypeID, row.TripPurposeID] = row.TripGenerationRatePerHour
            # print("|  Loaded ", len(relevantDemand), " demand classes")

    def __iter__(self):
        return iter(self.tripClasses.items())


class OriginDestination:
    """
    A class to import and store the origin and destination of trips.
    """

    def __init__(self, timePeriods, distanceBins, population, transitionMatrices, modelData, scenarioData):
        self.__ods = pd.DataFrame()
        self.__distances = pd.DataFrame()
        self.__originDestination = dict()
        self.__currentTimePeriod = "BAD"
        self.__timePeriods = timePeriods
        self.__distanceBins = distanceBins
        self.__population = population
        self.__transitionMatrices = transitionMatrices
        self.__modelData = modelData
        self.__scenarioData = scenarioData

    def setTimePeriod(self, timePeriod: int):
        self.__currentTimePeriod = timePeriod

    @property
    def originDestination(self):
        if self.__currentTimePeriod not in self.__originDestination:
            self.__originDestination[self.__currentTimePeriod] = dict()
        return self.__originDestination[self.__currentTimePeriod]

    @timebudget
    def importOriginDestination(self, ods: pd.DataFrame, distances: pd.DataFrame, modeAvailability: pd.DataFrame):
        self.__ods = ods
        self.__distances = distances
        print("|  Loaded ", len(ods), " ODs and ", len(distances), "unique distance bins")

        for timePeriodId, duration in self.__timePeriods:
            self.initializeTimePeriod(timePeriodId, self.__timePeriods.getTimePeriodName(timePeriodId))

        for originMicrotype in self.__scenarioData.microtypeIds:
            for destinationMicrotype in self.__scenarioData.microtypeIds:
                sub = modeAvailability.loc[(modeAvailability['OriginMicrotypeID'] == originMicrotype) & (
                        modeAvailability['DestinationMicrotypeID'] == destinationMicrotype), :]
                for distanceBin in self.__distances.DistanceBinID:
                    odi = ODindex(originMicrotype, destinationMicrotype, distanceBin)
                    if odi in self.__scenarioData.odiToIdx:
                        for row in sub.itertuples():
                            self.__modelData['toTransitLayer'][
                                self.__scenarioData.odiToIdx[odi], self.__scenarioData.transitLayerToIdx[
                                    row.TransitLayer]] = row.Portion

        for transitLayer, idx in self.__scenarioData.transitLayerToIdx.items():
            modeIncluded = np.array(
                [('no' + mode).lower() not in transitLayer for mode in self.__scenarioData.passengerModeToIdx.keys()])
            self.__modelData['transitLayerUtility'][~modeIncluded, idx] -= 1e6

        # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## #
        # Maybe use this to change availability layers based on mode params
        ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ##
        #
        # modeInTransitLayer = (self.__modelData['transitLayerUtility'] == 0).T
        #
        # fromLayers = modeInTransitLayer[~modeInTransitLayer[:, 3], :]
        # toLayers = modeInTransitLayer[modeInTransitLayer[:, 3], :]
        # toTransitLayer = self.__modelData['toTransitLayer']
        # currentlyAccessibleByOdi = toTransitLayer[:, modeInTransitLayer[:, 3]]
        # currentlyNotAccessibleByOdi = toTransitLayer[:, ~modeInTransitLayer[:, 3]]
        # odiToMode = self.__modelData['toTransitLayer'] @ (self.__modelData['transitLayerUtility'] == 0).T

        for odi, idx in self.__scenarioData.odiToIdx.items():
            self.__modelData['toStarts'][idx, self.__scenarioData.microtypeIdToIdx[odi.o]] = True
            self.__modelData['toEnds'][idx, self.__scenarioData.microtypeIdToIdx[odi.d]] = True
            self.__modelData['toDistanceByOrigin'][idx, self.__scenarioData.microtypeIdToIdx[
                odi.d]] = self.__distanceBins[odi.distBin]
            # TODO: Expand through distance to have a mode dimension, then filter and reallocate
            self.__modelData['toThroughDistance'][idx, :] = self.__transitionMatrices.assignmentMatrix(odi) * \
                                                            self.__distanceBins[odi.distBin]

    def __len__(self):
        return len(self.originDestination)

    def __setitem__(self, key: DemandIndex, value: dict):
        self.originDestination[key] = value

    def __getitem__(self, item: DemandIndex):
        if item not in self.originDestination:
            # print("OH NO, no origin destination defined for ", str(item), " in ", self.__currentTimePeriod)
            subitem = self.__distances.loc[(self.__distances["OriginMicrotypeID"] == item.homeMicrotype) & (
                    self.__distances["DestinationMicrotypeID"] == item.homeMicrotype) & (
                                                   self.__distances["TripPurposeID"] == item.tripPurpose)]
            out = dict()
            for row in subitem.itertuples():
                out[ODindex(row.OriginMicrotypeID, row.DestinationMicrotypeID, row.DistanceBinID)] = row.Portion
            self.originDestination[item] = out
            return out
        else:
            return self.originDestination[item]

    def __contains__(self, item):
        return item in self.originDestination

    def initializeTimePeriod(self, timePeriod, timePeriodID):
        self.__currentTimePeriod = timePeriod
        if timePeriod not in self.__originDestination:
            # print("|  Loaded ", len(self.__ods.loc[self.__ods["TimePeriodID"] == timePeriodID]), " distance bins")
            relevantODs = self.__ods.loc[self.__ods["TimePeriodID"] == timePeriodID]
            merged = relevantODs.merge(self.__distances,
                                       on=["TripPurposeID", "OriginMicrotypeID", "DestinationMicrotypeID"],
                                       suffixes=("_OD", "_Dist"),
                                       how="inner")
            for tripClass, grouped in merged.groupby(["HomeMicrotypeID", "PopulationGroupTypeID", "TripPurposeID"]):
                grouped["tot"] = grouped["Portion_OD"] * grouped["Portion_Dist"]
                tot = np.sum(grouped["tot"])
                grouped["tot"] = grouped["tot"] / tot
                # if abs(tot - 1) > 0.1:  # TODO: FIX
                #     print(f"Oops, totals for {tripClass} add up to {tot}")
                distribution = dict()
                for row in grouped.itertuples():
                    distribution[
                        ODindex(row.OriginMicrotypeID, row.DestinationMicrotypeID, row.DistanceBinID)] = row.tot
                self[DemandIndex(*tripClass)] = distribution
        # for row in relevantDemand.itertuples():
        #     self[row.PopulationGroupTypeID, row.TripPurposeID] = row.TripGenerationRatePerHour


class TransitionMatrix:
    def __init__(self, microtypeIdToIdx, matrix=None, diameters=None):
        self.__microtypeIds = list(microtypeIdToIdx.keys())
        self.__microtypeIdToIdx = microtypeIdToIdx
        self.__averageSpeeds = np.zeros(len(microtypeIdToIdx))
        if isinstance(matrix, pd.DataFrame):
            self.__matrix = pd.DataFrame(0.0, index=self.__microtypeIds, columns=self.__microtypeIds).add(
                matrix, fill_value=0.0)
        elif matrix is None:
            self.__matrix = pd.DataFrame(0.0, index=self.__microtypeIds, columns=self.__microtypeIds)
        elif isinstance(matrix, np.ndarray):
            self.__matrix = pd.DataFrame(matrix, index=self.__microtypeIds, columns=self.__microtypeIds)
        else:
            print("ERROR INITIALIZING TRANSITION MATRIX")
        if diameters is None:
            diameters = np.ones(len(self.__microtypeIds))
        self.__diameters = diameters

    @property
    def averageSpeeds(self) -> np.ndarray:
        return self.__averageSpeeds

    @property
    def diameters(self) -> np.ndarray:
        return self.__diameters

    def setAverageSpeeds(self, averageSpeeds: np.ndarray):
        self.__averageSpeeds = averageSpeeds

    @property
    def names(self) -> list:
        return self.__microtypeIds

    @property
    def matrix(self) -> pd.DataFrame:
        return self.__matrix

    def __getitem__(self, item):
        return {mId: self.__matrix.loc[idx, :].values for mId, idx in self.__microtypeIdToIdx}

    def __add__(self, other):
        if isinstance(other, TransitionMatrix):
            self.__matrix += other.__matrix
            return self  # TransitionMatrix(self.__names, self.matrix + other.matrix)
        else:
            print("ERROR ADDING TRANSITION MATRIX")
            return self

    def __radd__(self, other):
        if isinstance(other, TransitionMatrix):
            self.__matrix += other.__matrix
            return self  # TransitionMatrix(self.__names, self.matrix + other.matrix)
        else:
            print("ERROR ADDING TRANSITION MATRIX")
            return self

    def addAndMultiply(self, other, multiplier):
        self.__matrix += other.__matrix.to_numpy() * multiplier
        return self

    def __mul__(self, other):
        # self.__matrix *= other
        return TransitionMatrix(self.__microtypeIdToIdx, self.matrix * other)

    def idx(self, idx):
        return self.__microtypeIdToIdx[idx]

    def fillZeros(self):
        self.__matrix += 1. / (len(self.__microtypeIds) ** 2)
        return self

    def updateMatrix(self, other):
        self.__matrix = other.matrix

    def getSteadyState(self, tripStartRate) -> (float, np.ndarray):
        X = np.transpose(self.matrix.to_numpy())  # Q: Should this be transposed?
        X2 = np.hstack([X, (tripStartRate / tripStartRate.sum()).reshape(-1, 1)])
        X3 = np.vstack([X2, 1 - X2.sum(axis=0)])
        # val, vec = eigs(X, k=1, which='LM')
        # dists = self.diameters / (1 - np.real_if_close(val))
        dists3 = self.diameters / (X3[-1, :-1])
        val2, vec2 = eigs(X3, k=1, which='LM')
        # weights = np.real_if_close(vec / np.sum(vec))
        weights2 = np.real_if_close(vec2[:-1] / np.sum(vec2[:-1]))
        # dist = np.average(dists, weights=weights.reshape(len(dists), ))
        dist3 = np.average(dists3, weights=weights2.reshape(len(dists3), ))
        return dist3, np.squeeze(weights2)


class TransitionMatrices:
    def __init__(self, scenarioData, supplyData):
        self.__names = []
        self.__scenarioData = scenarioData
        self.__diameters = np.ndarray([])
        self.__data = dict()
        self.__transitionMatrices = dict()
        self.__currentTimePeriod = 0
        self.__numpy = supplyData['transitionMatrices']
        self.__assignmentMatrices = np.zeros((len(scenarioData.odiToIdx), len(scenarioData.microtypeIdToIdx)))
        self.__baseIdx = dict()
        self.__transitionMatrix = supplyData['transitionMatrix']

    def setTimePeriod(self, currentTimePeriod):
        self.__currentTimePeriod = currentTimePeriod

    @property
    def microtypeIdToIdx(self):
        return self.__scenarioData.microtypeIdToIdx

    @property
    def transitionMatrix(self):
        return self.__transitionMatrix[self.__currentTimePeriod, :, :]

    @property
    def odiToIdx(self):
        return self.__scenarioData.odiToIdx

    # @property
    # def idx(self):
    #     if self.__currentTimePeriod in self.__idx:
    #         return self.__idx[self.__currentTimePeriod]
    #     else:
    #         return dict()

    def assignmentMatrix(self, item: ODindex):
        return self.__assignmentMatrices[self.odiToIdx[item], :]

    # def __getitem__(self, item: ODindex):
    #     if (item.o, item.d, item.distBin) in self.__transitionMatrices:
    #         return self.__transitionMatrices[(item.o, item.d, item.distBin)]
    #     else:
    #         if (item.o, item.d, item.distBin) in self.__data:
    #             out = TransitionMatrix(self.microtypeIdToIdx, self.__data[(item.o, item.d, item.distBin)],
    #                                    diameters=self.__diameters)
    #             self.__transitionMatrices[(item.o, item.d, item.distBin)] = out
    #             return out
    #         else:
    #             # print(f"No transition matrix found for {(item.o, item.d, item.distBin)}")
    #             out = TransitionMatrix(self.microtypeIdToIdx).fillZeros()
    #             return out

    # def reIndex(self, odiToIdx, currentTimePeriod):
    #     newNumpy = self.__baseNumpy.copy()
    #     for odi, idx in odiToIdx.items():
    #         newNumpy[idx, :, :] = self.__baseNumpy[self.__baseIdx.get(idx, -1), :, :]
    #     self.__idx[currentTimePeriod] = odiToIdx
    #     self.__currentTimePeriod = currentTimePeriod
    #     self.__numpy[currentTimePeriod] = newNumpy

    # def adoptMicrotypes(self, microtypes: pd.DataFrame):
    #     self.__names = microtypes["MicrotypeID"].to_list()
    #     self.__diameters = microtypes["DiameterInMiles"].to_numpy()

    # def getIdx(self, item) -> int:
    #     return self.idx.get(item, -1)

    def emptyWeights(self) -> np.ndarray:
        return np.zeros(self.transitionMatrix.shape[0])

    def averageMatrix(self, weights: np.ndarray) -> np.ndarray:
        if np.sum(weights) > 0.0:
            return np.average(self.__numpy, axis=0, weights=weights)
        else:
            return np.zeros_like(self.transitionMatrix) + 1. / (len(self.microtypeIdToIdx) ** 2)

    @timebudget
    def importTransitionMatrices(self, matrices: pd.DataFrame, microtypeIDs: pd.DataFrame, distanceBins: pd.DataFrame):
        default = pd.DataFrame(0.0, index=microtypeIDs.MicrotypeID, columns=microtypeIDs.MicrotypeID)
        for key, val in matrices.groupby(level=[0, 1, 2]):
            mat = val.copy()  # .loc[:, val.columns.isin(self.microtypeIdToIdx.keys())]
            odi = ODindex(*key)
            if odi in self.odiToIdx:
                df = mat.set_index(mat.index.droplevel([0, 1, 2])).add(default, fill_value=0.0)
                df = df[df.index]
                # self.__data[odi] = df  # TODO: Delete this
                self.__numpy[self.odiToIdx[odi], :, :] = df.to_numpy()
                startVec = np.zeros((1, len(microtypeIDs)))
                startVec[0, self.microtypeIdToIdx[odi.o]] = 1
                X4 = np.vstack([df.values, startVec])
                X5 = np.hstack([X4, 1 - X4.sum(axis=1).reshape((-1, 1))])
                lam, vec = eigs(X5.transpose(), k=1, which='LM')
                if vec.shape[1] > 1:
                    # Something weird about eigs?
                    vec = vec[:, 0]
                tripDistribution = np.real_if_close(vec[:-1]) / np.real_if_close(vec[:-1]).sum()
                if np.all(np.isreal(lam)):
                    if np.abs(np.sum(tripDistribution) - 1.0) > 0.1:
                        print('Something went wrong in loading transition matrices')
                    self.__assignmentMatrices[self.odiToIdx[odi], :] = np.squeeze(tripDistribution)
                else:
                    print('Imaginary transition matrix')
                    self.__assignmentMatrices[self.odiToIdx[odi], :] = X4.sum(axis=0) / X4.sum()
                # realLengths = np.squeeze((np.eye(len(microtypeIDs)) - df.values).T @ np.ones_like(startVec.T))
                # meanSteps = np.linalg.inv(np.eye(len(microtypeIDs)) - df.values.transpose()) @ \
                #             np.ones((len(microtypeIDs), 1))
        goodAssignments = np.isclose(self.__assignmentMatrices.sum(axis=1), 1.0)
        meanDistribution = np.mean(self.__assignmentMatrices[goodAssignments, :], axis=0)
        self.__assignmentMatrices[~goodAssignments, :] = meanDistribution
        print("|  Loaded ", str(matrices.size), " transition probabilities")
        print("-------------------------------")
