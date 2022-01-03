from itertools import product

import numpy as np

np.set_printoptions(precision=5)
import pandas as pd

from .OD import OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit, TransitionMatrices
from .choiceCharacteristics import CollectedChoiceCharacteristics
from .microtype import MicrotypeCollection
from .misc import DistanceBins, TimePeriods
from .population import Population


class Externalities:
    def __init__(self, scenarioData):
        self.__scenarioData = scenarioData
        self.__numpyPerPassengerMile = np.ndarray(0)
        self.__numpyPerVehicleMile = np.ndarray(0)

    def init(self):
        self.__numpyPerPassengerMile = np.zeros(
            (len(self.__scenarioData.microtypeIdToIdx), len(self.__scenarioData.modeToIdx)))
        self.__numpyPerVehicleMile = np.zeros(
            (len(self.__scenarioData.microtypeIdToIdx), len(self.__scenarioData.modeToIdx)))
        df = self.__scenarioData["modeExternalities"]
        for mode, modeIdx in self.__scenarioData.modeToIdx.items():
            for mId, mIdx in self.__scenarioData.microtypeIdToIdx.items():
                if (mId, mode) in df.index:
                    self.__numpyPerPassengerMile[mIdx, modeIdx] = df['CostPerPassengerMile'].iloc[
                        df.index.get_loc((mId, mode))]
                    self.__numpyPerVehicleMile[mIdx, modeIdx] = df['CostPerVehicleMile'].iloc[
                        df.index.get_loc((mId, mode))]

    def calcuate(self, microtypes: MicrotypeCollection) -> np.ndarray:
        totalExternalities = self.__numpyPerVehicleMile * microtypes.vehicleDistanceByMode + \
                             self.__numpyPerPassengerMile * microtypes.passengerDistanceByMode
        return totalExternalities


class TotalUserCosts:
    def __init__(self, total=0., totalEqualVOT=0., totalIVT=0., totalOVT=0., demandForTripsPerHour=0.,
                 demandForPMTPerHour=0.):
        self.total = total
        self.totalEqualVOT = totalEqualVOT
        self.totalIVT = totalIVT
        self.totalOVT = totalOVT
        self.demandForTripsPerHour = demandForTripsPerHour
        self.demandForPMTPerHour = demandForPMTPerHour

    def __str__(self):
        return str(self.total) + ' ' + str(self.totalEqualVOT)

    def __mul__(self, other):
        result = TotalUserCosts()
        result.total = self.total * other
        result.totalEqualVOT = self.totalEqualVOT * other
        result.totalIVT = self.totalIVT * other
        result.totalOVT = self.totalOVT * other
        result.demandForTripsPerHour = self.demandForTripsPerHour * other
        result.demandForPMTPerHour = self.demandForPMTPerHour * other
        return result

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        self.total *= other
        self.totalEqualVOT *= other
        self.totalIVT *= other
        self.totalOVT *= other
        self.demandForTripsPerHour *= other
        self.demandForPMTPerHour *= other
        return self

    def copy(self):
        return TotalUserCosts(self.total, self.totalEqualVOT, self.totalIVT, self.totalOVT, self.demandForTripsPerHour,
                              self.demandForPMTPerHour)

    def __add__(self, other):
        out = self.copy()
        out.total += other.total
        out.totalEqualVOT += other.totalEqualVOT
        out.totalIVT += other.totalIVT
        out.totalOVT += other.totalOVT
        out.demandForTripsPerHour += other.demandForTripsPerHour
        out.demandForPMTPerHour += other.demandForPMTPerHour
        return out

    def toDataFrame(self, index=None):
        return pd.DataFrame({"totalCost": self.total, "demandForTripsPerHour": self.demandForTripsPerHour,
                             "inVehicleTime": self.totalIVT, "outOfVehicleTime": self.totalOVT,
                             "demandForPMTPerHour": self.demandForPMTPerHour}, index=index)


class CollectedTotalUserCosts:
    def __init__(self):
        self.__costsByPopulationAndMode = dict()
        self.total = 0.
        self.totalEqualVOT = 0.
        self.demandForTripsPerHour = 0.
        self.demandForPMTPerHour = 0.

    def __setitem__(self, key: (DemandIndex, str), value: TotalUserCosts):
        self.__costsByPopulationAndMode[key] = value
        # self.updateTotals(value)

    def __getitem__(self, item) -> TotalUserCosts:
        if isinstance(item, DemandIndex):
            print('BAD')
            return TotalUserCosts()
        elif isinstance(item, str):
            print('BAD')
            return TotalUserCosts()
        elif isinstance(item, tuple):
            return self.__costsByPopulationAndMode[item]
        else:
            print("BADDDDD")
            return TotalUserCosts()

    def __iter__(self):
        return iter(self.__costsByPopulationAndMode.items())

    def updateTotals(self):
        self.total = sum([c.total for c in self.__costsByPopulationAndMode.values()])
        self.totalEqualVOT = sum([c.totalEqualVOT for c in self.__costsByPopulationAndMode.values()])
        self.demandForTripsPerHour = sum([c.demandForTripsPerHour for c in self.__costsByPopulationAndMode.values()])
        self.demandForPMTPerHour = sum([c.demandForPMTPerHour for c in self.__costsByPopulationAndMode.values()])
        return self

    def copy(self):
        out = CollectedTotalUserCosts()
        out.total = self.total
        out.totalEqualVOT = self.totalEqualVOT
        out.demandForTripsPerHour = self.demandForTripsPerHour
        out.demandForPMTPerHour = self.demandForPMTPerHour
        out.__costsByPopulationAndMode = self.__costsByPopulationAndMode.copy()
        return out

    def __imul__(self, other):
        for item in self.__costsByPopulationAndMode.keys():
            self.__costsByPopulationAndMode[item] = self.__costsByPopulationAndMode[item] * other
        return self

    def __mul__(self, other):
        out = self.copy()
        for item in out.__costsByPopulationAndMode.keys():
            out.__costsByPopulationAndMode[item] = out.__costsByPopulationAndMode[item] * other
        return out

    def __rmul__(self, other):
        return self * other

    def __iadd__(self, other):
        for item, cost in other:
            self.__costsByPopulationAndMode[item] = self.__costsByPopulationAndMode.setdefault(item,
                                                                                               TotalUserCosts()) + cost
        return self

    def toDataFrame(self, index=None) -> pd.DataFrame:
        muc = pd.concat([val.toDataFrame(pd.MultiIndex.from_tuples([key[0].toTupleWith(key[1])])) for key, val in
                         self.__costsByPopulationAndMode.items()])
        muc.index.set_names(['homeMicrotype', 'populationGroupType', 'tripPurpose', 'mode'], inplace=True)
        return muc.swaplevel(0, -1)
        # return pd.concat([val.toDataFrame([key.toIndex()]) for key, val in self.__costs.items()])

    def groupBy(self, vals) -> pd.DataFrame:
        df = self.toDataFrame()
        return df.groupby(level=vals).agg(sum)


class Demand:
    def __init__(self, scenarioData, numpyData):
        self.__scenarioData = scenarioData
        self.__modes = list(scenarioData.modeToIdx.keys())
        self.__modeSplit = dict()
        self.__modeSplitData = numpyData['modeSplit']
        self.__currentUtility = numpyData['utilities']
        self.__tripRate = numpyData['tripRate']
        self.__toStarts = numpyData['toStarts']
        self.__toEnds = numpyData['toEnds']
        self.__toThroughDistance = numpyData['toThroughDistance']
        self.__seniorODIs = np.array([di.isSenior() for di in self.diToIdx.keys()])
        self.__shape = tuple
        self.tripRate = 0.0
        self.demandForPMT = 0.0
        self.pop = 0.0
        self.timePeriodDuration = 0.0
        self.__population = None
        self.__originDestination = None
        self.__tripGeneration = None
        # self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__transitionMatrices = None

    @property
    def toThroughDistance(self):
        return self.__toThroughDistance

    @property
    def odiToIdx(self):
        return self.__scenarioData.odiToIdx

    @property
    def diToIdx(self):
        return self.__scenarioData.diToIdx

    @property
    def modeToIdx(self):
        return self.__scenarioData.modeToIdx

    @property
    def microtypeIdToIdx(self):
        return self.__scenarioData.microtypeIdToIdx

    def modeSplitDataFrame(self):
        dis = pd.MultiIndex.from_tuples([di.toTuple() for di in self.diToIdx.keys()],
                                        names=('homeMicrotype', 'populationGroupType', 'tripPurpose'))
        odis = pd.MultiIndex.from_tuples([odi.toTuple() for odi in self.odiToIdx.keys()],
                                         names=('originMicrotype', 'destinationMicrotype', 'distanceBin'))
        modes = list(self.modeToIdx.keys())
        tuples = [(a, b, c, d, e, f, g) for (a, b, c), (d, e, f), g in product(dis, odis, modes)]
        mi = pd.MultiIndex.from_tuples(tuples, names=(
            'homeMicrotype', 'populationGroupType', 'tripPurpose', 'originMicrotype', 'destinationMicrotype',
            'distanceBin',
            'mode'))
        return pd.DataFrame({'Mode split': self.__modeSplitData.flatten(),
                             'Utility': self.__currentUtility.flatten(),
                             'Trips': np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData).flatten()},
                            index=mi)

    def updateTripStartRate(self, newTripStartRate):
        self.__tripRate[self.__tripRate > 0] = newTripStartRate
        # nonZero = tripRate > 0
        # tripRate[nonZero] = newTripStartRate
        # newTripRate = np.ones(np.shape(tripRate)) * newTripStartRate
        # np.copyto(self.__tripRate, newTripRate)

    def nModes(self):
        return len(self.__modes)

    def keys(self):
        return product(self.__scenarioData.diToIdx.keys(), self.__scenarioData.odiToIdx.keys())

    # def __iter__(self) -> ((DemandIndex, ODindex), np.ndarray):
    #     return np.ndenumerate(self.__numpy)

    def __setitem__(self, key: (DemandIndex, ODindex), value: ModeSplit):
        self.__modeSplit[key] = value

    @property
    def modeSplitData(self):
        return self.__modeSplitData

    def __getitem__(self, item: (DemandIndex, ODindex)) -> ModeSplit:
        if item in self:
            return self.__modeSplit[item]
        else:  # else return empty mode split
            (demandIndex, odi) = item
            print("WTF")

    def __contains__(self, item):
        """ Return true if the correct value"""
        if item in self.__modeSplit:
            return True
        else:
            return False

    def initializeDemand(self, population: Population, originDestination: OriginDestination,
                         tripGeneration: TripGeneration, microtypes: MicrotypeCollection,
                         distanceBins: DistanceBins, transitionMatrices: TransitionMatrices, timePeriods: TimePeriods,
                         currentTimePeriod: int, numpyData: dict, multiplier=1.0):
        self.__population = population
        self.__distanceBins = distanceBins
        self.__transitionMatrices = transitionMatrices
        self.__originDestination = originDestination
        self.__tripGeneration = tripGeneration
        self.timePeriodDuration = timePeriods[currentTimePeriod]

        weights = np.zeros(len(self.odiToIdx), dtype=float)

        self.__shape = self.__modeSplitData.shape
        self.__modeSplitData[:, :, self.modeToIdx['auto']] = 0.7
        self.__modeSplitData[:, :, self.modeToIdx['walk']] = 0.3

        for demandIndex, utilityParams in population:
            od = originDestination[demandIndex]
            ratePerHourPerCapita = tripGeneration[demandIndex.populationGroupType, demandIndex.tripPurpose] * multiplier
            pop = population.getPopulation(demandIndex.homeMicrotype, demandIndex.populationGroupType)

            for odi, portion in od.items():
                tripRatePerHour = ratePerHourPerCapita * pop * portion
                self.tripRate += tripRatePerHour
                demandForPMT = ratePerHourPerCapita * pop * portion * distanceBins[odi.distBin]

                self.demandForPMT += demandForPMT
                self.pop += pop

                currentODindex = self.odiToIdx[odi]
                if demandIndex in self.diToIdx:
                    currentPopIndex = self.diToIdx[demandIndex]
                    weights[currentODindex] += tripRatePerHour  # demandForPMT # CHANGED
                    self.__tripRate[currentPopIndex, currentODindex] = tripRatePerHour
                else:
                    if tripRatePerHour > 0:
                        print("What do we have here? Lost {} trips".format(tripRatePerHour))

        otherMatrix = transitionMatrices.averageMatrix(weights)
        microtypes.transitionMatrix.updateMatrix(otherMatrix)

    def updateTripGeneration(self, microtypes: MicrotypeCollection, multiplier=1.0):
        self.tripRate = 0.0
        self.pop = 0.0
        self.demandForPMT = 0.0
        weights = np.zeros(len(self.odiToIdx), dtype=float)

        # TODO: Vectorize this
        for demandIndex, utilityParams in self.__population:
            od = self.__originDestination[demandIndex]
            ratePerHourPerCapita = self.__tripGeneration[
                                       demandIndex.populationGroupType, demandIndex.tripPurpose] * multiplier
            pop = self.__population.getPopulation(demandIndex.homeMicrotype, demandIndex.populationGroupType)

            for odi, portion in od.items():
                tripRatePerHour = ratePerHourPerCapita * pop * portion
                self.tripRate += tripRatePerHour
                demandForPMT = ratePerHourPerCapita * pop * portion * self.__distanceBins[odi.distBin]

                self.demandForPMT += demandForPMT
                self.pop += pop

                currentODindex = self.odiToIdx[odi]
                currentPopIndex = self.diToIdx[demandIndex]
                weights[currentODindex] += tripRatePerHour
                self.__tripRate[currentPopIndex, currentODindex] = tripRatePerHour

        otherMatrix = self.__transitionMatrices.averageMatrix(weights)
        microtypes.transitionMatrix.updateMatrix(otherMatrix)

    def updateMFD(self, microtypes: MicrotypeCollection, nIters=3, utilitiesArray=None, modeSplitArray=None):
        if utilitiesArray is not None:
            np.copyto(self.__modeSplitData, modeSplitFromUtils(utilitiesArray))
            np.copyto(self.__currentUtility, utilitiesArray)
        if modeSplitArray is not None:
            np.copyto(self.__modeSplitData, modeSplitArray)
        for microtypeID, microtype in microtypes:
            microtype.resetDemand()

        startsByMode = np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData)
        discountStartsByMode = np.einsum('...,...i->...i', self.__tripRate[self.__seniorODIs, :],
                                         self.__modeSplitData[self.__seniorODIs, :, :])
        startsByOrigin = np.einsum('ijk,jl->lk', startsByMode, self.__toStarts)
        discountStartsByOrigin = np.einsum('ijk,jl->lk', discountStartsByMode, self.__toStarts)
        startsByDestination = np.einsum('ijk,jl->lk', startsByMode, self.__toEnds)
        passengerMilesByMicrotype = np.einsum('ijk,jl->lk', startsByMode, self.__toThroughDistance)
        vehicleMilesByMicrotype = passengerMilesByMicrotype.copy()

        for mode, modeIdx in self.modeToIdx.items():
            for mID, mIdx in self.microtypeIdToIdx.items():
                if microtypes[mID].networks.getMode(mode).fixedVMT:
                    vehicleMilesByMicrotype[mIdx, modeIdx] = microtypes[mID].networks.getModeVMT(mode)
        newData = np.stack([startsByOrigin, startsByDestination, passengerMilesByMicrotype, vehicleMilesByMicrotype,
                            discountStartsByOrigin], axis=-1)

        microtypes.updateNumpyDemand(newData)
        weights = np.sum(startsByMode[:, :, self.modeToIdx["auto"]], axis=0)

        """
        Step one: Update transition matrix (depends only on mode split)
        """

        otherMatrix = self.__transitionMatrices.averageMatrix(weights)
        microtypes.transitionMatrix.updateMatrix(otherMatrix)

        """
        Step two: Start from uncongested networks with no blocked distance
        """

        for microtypeID, microtype in microtypes:
            for n in microtype.networks:
                # n.getNetworkStateData().resetBlockedDistance()
                # n.getNetworkStateData().resetNonAutoAccumulation()
                n.resetSpeeds()

        for it in range(nIters):
            # for microtypeID, microtype in microtypes:
            #     microtype.updateNetworkSpeeds(1)
            """
            Step three: Update car average speeds at the microtype level, given blocked distance
            """

            # print([(a, b.blockedDistance, b.nonAutoAccumulation) for a, b in microtypes.collectedNetworkStateData if (b.nonAutoAccumulation > 0)])
            microtypes.transitionMatrixMFD(self.timePeriodDuration)
            microtypes.updateDedicatedDistance()
            """
            Step four: Update blocked distance, given average car speeds
            """
            for microtypeID, microtype in microtypes:
                microtype.updateNetworkSpeeds()

    def utility(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics):
        return utils(self.__population.numpy, collectedChoiceCharacteristics.numpy)

    def currentUtilities(self):
        return self.__currentUtility

    def calculateUtilities(self, choiceCharacteristicsArray: np.ndarray) -> np.ndarray:
        newUtilities = utilsWithExcludedModes(self.__population.numpy, choiceCharacteristicsArray,
                                              self.__population.transitLayerUtility)
        # if self.__currentUtility.size > 0:
        #     np.copyto(self.__currentUtility, newUtilities)
        # else:
        #     self.__currentUtility = newUtilities
        return newUtilities

    def getTotalModeSplit(self, userClass=None, microtypeID=None, distanceBin=None, otherModeSplit=None) -> ModeSplit:
        demandForTrips = 0
        demandForDistance = 0
        trips = dict()
        for (di, odi), ms in self.__modeSplit.items():
            relevant = ((userClass is None) or (di.populationGroupType == userClass)) & (
                    (microtypeID is None) or (di.homeMicrotype == microtypeID)) & (
                               (distanceBin is None) or (odi.distBin == distanceBin))
            if relevant:
                # ar = self.__modeSplitData[self.__diToIdx[di], self.__odiToIdx[odi], :]
                for mode, split in ms:
                    new_demand = trips.setdefault(mode, 0) + split * ms.demandForTripsPerHour
                    trips[mode] = new_demand
                demandForTrips += ms.demandForTripsPerHour
                demandForDistance += ms.demandForPmtPerHour
        for mode in trips.keys():
            if otherModeSplit is not None:
                trips[mode] /= (demandForTrips * 2.)
                trips[mode] += otherModeSplit[mode] / 2.
            else:
                trips[mode] /= demandForTrips
        return ModeSplit(trips, demandForTrips, demandForDistance)

    def getMatrixModeCounts(self):
        modeCounts = np.einsum('ij,ijk->k', self.__tripRate, self.__modeSplitData)
        return modeCounts

    def getMatrixUserCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics) -> np.ndarray:
        startsByMode = np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData)
        costByMode = - utils(self.__population.numpyCost, collectedChoiceCharacteristics.numpy)
        return startsByMode * costByMode

    def getSummedCharacteristics(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics) -> np.ndarray:
        startsByMode = np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData)
        totalsByModeAndCharacteristic = np.einsum('ijk,ijkl->kl', startsByMode, collectedChoiceCharacteristics.numpy)
        if np.any(np.isnan(totalsByModeAndCharacteristic)):
            print('Something went wrong')
        return totalsByModeAndCharacteristic

    def getUserCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                     originDestination: OriginDestination) -> CollectedTotalUserCosts:
        out = CollectedTotalUserCosts()
        for demandIndex, utilityParams in self.__population:
            od = originDestination[demandIndex]
            demandClass = self.__population[demandIndex]
            for odi, portion in od.items():
                ms = self[(demandIndex, odi)]

                for mode in ms.keys():
                    mcc = collectedChoiceCharacteristics[odi, mode]
                    cost, inVehicle, outVehicle, demandForTripsPerHour, distance = demandClass.getCostPerCapita(mcc, ms,
                                                                                                                [mode])
                    if demandForTripsPerHour > 0:
                        out[demandIndex, mode] = TotalUserCosts(cost * demandForTripsPerHour, 0.0,
                                                                inVehicle * demandForTripsPerHour,
                                                                outVehicle * demandForTripsPerHour,
                                                                demandForTripsPerHour,
                                                                demandForTripsPerHour * distance)
        return out.updateTotals()

    def __str__(self):
        return "Trips: " + str(self.tripRate) + ", PMT: " + str(self.demandForPMT)


def utils(popVars: np.ndarray, choiceChars: np.ndarray) -> np.ndarray:
    """ Indices:
    i: population group (demand index)
    j: OD index
    k: mode
    l: parameter
    """
    utils = np.einsum('ikl,ijkl->ijk', popVars, choiceChars)
    return utils


def utilsWithExcludedModes(popVars: np.ndarray, choiceChars: np.ndarray, transitLayerUtility: np.ndarray) -> np.ndarray:
    """ Indices:
    i: population group (demand index)
    j: OD index
    k: mode
    l: parameter
    """
    utils = np.einsum('ikl,ijkl->ijk', popVars, choiceChars)
    paddedUtils = utils[:, :, :, None]  # np.repeat(utils[:, :, :, None], transitLayerUtility.shape[-1], axis=3)
    paddedTransitLayer = transitLayerUtility[None, None, :, :]
    # np.repeat(np.repeat(transitLayerUtility[None, :, :], utils.shape[1], axis=0)[None, :, :, :],
    # utils.shape[0], axis=0)
    return paddedUtils + paddedTransitLayer


def modeSplitFromUtils(utilities: np.ndarray) -> np.ndarray:
    expUtils = np.exp(utilities)
    probabilities = expUtils / np.expand_dims(np.nansum(expUtils, axis=2), 2)
    probabilities[np.isnan(expUtils)] = 0
    return probabilities


def modeSplitFromUtilsWithExcludedModes(utilities: np.ndarray, transitLayerPortion: np.ndarray) -> np.ndarray:
    expUtils = np.exp(utilities)
    probabilities = expUtils / np.expand_dims(np.nansum(expUtils, axis=2), 2)
    probabilities[np.isnan(expUtils)] = 0
    """ Indices:
    i: population group (demand index)
    j: OD index
    k: mode
    l: transit layer
    """
    weightedProbabilities = np.einsum('ijkl,jl->ijk', probabilities, transitLayerPortion)
    return weightedProbabilities


def correctModeSplit(modeSplit: np.ndarray) -> np.ndarray:
    modeSplit[np.isnan(modeSplit)] = 0.0
    if np.any(modeSplit < 0.0):
        badSplits = np.any(modeSplit < 0, axis=-1)

        positiveValues = modeSplit[badSplits, :]
        positiveValues[positiveValues < 0] = - 0.5 * positiveValues[positiveValues < 0]

        corrected = positiveValues
        modeSplit[badSplits, :] = corrected
        return modeSplit / modeSplit.sum(axis=-1)[:, None]
    else:
        return modeSplit / modeSplit.sum(axis=-1)[:, None]
