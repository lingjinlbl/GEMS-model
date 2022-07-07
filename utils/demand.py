from itertools import product

import numpy as np

from .data import ScenarioData

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


class Accessibility:
    def __init__(self, scenarioData, data):
        self.__scenarioData = scenarioData
        self.__data = data
        fixedData = data.getInvariants()
        self.__activityDensity = fixedData["activityDensity"]
        self.__toEnds = fixedData["toEnds"]
        self.__toTripPurpose = fixedData["toTripPurpose"]
        self.__toHomeMicrotype = fixedData["toHomeMicrotype"]
        self.__toODI = fixedData["toODI"]
        demand = data.getDemand()
        self.__tripRate = demand["tripRate"]
        self.__utilities = demand["utilities"]
        self.__modeSplit = demand["modeSplit"]

    @property
    def tripPurposeToIdx(self):
        return self.__scenarioData.tripPurposeToIdx

    @property
    def microtypeIdToIdx(self):
        return self.__scenarioData.microtypeIdToIdx

    @property
    def passengerModeToIdx(self):
        return self.__scenarioData.passengerModeToIdx

    @property
    def populationGroupToIdx(self):
        return self.__scenarioData.populationGroupToIdx

    @property
    def timePeriods(self):
        return self.__scenarioData["timePeriods"]["TimePeriodID"]

    def init(self):
        np.copyto(self.__activityDensity, pd.pivot_table(self.__scenarioData["activityDensity"], values="DensityKmSq",
                                                         columns="TripPurposeID", index="MicrotypeID",
                                                         aggfunc="sum").reindex(
            index=pd.Index(self.microtypeIdToIdx.keys()),
            columns=pd.Index(self.tripPurposeToIdx.keys()),
            fill_value=0.0).fillna(0.0))

    def calculate(self):
        activityDensity = self.__activityDensity
        toODI = self.__toODI
        toEnds = self.__toEnds
        inverseUtility = np.exp(self.__utilities)
        tripRate = self.__tripRate
        """
        t -> time: 3
        i -> demand index (homeMicrotypeID, populationGroupTypeID, tripPurposeID): 16
        o -> odi: 32
        m -> mode: 5
        d -> destination: 4
        p -> tripPurpose: 2
        h -> home microtype
        g -> population group
        ----- 
        
        """
        midx = pd.MultiIndex.from_product(
            [self.microtypeIdToIdx.keys(), self.tripPurposeToIdx.keys(), self.populationGroupToIdx.keys(),
             self.passengerModeToIdx.keys()],
            names=["Microtype ID", "Trip Purpose", "Population Group", "Mode"])
        accessibilityArray = np.einsum("tiom,tio,od,ihpg,dp->thpgm", inverseUtility,
                                       tripRate / (tripRate.sum(axis=2)[:, :, None] + 1.), toEnds, toODI,
                                       activityDensity,
                                       optimize=['einsum_path', (0, 1), (0, 2), (1, 2), (0, 1)])
        out = pd.DataFrame(accessibilityArray.reshape((accessibilityArray.shape[0], -1)).T, index=midx)
        return out

    def calculateByDI(self):
        activityDensity = self.__activityDensity
        toODI = self.__toODI
        toEnds = self.__toEnds
        inverseUtility = np.exp(self.__utilities)
        modeSplit = self.__modeSplit
        tripRate = self.__tripRate
        """
        t -> time: 3
        i -> demand index (homeMicrotypeID, populationGroupTypeID, tripPurposeID): 16
        o -> odi: 32
        m -> mode: 5
        d -> destination: 4
        p -> tripPurpose: 2
        h -> home microtype
        g -> population group
        ----- "Microtype ID", "Population Group", "Trip Purpose"

        """
        midx = pd.MultiIndex.from_product(
            [self.microtypeIdToIdx.keys(), self.populationGroupToIdx.keys(), self.tripPurposeToIdx.keys()],
            names=["Microtype ID", "Population Group", "Trip Purpose"])
        accessibilityArray = np.einsum("tiom,tio,tiom,ihpg,od,dp->hpg", inverseUtility,
                                       tripRate / (tripRate.sum(axis=2)[:, :, None] + 1.), modeSplit, toODI, toEnds,
                                       activityDensity,
                                       optimize=['einsum_path', (0, 2), (0, 4), (1, 2), (1, 2), (0, 1)])
        out = dict()
        for hm, midx in self.microtypeIdToIdx.items():
            da = dict()
            for tp, pidx in self.tripPurposeToIdx.items():
                da[tp] = pd.Series(accessibilityArray[midx, pidx, :], index=self.populationGroupToIdx.keys())
            out[hm] = pd.DataFrame(da)
        return pd.concat(out, names=['Home Microtype', 'Population Group'])


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
        self.__microtypeCosts = numpyData['microtypeCosts']
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
    def microtypeCosts(self):
        return self.__microtypeCosts

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
    def passengerModeToIdx(self):
        return self.__scenarioData.passengerModeToIdx

    @property
    def microtypeIdToIdx(self):
        return self.__scenarioData.microtypeIdToIdx

    @property
    def utilities(self):
        return self.__currentUtility

    @utilities.setter
    def utilities(self, newUtilities):
        np.copyto(self.__currentUtility, newUtilities)

    def modeSplitDataFrame(self):
        dis = pd.MultiIndex.from_tuples([di.toTuple() for di in self.diToIdx.keys()],
                                        names=('homeMicrotype', 'populationGroupType', 'tripPurpose'))
        odis = pd.MultiIndex.from_tuples([odi.toTuple() for odi in self.odiToIdx.keys()],
                                         names=('originMicrotype', 'destinationMicrotype', 'distanceBin'))
        modes = list(self.passengerModeToIdx.keys())
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
        if 'walk' in self.modeToIdx:
            self.__modeSplitData[:, :, self.modeToIdx['walk']] = 0.3
        elif 'bus' in self.modeToIdx:
            self.__modeSplitData[:, :, self.modeToIdx['bus']] = 0.3  # // TODO: CHANGE BACK
        else:
            raise NotImplementedError("Currently the model doesn't work without either a walk or bus mode")

        for demandIndex, utilityParams in population:
            od = originDestination[demandIndex]
            ratePerHourPerCapita = tripGeneration[demandIndex.populationGroupType, demandIndex.tripPurpose] * multiplier
            pop = population.getPopulation(demandIndex.homeMicrotype, demandIndex.populationGroupType)

            for odi, portion in od.items():
                tripRatePerHour = ratePerHourPerCapita * pop * portion
                if tripRatePerHour <= 0.0:
                    continue
                self.tripRate += tripRatePerHour
                demandForPMT = ratePerHourPerCapita * pop * portion * distanceBins[odi.distBin]

                self.demandForPMT += demandForPMT
                self.pop += pop
                if odi in self.odiToIdx:
                    currentODindex = self.odiToIdx[odi]
                    if demandIndex in self.diToIdx:
                        currentPopIndex = self.diToIdx[demandIndex]
                        weights[currentODindex] += demandForPMT  # tripRatePerHour  # demandForPMT # CHANGED
                        self.__tripRate[currentPopIndex, currentODindex] = tripRatePerHour
                    else:
                        if tripRatePerHour > 0:
                            print("What do we have here? Lost {} trips".format(tripRatePerHour))
                else:
                    print("Lost {} trips from ODI ".format(tripRatePerHour), str(odi))
        microtypes.updateTransitionMatrix(transitionMatrices.averageMatrix(weights))

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
                weights[currentODindex] += demandForPMT  # tripRatePerHour
                self.__tripRate[currentPopIndex, currentODindex] = tripRatePerHour

        microtypes.updateTransitionMatrix(self.__transitionMatrices.averageMatrix(weights))

    def updateMFD(self, microtypes: MicrotypeCollection, nIters=3, utilitiesArray=None, modeSplitArray=None):
        if utilitiesArray is not None:
            np.copyto(self.__modeSplitData, modeSplitFromUtils(utilitiesArray))
            np.copyto(self.__currentUtility, utilitiesArray)
        if modeSplitArray is not None:
            np.copyto(self.__modeSplitData, modeSplitArray)
        for microtypeID, microtype in microtypes:
            microtype.resetDemand()

        startsByMode = np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData,
                                 optimize=['einsum_path', (0, 1)])
        discountStartsByMode = np.einsum('...,...i->...i', self.__tripRate[self.__seniorODIs, :],
                                         self.__modeSplitData[self.__seniorODIs, :, :],
                                         optimize=['einsum_path', (0, 1)])
        startsByOrigin = np.einsum('ijk,jl->lk', startsByMode, self.__toStarts, optimize=['einsum_path', (0, 1)])
        discountStartsByOrigin = np.einsum('ijk,jl->lk', discountStartsByMode, self.__toStarts,
                                           optimize=['einsum_path', (0, 1)])
        startsByDestination = np.einsum('ijk,jl->lk', startsByMode, self.__toEnds, optimize=['einsum_path', (0, 1)])
        passengerMilesByMicrotype = np.einsum('ijk,jl->lk', startsByMode, self.__toThroughDistance,
                                              optimize=['einsum_path', (0, 1)])
        vehicleMilesByMicrotype = passengerMilesByMicrotype.copy()

        for mode, modeIdx in self.modeToIdx.items():
            for mID, mIdx in self.microtypeIdToIdx.items():
                if microtypes[mID].networks.fixedVMT(mode):
                    if mode in self.passengerModeToIdx:
                        vehicleMilesByMicrotype[mIdx, modeIdx] = microtypes[mID].networks.getModeVMT(mode)
                        # We're dealing with fixed VMT from freight elsewhere
        newData = np.stack([startsByOrigin, startsByDestination, passengerMilesByMicrotype, vehicleMilesByMicrotype,
                            discountStartsByOrigin], axis=-1)

        microtypes.updateNumpyPassengerDemand(newData)
        weights = np.array(
            [self.__scenarioData.distanceBinToDistance[odi.distBin] for odi in self.odiToIdx.keys()]) * np.sum(
            startsByMode[:, :, self.modeToIdx["auto"]], axis=0)
        # weights = np.sum(startsByMode[:, :, self.modeToIdx["auto"]], axis=0)

        """
        Step one: Update transition matrix (depends only on mode split)
        """

        microtypes.updateTransitionMatrix(self.__transitionMatrices.averageMatrix(weights))

        """
        Step two: Start from uncongested networks with no blocked distance
        """

        for microtypeID, microtype in microtypes:
            for n in microtype.networks:
                n.resetSpeeds()

        for it in range(nIters):
            """
            Step three: Update car average speeds at the microtype level, given blocked distance
            """

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

    def calculateUtilities(self, choiceCharacteristicsArray: np.ndarray, excludeModes=True) -> np.ndarray:
        if excludeModes:
            return utilsWithExcludedModes(self.__population.numpy, choiceCharacteristicsArray,
                                          self.__population.transitLayerUtility)
        else:
            return utils(self.__population.numpy, choiceCharacteristicsArray)

    def getTotalModeSplit(self, userClass=None, microtypeID=None, distanceBin=None, otherModeSplit=None) -> ModeSplit:
        demandForTrips = 0
        demandForDistance = 0
        trips = dict()
        for (di, odi), ms in self.__modeSplit.items():
            relevant = ((userClass is None) or (di.populationGroupType == userClass)) & (
                    (microtypeID is None) or (di.homeMicrotype == microtypeID)) & (
                               (distanceBin is None) or (odi.distBin == distanceBin))
            if relevant:
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

    def getMatrixPolicyRevenues(self):
        tripCounts = np.einsum('ij,ijk->ijk', self.__tripRate, self.__modeSplitData)
        toStarts = self.__toStarts
        toEnds = self.__toEnds
        toThroughDistance = self.__toThroughDistance
        microtypeCosts = self.microtypeCosts
        nPassengerModes = len(self.passengerModeToIdx)
        startRevenues = np.einsum("dom, ol, ldm->lm", tripCounts, toStarts,
                                  microtypeCosts[:, :, :nPassengerModes,
                                  ScenarioData.costTypeToIdx['perStartPublicCost']])
        endRevenues = np.einsum("dom, ol, ldm->lm", tripCounts, toEnds,
                                microtypeCosts[:, :, :nPassengerModes, ScenarioData.costTypeToIdx['perEndPublicCost']])
        throughRevenues = np.einsum("dom, ol, ldm->lm", tripCounts, toThroughDistance,
                                    microtypeCosts[:, :, :nPassengerModes,
                                    ScenarioData.costTypeToIdx['perMilePublicCost']])

        return startRevenues + endRevenues + throughRevenues

    def getSummedCharacteristics(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics) -> np.ndarray:
        startsByMode = np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData)
        totalsByModeAndCharacteristic = np.einsum('ijk,ijkl->kl', startsByMode, collectedChoiceCharacteristics.numpy)
        if np.any(np.isnan(totalsByModeAndCharacteristic)):
            print('Something went wrong')
        return totalsByModeAndCharacteristic

    # def getUserCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
    #                  originDestination: OriginDestination) -> CollectedTotalUserCosts:
    #     out = CollectedTotalUserCosts()
    #     for demandIndex, utilityParams in self.__population:
    #         od = originDestination[demandIndex]
    #         demandClass = self.__population[demandIndex]
    #         for odi, portion in od.items():
    #             ms = self[(demandIndex, odi)]
    #
    #             for mode in ms.keys():
    #                 mcc = collectedChoiceCharacteristics[odi, mode]
    #                 cost, inVehicle, outVehicle, demandForTripsPerHour, distance = demandClass.getCostPerCapita(mcc, ms,
    #                                                                                                             [mode])
    #                 if demandForTripsPerHour > 0:
    #                     out[demandIndex, mode] = TotalUserCosts(cost * demandForTripsPerHour, 0.0,
    #                                                             inVehicle * demandForTripsPerHour,
    #                                                             outVehicle * demandForTripsPerHour,
    #                                                             demandForTripsPerHour,
    #                                                             demandForTripsPerHour * distance)
    #     return out.updateTotals()

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
    # TODO Optimize this
    weightedProbabilities = np.einsum('ijkl,jl->ijk', probabilities, transitLayerPortion,
                                      optimize=['einsum_path', (0, 1)])
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
