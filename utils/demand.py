from .OD import TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit
from .population import Population
from .microtype import MicrotypeCollection
from .misc import TimePeriods, DistanceBins
from .choiceCharacteristics import CollectedChoiceCharacteristics


class Demand:
    def __init__(self):
        self.__modeSplit = dict()
        self.tripRate = 0.0
        self.demandForPMT = 0.0
        self.__population = Population()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins

    def __setitem__(self, key: (DemandIndex, ODindex), value: ModeSplit):
        self.__modeSplit[key] = value

    def __getitem__(self, item: (DemandIndex, ODindex)) -> ModeSplit:
        return self.__modeSplit[item]

    def initializeDemand(self, population: Population, originDestination: OriginDestination, tripGeneration: TripGeneration,
                         trips: TripCollection, microtypes: MicrotypeCollection, distanceBins: DistanceBins):
        self.__population = population
        self.__trips = trips
        self.__distanceBins = distanceBins
        for demandIndex, utilityParams in population:
            od = originDestination[demandIndex]
            rate = tripGeneration[demandIndex.populationGroupType, demandIndex.tripPurpose]
            pop = population.getPopulation(demandIndex.homeMicrotype, demandIndex.populationGroupType)
            for odi, portion in od.items():
                trip = trips[odi]
                common_modes = []
                for microtypeID, allocation in trip.allocation:
                    if allocation > 0:
                        common_modes.append(microtypes[microtypeID].mode_names)
                modes = set.intersection(*common_modes)
                tripRate = rate * pop
                self.tripRate += tripRate
                demandForPMT = rate * pop * distanceBins[odi.distBin]
                self.demandForPMT += demandForPMT
                modeSplit = dict()
                for mode in modes:
                    if mode == "auto":
                        modeSplit[mode] = 1.0
                    else:
                        modeSplit[mode] = 0.0
                self[demandIndex, odi] = ModeSplit(modeSplit, tripRate, demandForPMT)

    def updateMFD(self, microtypes: MicrotypeCollection):
        for microtypeID, microtype in microtypes:
            microtype.resetDemand()

        for (di, odi), ms in self.__modeSplit.items():
            assert(isinstance(ms, ModeSplit))
            assert(isinstance(odi, ODindex))
            assert(isinstance(di, DemandIndex))
            for mode, split in ms:
                microtypes[odi.o].addModeStarts(mode, ms.demandForTrips * split / 100) # TODO: UNSCALE
                microtypes[odi.d].addModeEnds(mode, ms.demandForTrips * split / 100)
                for k, portion in self.__trips[odi].allocation:
                    microtypes[k].addModeDemandForPMT(mode, ms.demandForTrips * split / 100, self.__distanceBins[odi.distBin])

        for microtypeID, microtype in microtypes:
            microtype.updateNetworkSpeeds(10)

    def updateModeSplit(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                        originDestination: OriginDestination):
        for demandIndex, utilityParams in self.__population:
            od = originDestination[demandIndex]
            for odi, portion in od.items():
                dg = self.__population[demandIndex]
                ms = self.__population[demandIndex].updateModeSplit(collectedChoiceCharacteristics[odi])
                self[demandIndex, odi].updateMapping(ms)


    def __str__(self):
        return "Trips: " + str(self.tripRate) + ", PMT: " + str(self.demandForPMT)

