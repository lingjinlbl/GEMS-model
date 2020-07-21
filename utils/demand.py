from .OD import TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit
from .choiceCharacteristics import CollectedChoiceCharacteristics, filterAllocation
from .microtype import MicrotypeCollection
from .misc import DistanceBins
from .population import Population


class Demand:
    def __init__(self):
        self.__modeSplit = dict()
        self.tripRate = 0.0
        self.demandForPMT = 0.0
        self.__population = Population()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()

    def __setitem__(self, key: (DemandIndex, ODindex), value: ModeSplit):
        self.__modeSplit[key] = value

    def __getitem__(self, item: (DemandIndex, ODindex)) -> ModeSplit:
        return self.__modeSplit[item]

    def initializeDemand(self, population: Population, originDestination: OriginDestination,
                         tripGeneration: TripGeneration, trips: TripCollection, microtypes: MicrotypeCollection,
                         distanceBins: DistanceBins, multiplier=1.0):
        self.__population = population
        self.__trips = trips
        self.__distanceBins = distanceBins
        for demandIndex, utilityParams in population:
            od = originDestination[demandIndex]
            ratePerHourPerCapita = tripGeneration[demandIndex.populationGroupType, demandIndex.tripPurpose] * multiplier
            pop = population.getPopulation(demandIndex.homeMicrotype, demandIndex.populationGroupType)
            for odi, portion in od.items():
                trip = trips[odi]
                common_modes = [microtypes[trip.odIndex.o].mode_names, microtypes[trip.odIndex.d].mode_names]
                # # Now we're switching over to only looking at origin and destination modes
                # common_modes = []
                # for microtypeID, allocation in trip.allocation:
                #     if allocation > 0:
                #         common_modes.append(microtypes[microtypeID].mode_names)
                modes = set.intersection(*common_modes)
                tripRatePerHour = ratePerHourPerCapita * pop
                self.tripRate += tripRatePerHour
                demandForPMT = ratePerHourPerCapita * pop * distanceBins[odi.distBin]
                self.demandForPMT += demandForPMT
                modeSplit = dict()
                for mode in modes:
                    if mode == "auto":
                        modeSplit[mode] = 1.0
                    else:
                        modeSplit[mode] = 0.0
                self[demandIndex, odi] = ModeSplit(modeSplit, tripRatePerHour, demandForPMT)

    def updateMFD(self, microtypes: MicrotypeCollection):
        for microtypeID, microtype in microtypes:
            microtype.resetDemand()

        for (di, odi), ms in self.__modeSplit.items():
            assert (isinstance(ms, ModeSplit))
            assert (isinstance(odi, ODindex))
            assert (isinstance(di, DemandIndex))
            for mode, split in ms:
                microtypes[odi.o].addModeStarts(mode, ms.demandForTripsPerHour * split)
                microtypes[odi.d].addModeEnds(mode, ms.demandForTripsPerHour * split)
                newAllocation = filterAllocation(mode, self.__trips[odi].allocation, microtypes)
                # through_microtypes = []
                # allocation = []
                # for m, a in self.__trips[odi].allocation:
                #     if (a > 0) & (mode in microtypes[m].mode_names):
                #         through_microtypes.append(m)
                #         allocation.append(a)
                # allocation = np.array(allocation)
                # allocation /= np.sum(allocation)
                for k, portion in newAllocation.items():
                    microtypes[k].addModeDemandForPMT(mode, ms.demandForTripsPerHour * split,
                                                      self.__distanceBins[odi.distBin])

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

    def getTotalModeSplit(self) -> dict:
        demand = 0
        trips = dict()
        for ms in self.__modeSplit.values():
            for mode, split in ms:
                new_demand = trips.setdefault(mode, 0) + split * ms.demandForTripsPerHour
                trips[mode] = new_demand
            demand += ms.demandForTripsPerHour
        for mode in trips.keys():
            trips[mode] /= demand
        return trips

    def __str__(self):
        return "Trips: " + str(self.tripRate) + ", PMT: " + str(self.demandForPMT)
