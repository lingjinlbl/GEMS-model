from utils.OD import TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit
from utils.population import Population
from utils.microtype import Microtype, MicrotypeCollection
from utils.misc import TimePeriods, DistanceBins


class Demand:
    def __init__(self):
        self.__modeSplit = dict()
        self.tripRate = 0.0
        self.demandForPMT = 0.0
        self.__population = Population

    def __setitem__(self, key: (DemandIndex, ODindex), value: ModeSplit):
        self.__modeSplit[key] = value

    def __getitem__(self, item: (DemandIndex, ODindex)) -> ModeSplit:
        return self.__modeSplit[item]

    def initializeDemand(self, population: Population, originDestination: OriginDestination, tripGeneration: TripGeneration,
                         trips: TripCollection, microtypes: MicrotypeCollection, distanceBins: DistanceBins):
        self.__population = population
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


    def __str__(self):
        return "Trips: " + str(self.tripRate) + ", PMT: " + str(self.demandForPMT)

