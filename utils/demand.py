from utils.OD import Trip, TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit
from utils.population import Population
from utils.microtype import MicrotypeCollection
import pandas as pd


class Demand:
    def __init__(self):
        self.__demand = dict()
        self.__modeSplit = dict()

    def __setitem__(self, key: (DemandIndex, ODindex), value: float):
        self.__demand[key] = value

    def __getitem__(self, item):
        return self.__demand[item]

    def initializeDemand(self, population: Population, originDestination: OriginDestination, tripGeneration: TripGeneration,
                         trips: TripCollection, microtypes: MicrotypeCollection):
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
                self[demandIndex, odi] = rate * pop
                modeSplit = dict()
                for mode in modes:
                    if mode == "auto":
                        modeSplit[mode] = 1.0
                    else:
                        modeSplit[mode] = 0.0
                self.__modeSplit[demandIndex, odi] = ModeSplit(modeSplit)
                print('A')

