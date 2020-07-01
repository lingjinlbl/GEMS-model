from utils.OD import Trip, TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex
from utils.population import Population
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
                         trips: TripCollection):
        for it, out in population:
            od = originDestination[it]
            print(it)
