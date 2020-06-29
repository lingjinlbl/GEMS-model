import pandas as pd
import os
from utils.microtype import MicrotypeCollection
from utils.network import Network, NetworkCollection, NetworkFlowParams, BusModeParams, \
    AutoModeParams, Costs
from utils.OD import Trip, TripCollection
from utils.population import PopulationGroup, Population
from typing import Dict, List


class Model:
    def __init__(self, path: str):
        self.path = path
        self.microtypes = MicrotypeCollection(path)
        self.trips = TripCollection()
        self.population = Population()
        self.readFiles()

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    def readFiles(self):
        subNetworkData = pd.read_csv(os.path.join(self.path, "SubNetworks.csv"))
        modeToSubNetworkData = pd.read_csv(os.path.join(self.path, "ModeToSubNetwork.csv"))
        self.microtypes.importMicrotypes(subNetworkData, modeToSubNetworkData)
        self.trips.importTrips(pd.read_csv(os.path.join(self.path, "MicrotypeAssignment.csv")))
        self.population.importPopulation(pd.read_csv(os.path.join(self.path, "Population.csv")))



if __name__ == "__main__":
    a = Model("input-data")
    print("aah")
