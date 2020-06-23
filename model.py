import pandas as pd
import os
from utils.microtype import Microtype
from utils.network import Network, NetworkCollection, NetworkFlowParams, Mode, BusMode, BusModeParams
from typing import Dict, List


class Model:
    def __init__(self, path: str):
        self.path = path
        self.__microtypes = dict()
        self.__population = dict()
        self.readFiles()

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    def readFiles(self):
        microtypes = pd.read_csv(os.path.join(self.path, "microtypes.csv"))
        subNetworks = pd.read_csv(os.path.join(self.path, "subnetworks.csv"))
        modeToSubnetwork = pd.read_csv(os.path.join(self.path, "mode-to-subnetwork.csv"))
        for m, grouped in subNetworks.groupby('MicrotypeID'):
            subNetworkToModes = dict()
            allModes = set()
            for row in grouped.itertuples():
                joined = modeToSubnetwork.loc[modeToSubnetwork['SubnetworkID'] == row.SubnetworkID]
                subNetwork = Network(row.Length, NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50))
                for n in joined.itertuples():
                    subNetworkToModes.setdefault(subNetwork, []).append(n.ModeType)
                    allModes.add(n.ModeType)
            for mode in allModes:
                vals = pd.read_csv(os.path.join(self.path,"modes",mode.lower()+".csv"))
                print("AAH")



if __name__ == "__main__":
    a = Model("input-data")
