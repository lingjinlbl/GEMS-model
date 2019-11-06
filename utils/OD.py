import numpy as np
from utils.microtype import Microtype
from typing import Dict, List


class Allocation:
    def __init__(self, mapping=None):
        if mapping is None:
            self._mapping = Dict[Microtype,float]
        else:
            assert(isinstance(mapping, Dict))
            self._mapping = mapping

    def __setitem__(self, key, value):
        self._mapping[key] = value

    def __getitem__(self, item):
        return self._mapping[item]

    def keys(self):
        return self._mapping.keys()


class ModeSplit:
    def __init__(self, mapping=None):
        if mapping is None:
            self._mapping = Dict[str, float]
        else:
            assert (isinstance(mapping, Dict))
            self._mapping = mapping

    def __setitem__(self, key, value):
        self._mapping[key] = value

    def __getitem__(self, item):
        return self._mapping[item]

    def keys(self):
        return self._mapping.keys()

class DemandUnit:
    def __init__(self, distance: float, demand: float, allocation=None, mode_split=None):
        if allocation is None:
            allocation = Allocation()
        if mode_split is None:
            mode_split = ModeSplit
        self.distance = distance
        self.demand = demand
        self.allocation = allocation
        self.mode_split = mode_split

    def __setitem__(self, key: Microtype, value: Dict[str, float]):
        self.allocation[key] = value

    def __getitem__(self, item: Microtype):
        return self.allocation[item]


class OD:
    def __init__(self, origin_microtype: Microtype, destination_microtype:Microtype, distance_bins=None):
        if distance_bins is None:
            distance_bins = []
        else:
            assert (isinstance(distance_bins, List[DemandUnit]))
        self.origin = origin_microtype
        self.destination = destination_microtype
        self.distance_bins = distance_bins

    def __setitem__(self, key, value: DemandUnit):
        self.distance_bins[key] = value

    def __len__(self):
        return len(self.distance_bins)

    def __getitem__(self, item):
        return self.distance_bins[item]

    def append(self, demand_unit: DemandUnit):
        self.distance_bins.append(demand_unit)

    def updateDemandInMicrotypes(self):
        for dist_bin in self.distance_bins:
            assert (isinstance(dist_bin, DemandUnit))
            mode_split = dist_bin.mode_split
            for microtype in dist_bin.allocation.keys():
                for mode in mode_split.keys():
                    microtype.addModeDemand(mode, dist_bin.demand * mode_split[mode])
