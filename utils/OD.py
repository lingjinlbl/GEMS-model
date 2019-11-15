import numpy as np
from utils.microtype import Microtype
from typing import Dict, List


class Allocation:
    def __init__(self, mapping=None):
        if mapping is None:
            self._mapping = Dict[Microtype, float]
        else:
            assert (isinstance(mapping, Dict))
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

class ChoiceCharacteristics:
    def __init__(self, travel_time=0., cost=0., wait_time=0.):
        self.travel_time = travel_time
        self.cost = cost
        self.wait_time = wait_time

    def __add__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

    def __iadd__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

class ModeCharacteristics:
    def __init__(self, modes: List[str]):
        self._modes = modes
        self.characteristics = dict()
        for mode in modes:
            self.characteristics[mode] = ChoiceCharacteristics()

    def __getitem__(self, item):
        if item in self._modes:
            return self.characteristics[item]

    def __setitem__(self, key, value):
        self.characteristics[key] = value

class DemandUnit:
    def __init__(self, distance: float, demand: float, allocation=None, mode_split=None):
        if allocation is None:
            allocation = Allocation()
        if mode_split is None:
            mode_split = ModeSplit({'car': 1.0})
        self.distance = distance
        self.demand = demand
        self.allocation = allocation
        self.mode_split = mode_split

    def __setitem__(self, key: Microtype, value: Dict[str, float]):
        self.allocation[key] = value

    def __getitem__(self, item: Microtype):
        return self.allocation[item]


class ODindex:
    def __init__(self, o: Microtype, d: Microtype, distBin: int):
        self.o = o
        self.d = d
        self.distBin = distBin

    def __eq__(self, other):
        if isinstance(other, ODindex):
            if (self.o == other.o) & (self.distBin == other.distBin) & (self.d == other.d):
                return True
            else:
                return False
        else:
            return False

    def __hash__(self):
        return hash((self.o, self.d, self.distBin))