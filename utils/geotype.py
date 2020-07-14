from typing import Dict, List

import numpy as np

import utils.OD as od
from utils.microtype import Microtype


def getModeSplit(mcc: od.ModeCharacteristics) -> od.ModeSplit:
    VOTT = 18 / 3600.
    utils = np.array([])
    k = 1.0
    modes = list(mcc.keys())
    for mode in modes:
        util = 0.
        util += -mcc[mode].travel_time * VOTT
        util += -mcc[mode].wait_time * VOTT
        util += -mcc[mode].cost
        utils = np.append(utils, util)
    exp_utils = np.exp(utils * k)
    probs = exp_utils / np.sum(exp_utils)
    mode_split = dict()
    for ind in range(np.size(probs)):
        mode_split[modes[ind]] = probs[ind]
    return od.ModeSplit(mode_split)


class Geotype:
    def __init__(self, distbins: Dict[int, float], microtypes=None):
        if microtypes is None:
            self._microtypes = []
        else:
            print('NOT READY YET')
            self._microtypes = List[Microtype]()
        self.demand_structure = dict()
        self.mode_choice_characteristics = dict()
        self.distbins = distbins

    def init_ODs(self):
        for m_o in self._microtypes:
            assert isinstance(m_o, Microtype)
            for m_d in self._microtypes:
                assert isinstance(m_d, Microtype)
                for distbin in self.distbins.keys():
                    odi = od.ODindex(m_o, m_d, distbin)
                    du_default = od.DemandUnit(distance=self.distbins[distbin], demand=0.0,
                                               allocation=od.Allocation({m_o: 0.5, m_d: 0.5}))
                    self.demand_structure[odi] = du_default
                    modes = list(set(m_o.mode_names).intersection(m_d.mode_names))
                    choice_characteristics_default = od.ModeCharacteristics(modes)
                    self.mode_choice_characteristics[odi] = choice_characteristics_default

    def appendMicrotype(self, microtype: Microtype):
        self._microtypes.append(microtype)

    def __iadd__(self, other):
        if isinstance(other, Microtype):
            self.appendMicrotype(other)
            return self
        else:
            print('BAD NEWS, BUDDY')
            return self

    def __add__(self, other):
        if isinstance(other, Microtype):
            self.appendMicrotype(other)
            return self
        else:
            print('BAD NEWS, BUDDY')
            return self

    def appendDemandData(self, odi: od.ODindex, demand: od.DemandUnit):
        self.demand_structure[odi] = demand

    def resetDemand(self):
        for microtype in self._microtypes:
            microtype._baseSpeed = microtype.network_params.getBaseSpeed()
            for mode in microtype.modes:
                microtype.setModeDemand(mode, 0., 1000.)

    def allocateDemandToMicrotypes(self):
        for mt in self._microtypes:
            assert (isinstance(mt, Microtype))
            mt.resetDemand()
        for odi in self.demand_structure.keys():
            du = self.demand_structure[odi]
            assert (isinstance(du, od.DemandUnit))
            assert (isinstance(odi, od.ODindex))
            for mode in du.mode_split.keys():
                odi.o.addModeStarts(mode, du.demand * du.mode_split[mode])
                odi.d.addModeEnds(mode, du.demand * du.mode_split[mode])
                for mt in du.allocation.keys():
                    assert (isinstance(mt, Microtype))
                    mt.addModeDemandForPMT(mode, du.demand * du.mode_split[mode] * du.allocation[mt],
                                           self.distbins[odi.distBin])

    def updateMicrotypeModeCharacteristics(self, iter_max=50):
        for mt in self._microtypes:
            assert isinstance(mt, Microtype)
            mt.updateNetworkSpeeds(iter_max)

    def updateChoiceCharacteristics(self):
        for odi in self.demand_structure.keys():
            du = self.demand_structure[odi]
            assert isinstance(du, od.DemandUnit)
            self.mode_choice_characteristics[odi] = du.getChoiceCharacteristics()

    def updateModeSplit(self):
        for odi in self.demand_structure.keys():
            du = self.demand_structure[odi]
            assert isinstance(du, od.DemandUnit)
            # print('-----')
            # print(du.mode_split)
            du.updateModeSplit(getModeSplit(self.mode_choice_characteristics[odi]))
            # print(du.mode_split)

    def equilibriumModeChoice(self, n_iters=20):
        for iter in range(n_iters):
            self.allocateDemandToMicrotypes()
            self.updateMicrotypeModeCharacteristics()
            self.updateChoiceCharacteristics()
            self.updateModeSplit()

    def getModeSplit(self, mode):
        total_demand = 0.0
        mode_demand = 0.0
        for odi in self.demand_structure.keys():
            du = self.demand_structure[odi]
            assert isinstance(du, od.DemandUnit)
            total_demand += du.demand * du.distance
            mode_demand += du.demand * du.distance * du.mode_split[mode]
        return mode_demand / total_demand

    def __str__(self):
        return 'Speeds: ' + str([str(mt._baseSpeed) + ' ,' for mt in self._microtypes])
