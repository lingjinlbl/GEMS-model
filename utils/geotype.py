import numpy as np
import utils.supply as supply
import utils.IO as io
import copy
import utils.IO as io
import utils.OD as od
from utils.microtype import Microtype, CollectedModeCharacteristics, ModeCharacteristics

from typing import Dict, List


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
                    du_default = od.DemandUnit(distance=self.distbins[distbin], demand=0.0)
                    self.demand_structure[odi] = du_default
                    modes = list(set(m_o.modes).intersection(m_d.modes))
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
            for mode in microtype.modes:
                microtype.setModeDemand(mode, 0.)

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

    def updateMicrotypeModeCharacteristics(self, iter_max=20):
        for mt in self._microtypes:
            assert isinstance(mt, Microtype)
            mt.findEquilibriumDensityAndSpeed(iter_max)
