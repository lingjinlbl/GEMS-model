import numpy as np
import utils.supply as supply
import utils.IO as io
import copy
import utils.IO as io
import utils.OD as od
from utils.IO import CollectedModeCharacteristics

from utils.microtype import Microtype
from typing import Dict, List

class Geotype:
    def __init__(self, distbins:Dict[int, float], microtypes = None):
        if microtypes is None:
            self._microtypes = []
        else:
            print('NOT READY YET')
            self._microtypes = List[Microtype]()
        self.demand_structure = dict()
        self.distbins = distbins

    def init_ODs(self):
        for m_o in self._microtypes:
            for m_d in self._microtypes:
                for distbin in self.distbins.keys():
                    odi = od.ODindex(m_o, m_d, distbin)
                    du_default = od.DemandUnit(distance=self.distbins[distbin], demand=0.0)
                    self.demand_structure[odi] = du_default


    def appendMicrotype(self, microtype: Microtype):
        self._microtypes.append(microtype)

    def appendDemandData(self, odi:od.ODindex, demand:od.DemandUnit):
        self.demand_structure[odi] = demand

    def resetDemand(self):
        for microtype in self._microtypes:
            for mode in microtype.modes:
                microtype.setModeDemand(mode, 0.)
