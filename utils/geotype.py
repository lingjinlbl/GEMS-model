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
    def __init__(self, microtypes = None):
        if microtypes is None:
            self._microtypes = []
        else:
            self._microtypes = microtypes
        self.demand_structure = Dict[od.ODindex,od.DemandUnit]

    def appendMicrotype(self, microtype: Microtype):
        self._microtypes.append(microtype)

    def appendDemandData(self, odi:od.ODindex, demand:od.DemandUnit):
        self.demand_structure[odi] = demand

    def resetDemand(self):
        for microtype in self._microtypes:
            for mode in microtype.modes:
                microtype.setModeDemand(mode, 0.)
