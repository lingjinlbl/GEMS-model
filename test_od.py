import utils.IO as io
import utils.OD as od

from utils.microtype import Microtype, CollectedModeCharacteristics, ModeCharacteristics, Costs
from utils.supply import ModeParams, BusParams
from utils.Network import Network
from utils.geotype import Geotype

costs = {'car': Costs(0.0003778, 0., 10., 1.0), 'bus': Costs(0., 2.5, 0., 1.)}

network_params = Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = BusParams(road_network_fraction=500, relative_length=3.0,
                               fixed_density=95. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                               passenger_wait=5.)

car_params_default = ModeParams(relative_length=1.0)

modeCharacteristics = ModeCharacteristics('car', car_params_default) + ModeCharacteristics('bus', bus_params_default)

m = Microtype(network_params, modeCharacteristics, costs)

network_params2 = Network(0.075, 14.42, 1.88, 0.145, 0.177, 600, 50)

m2 = Microtype(network_params2, modeCharacteristics, costs)

distbins = {0: 1000.0, 1: 2000, 2: 1000}
demandbydistbin = {0: 30 / 600., 1: 15 / 600., 2: 50 / 600.}
modesplitbydistbin = {0: od.ModeSplit({'car': 0.75, 'bus': 0.25}), 1: od.ModeSplit({'car': 1.0}),
                      2: od.ModeSplit({'car': 0.75, 'bus': 0.25})}

g = Geotype(distbins=distbins) + m + m2

g.appendDemandData(od.ODindex(m, m, 0),
                   od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m: 1.0}), modesplitbydistbin[0]))
g.appendDemandData(od.ODindex(m, m, 1),
                   od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m: 1.0}), modesplitbydistbin[1]))
g.appendDemandData(od.ODindex(m, m2, 2),
                   od.DemandUnit(distbins[2], demandbydistbin[2], od.Allocation({m: 0.15, m2: 0.85}),
                                 modesplitbydistbin[2]))

g.allocateDemandToMicrotypes()
g.updateMicrotypeModeCharacteristics()
g.updateChoiceCharacteristics()
