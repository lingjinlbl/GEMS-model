import utils.OD as od
import numpy as np
from utils.microtype import Microtype
from utils.network import Network, NetworkCollection, NetworkFlowParams, Mode, ModeParams, BusMode, BusModeParams, Costs
from utils.geotype import Geotype, getModeSplit
import matplotlib.pyplot as plt

costs = {'car': Costs(0.0003778, 0., 1.0, 1.0), 'bus': Costs(0., 2.5, 0., 1.)}

network_params_mixed = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_params_car = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)
network_params_bus = NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50)

nc1 = NetworkCollection({Network(250, network_params_mixed): ['bus', 'car'], Network(750, network_params_car): ['car']},
                        {'car': ModeParams('car'), 'bus': BusModeParams(1.0)})

m1 = Microtype(nc1, costs)

distbins = {0: 1000.0, 1: 2000}
demandbydistbin = {0: 20 / 600., 1: 25 / 600.}
modesplitbydistbin = {0: od.ModeSplit({'car': 0.75, 'bus': 0.25}), 1: od.ModeSplit({'car': 0.75, 'bus': 0.25})}

g = Geotype(distbins=distbins) + m1

g.appendDemandData(od.ODindex(m1, m1, 0),
                   od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m1: 1.0}), modesplitbydistbin[0]))

g.appendDemandData(od.ODindex(m1, m1, 1),
                   od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m1: 1.0}), modesplitbydistbin[1]))

g.equilibriumModeChoice()

parking_costs = np.arange(0., 8, 0.1)
road_speeds = np.zeros((np.size(parking_costs)))
car_mode_share = np.zeros((np.size(parking_costs)))

fig, (ax1, ax2) = plt.subplots(1, 2)

for ii in range(np.size(parking_costs)):
    costs = {'car': Costs(0.0003778, parking_costs[ii], 1.0, 1.0), 'bus': Costs(0., 2.5, 0., 1.)}
    nc1 = NetworkCollection(
        {Network(250, network_params_mixed): ['bus', 'car'], Network(750, network_params_car): ['car']},
        {'car': ModeParams('car'), 'bus': BusModeParams(1.0)})

    m1 = Microtype(nc1, costs)

    distbins = {0: 1000.0, 1: 2000}
    demandbydistbin = {0: 20 / 600., 1: 25 / 600.}
    modesplitbydistbin = {0: od.ModeSplit({'car': 0.75, 'bus': 0.25}), 1: od.ModeSplit({'car': 0.75, 'bus': 0.25})}

    g = Geotype(distbins=distbins) + m1

    g.appendDemandData(od.ODindex(m1, m1, 0),
                       od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m1: 1.0}), modesplitbydistbin[0]))

    g.appendDemandData(od.ODindex(m1, m1, 1),
                       od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m1: 1.0}), modesplitbydistbin[1]))

    g.equilibriumModeChoice()
    road_speeds[ii] = m1.getModeSpeed('car')
    car_mode_share[ii] = g.getModeSplit('car')

ax1.plot(parking_costs, car_mode_share)
p_low = ax2.plot(parking_costs, road_speeds)


for ii in range(np.size(parking_costs)):
    costs = {'car': Costs(0.0003778, parking_costs[ii], 1.0, 1.0), 'bus': Costs(0., 2.5, 0., 1.)}
    nc1 = NetworkCollection(
        {Network(250, network_params_mixed): ['bus', 'car'], Network(750, network_params_car): ['car']},
        {'car': ModeParams('car'), 'bus': BusModeParams(2.0)})

    m1 = Microtype(nc1, costs)

    distbins = {0: 1000.0, 1: 2000}
    demandbydistbin = {0: 20 / 600., 1: 25 / 600.}
    modesplitbydistbin = {0: od.ModeSplit({'car': 0.75, 'bus': 0.25}), 1: od.ModeSplit({'car': 0.75, 'bus': 0.25})}

    g = Geotype(distbins=distbins) + m1

    g.appendDemandData(od.ODindex(m1, m1, 0),
                       od.DemandUnit(distbins[0], demandbydistbin[0], od.Allocation({m1: 1.0}), modesplitbydistbin[0]))

    g.appendDemandData(od.ODindex(m1, m1, 1),
                       od.DemandUnit(distbins[1], demandbydistbin[1], od.Allocation({m1: 1.0}), modesplitbydistbin[1]))

    g.equilibriumModeChoice()
    road_speeds[ii] = m1.getModeSpeed('car')
    car_mode_share[ii] = g.getModeSplit('car')

ax1.plot(parking_costs, car_mode_share)
ax1.set_xlabel('Parking Cost ($)')
ax1.set_ylabel('Car Mode Share')

p_high = ax2.plot(parking_costs, road_speeds)
ax2.set_xlabel('Parking Cost ($)')
ax2.set_ylabel('Road Speed (m/s)')

plt.legend((p_low[0], p_high[0]), ('Low Bus Service', 'High Bus Service'))
