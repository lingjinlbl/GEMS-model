#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from utils.Network import Network
from utils.microtype import Microtype, CollectedModeCharacteristics, ModeCharacteristics
from utils.supply import BusParams, ModeParams
import scipy.ndimage as sp

network_params_default = Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = BusParams(road_network_fraction=500, relative_length=3.0,
                                  fixed_density=150. / 100., min_stop_time=15., stop_spacing=1. / 500.,
                                  passenger_wait=5.)

car_params_default = ModeParams(relative_length=1.0)

modeCharacteristics = CollectedModeCharacteristics()
modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default)
modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params_default)

m = Microtype(network_params_default, modeCharacteristics)
m.setModeDemand('car', 70 / (10 * 60), 1000.0)
m.setModeDemand('bus', 10 / (10 * 60), 1000.0)

total_demands = np.arange(0.02, 0.2, 0.005)
mode_splits = np.arange(0.3, 1.0, 0.05)

average_costs = np.zeros((np.size(total_demands), np.size(mode_splits)))
flows = np.zeros((np.size(total_demands), np.size(mode_splits)))
car_speeds = np.zeros((np.size(total_demands), np.size(mode_splits)))

for ii in range(np.size(total_demands)):
    for jj in range(np.size(mode_splits)):
        car_demand = total_demands[ii] * mode_splits[jj]
        bus_demand = total_demands[ii] * (1.0 - mode_splits[jj])
        m = Microtype(network_params_default, modeCharacteristics)
        m.setModeDemand('car', car_demand, 1000.0)
        m.setModeDemand('bus', bus_demand, 1000.0)
        m.findEquilibriumDensityAndSpeed()
        flows[ii, jj] = np.sum(m.getFlows())
        car_speeds[ii, jj] = m.getModeSpeed('car')
        average_costs[ii, jj] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

fig1 = plt.figure(figsize=(8, 5))

p1 = plt.contourf(mode_splits, total_demands, average_costs)  # , np.arange(0.08, 0.35, 0.02))
#p2 = plt.contour(bus_demands, car_demands, flows, np.arange(45, 200, 10), cmap='Greys')
cb1 = plt.colorbar(p1)
#cb2 = plt.colorbar(p2)

cb1.set_label('Average Travel Speed (m/s)')
#cb2.set_label('Total Passenger Flow')

plt.ylabel('Total Demand (trip starts / time)')
plt.xlabel('Car Mode Share')

g1 = np.gradient(average_costs)

slope = g1[1] / g1[0]

p3 = plt.contour(mode_splits, total_demands, slope, 0, linestyles='dashed', linewidths=2,
                 cmap='Reds')




totalDemand = 0.18
portions = np.arange(0.55, 0.7, 0.005)
oneDemandCosts = np.zeros(np.shape(portions))


bus_params = BusParams(road_network_fraction=500, relative_length=3.0,
                                  fixed_density=100. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)

modeCharacteristics = CollectedModeCharacteristics()
modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default)
modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params)

for ii in range(np.size(portions)):
    m = Microtype(network_params_default, modeCharacteristics)
    m.setModeDemand('car',  portions[ii] * totalDemand, 1000.0)
    m.setModeDemand('bus', (1. - portions[ii]) * totalDemand, 1000.0)
    m.findEquilibriumDensityAndSpeed()
    oneDemandCosts[ii] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

totalDemand = 0.18
portions = np.arange(0.55, 0.7, 0.005)
oneDemandCosts2 = np.zeros(np.shape(portions))

bus_params = BusParams(road_network_fraction=500, relative_length=3.0,
                                  fixed_density=85. / 100., min_stop_time=15., stop_spacing=1. / 250.,
                                  passenger_wait=5.)
modeCharacteristics = CollectedModeCharacteristics()
modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default, demand=70 / (10 * 60))
modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params, demand=17 / (10 * 60))

for ii in range(np.size(portions)):
    m = Microtype(network_params_default, modeCharacteristics)
    m.setModeDemand('car',  portions[ii] * totalDemand, 1000.0)
    m.setModeDemand('bus', (1. - portions[ii]) * totalDemand, 1000.0)
    m.findEquilibriumDensityAndSpeed()
    oneDemandCosts2[ii] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())

fig2 = plt.figure(figsize=(7, 4))
plt.plot(portions, oneDemandCosts, label = "More buses")
plt.plot(portions, oneDemandCosts2, label = "Fewer buses")

plt.legend()
plt.xlabel('Portion of trips by car')
plt.ylabel('Average Travel Speed')