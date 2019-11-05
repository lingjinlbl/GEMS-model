#!/usr/bin/env python3
import utils.IO as io
import numpy as np
import matplotlib.pyplot as plt

from utils.microtype import Microtype

network_params_default = io.Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
bus_params_default = io.BusParams(mean_trip_distance=1000, road_network_fraction=500, relative_length=3.0,
                                  fixed_density=85. / 100., min_stop_time=10., stop_spacing=1. / 250.,
                                  passenger_wait=4.)

car_params_default = io.ModeParams(mean_trip_distance=1000, relative_length=1.0)

modeCharacteristics = io.CollectedModeCharacteristics()
modeCharacteristics['car'] = io.ModeCharacteristics('car', car_params_default, demand=70 / (10 * 60))
modeCharacteristics['bus'] = io.ModeCharacteristics('bus', bus_params_default, demand=17 / (10 * 60))

m = Microtype(network_params_default, modeCharacteristics)

car_demands = np.arange(0.05, 0.12, 0.0015)
bus_demands = np.arange(0.001, 0.06, 0.001)

averageCosts = np.zeros((np.size(car_demands), np.size(bus_demands)))
flows = np.zeros((np.size(car_demands), np.size(bus_demands)))

for ii in range(np.size(car_demands)):
    for jj in range(np.size(bus_demands)):
        modeCharacteristics.setModeDemand('car', car_demands[ii])
        modeCharacteristics.setModeDemand('bus', bus_demands[jj])
        m = Microtype(network_params_default, modeCharacteristics)
        m.findEquilibriumDensityAndSpeed()
        flows[ii, jj] = np.sum(m.getFlows())
        averageCosts[ii, jj] = np.sum(m.getTotalTimes()) / np.sum(m.getFlows())
p1 = plt.contourf(bus_demands, car_demands, averageCosts, np.arange(0.08, 0.20, 0.01))
p2 = plt.contour(bus_demands, car_demands, flows, np.arange(45, 200, 10), cmap='Greys')
cb1 = plt.colorbar(p1)
cb2 = plt.colorbar(p2)

cb1.set_label('Social Cost per Passenger Trip')
cb2.set_label('Total Passenger Flow')

plt.xlabel('Bus Demand (trips / time)')
plt.ylabel('Car Demand (trips / time)')

g1 = np.gradient(averageCosts)
g2 = np.gradient(flows)

p3 = plt.contour(bus_demands, car_demands, g1[0] / g1[1] - g2[0] / g2[1], 0, linestyles='dashed', linewidths=2,
                 cmap='Reds')
"""
bus_params_default = {'k': 85. / 100.,
                      't_0': 10,
                      's_b': 1. / 250.,
                      'gamma_s': 4.,
                      'size': 3.0,
                      'meanTripDistance': 1000,
                      'L_mode': 500
                      }
mode_params_default = {'car': car_params_default, 'bus': bus_params_default}

totalDemand = 0.16
portions = np.arange(0.5, 1.0, 0.02)
oneDemandCosts = np.zeros(np.shape(portions))

for ii in range(np.size(portions)):
    demands = {'car': portions[ii] * totalDemand, 'bus': (1. - portions[ii]) * totalDemand}
    m2 = microtype.Microtype(modes, mode_params_default, network_params_default, demands)
    m2.findEquilibriumDensityAndSpeed()
    oneDemandCosts[ii] = np.sum(m2.getTotalTimes()) / np.sum(m2.getFlows())

plt.plot(portions, oneDemandCosts)


"""