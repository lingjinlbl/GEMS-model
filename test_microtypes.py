#!/usr/bin/env python3
import numpy as np
import microtype
import matplotlib.pyplot as plt

network_params_default = {'lambda': 0.068,
                          'u_f': 15.42,
                          'w': 1.88,
                          'kappa': 0.145,
                          'Q': 0.177,
                          'L': 1000,
                          'l': 50}
bus_params_default = {'k': 85. / 100.,
                      't_0': 10,
                      's_b': 1. / 250.,
                      'gamma_s': 4.,
                      'size': 3.0,
                      'meanTripDistance': 1000,
                      'L_mode': 500
                      }
car_params_default = {'meanTripDistance': 1000, 'size': 1.0}
modes = {'car', 'bus'}
mode_params_default = {'car': car_params_default, 'bus': bus_params_default}
demands = {'car': 20 / (10 * 60), 'bus': 1. / (10 * 60)}
m = microtype.Microtype(modes, mode_params_default, network_params_default, demands)
m.findEquilibriumDensityAndSpeed()

car_demands = np.arange(0.05, 0.12, 0.0015)
bus_demands = np.arange(0.001, 0.06, 0.001)

averageCosts = np.zeros((np.size(car_demands), np.size(bus_demands)))
flows = np.zeros((np.size(car_demands), np.size(bus_demands)))

for ii in range(np.size(car_demands)):
    for jj in range(np.size(bus_demands)):
        demands = {'car': car_demands[ii], 'bus': bus_demands[jj]}
        m2 = microtype.Microtype(modes, mode_params_default, network_params_default, demands)
        m2.findEquilibriumDensityAndSpeed()
        flows[ii, jj] = np.sum(m2.getFlows())
        averageCosts[ii, jj] = np.sum(m2.getTotalTimes()) / np.sum(m2.getFlows())
p1 = plt.contourf(bus_demands, car_demands,  averageCosts, np.arange(0.08,0.20,0.01))
p2 = plt.contour(bus_demands, car_demands, flows, np.arange(45,200,10), cmap='Greys')
cb1 = plt.colorbar(p1)
cb2 = plt.colorbar(p2)

cb1.set_label('Social Cost per Passenger Trip')
cb2.set_label('Total Passenger Flow')

plt.xlabel('Bus Demand (trips / time)')
plt.ylabel('Car Demand (trips / time)')
# %%
# import matplotlib.pyplot as plt
# plt.plot(supply.MFD(np.arange(1,100), 1000, network_params_default))


g1 = np.gradient(averageCosts)
g2 = np.gradient(flows)

p3 = plt.contour(bus_demands, car_demands,  g1[0]/g1[1] - g2[0]/g2[1],0,linestyles ='dashed', linewidths=2,cmap='Reds')