import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model, Optimizer

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

colorsModes = ['tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plotCosts(series: pd.Series, xLabel=''):
    f = plt.figure(figsize=(10, 4))
    ax = f.add_subplot(121)
    fig = plt.bar(series.columns, series.iloc[0, :].values, color=colors)
    plt.xticks(rotation=45)
    plt.ylabel('Social Cost')
    plt.title('Base cost contribution')
    ax2 = f.add_subplot(122)
    lines = [
        plt.plot(series.index, series.iloc[:, i] - series.iloc[0, i], color=colors[i], label=series.columns[i]) for
        i in range(series.shape[1])]
    total = plt.plot(series.index, series.sum(axis=1).values - series.iloc[0, :].sum(), color='k', linewidth=2.0,
                     label="Total")
    plt.legend(list(series.columns) + ['Total'])
    plt.xlabel(xLabel)
    plt.ylabel('Change in social cost')
    plt.title('Variation')
    f.tight_layout()
    return f


def plotModes(series: pd.Series, xLabel=''):
    f = plt.figure(figsize=(10, 4))
    ax = f.add_subplot(121)
    fig = plt.bar(series.columns, series.iloc[0, :].values, color=colorsModes)
    plt.xticks(rotation=45)
    plt.ylabel('Mode split')
    plt.title('Base mode split')
    ax2 = f.add_subplot(122)
    lines = [
        plt.plot(series.index, series.iloc[:, i] - series.iloc[0, i], color=colorsModes[i], label=series.columns[i]) for
        i in
        range(series.shape[1])]
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel("Change in mode split")
    plt.title('Variation')
    plt.tight_layout()
    return f


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=2)
optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'Bus')],
                      fromToSubNetworkIDs=[('A', 'Bus')],
                      method="min")
#
# model.data.updateMicrotypeNetworkLength('1', 0.71)
# model.data.updateMicrotypeNetworkLength('2', 0.71)
# model.data.updateMicrotypeNetworkLength('3', 0.95)
# model.data.updateMicrotypeNetworkLength('4', 0.71)
# model.data.updateMicrotypeNetworkLength('5', 0.55)
# model.data.updateMicrotypeNetworkLength('6', 0.6)

busAllocations = np.linspace(0., 0.6, 20)
collectedCosts = dict()
collectedModeSplits = dict()
for ba in busAllocations:
    optimizer.updateAndRunModel(np.array([ba, 180]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[ba] = allCosts.sum()
        collectedModeSplits[ba] = model.getModeSplit(microtypeID='A')

varyBusAllocation = pd.DataFrame(collectedCosts).transpose()
varyBusAllocationModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_bus = plotModes(varyBusAllocationModes, "Bus ROW allocation")
f1_bus = plotCosts(varyBusAllocation, "Bus ROW allocation")

busHeadway = np.linspace(60, 240, 20)
collectedCosts = dict()
collectedModeSplits = dict()
for bh in busHeadway:
    optimizer.updateAndRunModel(np.array([0., bh]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[bh] = allCosts.sum()
        collectedModeSplits[bh] = model.getModeSplit(microtypeID='A')

varyBusHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_headway = plotModes(varyBusHeadwayModes, "Bus headway")
f1_headway = plotCosts(varyBusHeadway, "Bus headway")

busCosts = np.linspace(0, 5., 20)
collectedCosts = dict()
collectedModeSplits = dict()
for bc in busCosts:
    model.interact.modifyModel(changeType=('fare', ('A', 'Bus')), value=bc)
    model.interact.modifyModel(changeType=('fareSenior', ('A', 'Bus')), value=bc / 2.0)
    optimizer.updateAndRunModel(np.array([0., 120.]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[bc] = allCosts.sum()
        collectedModeSplits[bc] = model.getModeSplit(microtypeID='A')

varyBusCost = pd.DataFrame(collectedCosts).transpose()
varyBusCostModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_buscost = plotModes(varyBusCostModes, "Bus fare")
f1_buscost = plotCosts(varyBusCost, "Bus fare")

collectedCosts = dict()
collectedModeSplits = dict()
for bh in busHeadway:
    for ba in busAllocations:
        optimizer.updateAndRunModel(np.array([ba, bh]))
        allCosts = optimizer.sumAllCosts()
        if optimizer.model.successful:
            collectedCosts[(ba, bh)] = allCosts.sum()
            collectedModeSplits[(ba, bh)] = model.getModeSplit(microtypeID='A')
varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()
f3 = plt.figure(figsize=(5, 4))
f3_compare = plt.contourf(busHeadway, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
plt.xlabel('Bus headway')
plt.ylabel('Bus ROW allocation')
cbar = plt.colorbar()
cbar.set_label("Total costs")
params, outcomes, mask = optimizer.plotConvergence()
plt.plot(params[:, 1] * 1000., params[:, 0], 'ok-')

""" 
'eps': 0.001, 'eta': 0.05
     fun: 11107.89474150491
     jac: array([-0.11045,  3.52667])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 276
     nit: 5
  status: 1
 success: True
       x: array([0.38612, 0.09783])

'eps': 0.001, 'eta': 0.1
     fun: 11107.894748087037
     jac: array([-0.48857,  3.65627])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 273
     nit: 5
  status: 1
 success: True
       x: array([0.38584, 0.09783])
       
       
'eps': 0.001, 'eta': 0.3,
     fun: 11236.963157226093
     jac: array([ 0.03367, 22.05223])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 249
     nit: 7
  status: 1
 success: True
       x: array([0.28059, 0.10105])
       
'eps': 0.001, 'eta': 0.5,
    fun: 11107.887189299972
     jac: array([-1.40775, 23.3317 ])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 207
     nit: 10
  status: 1
 success: True
       x: array([0.38501, 0.09795])
"""

print('done')

model = Model(ROOT_DIR + "/../input-data", nSubBins=2)
optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'Bus')],
                      fromToSubNetworkIDs=[('A', 'Bus'), ('A', 'Bike')],
                      method="min")
#
# model.data.updateMicrotypeNetworkLength('1', 0.71)
# model.data.updateMicrotypeNetworkLength('2', 0.71)
# model.data.updateMicrotypeNetworkLength('3', 0.95)
# model.data.updateMicrotypeNetworkLength('4', 0.71)
# model.data.updateMicrotypeNetworkLength('5', 0.55)
# model.data.updateMicrotypeNetworkLength('6', 0.6)

busAllocations = np.linspace(0., 0.5, 20)
collectedCosts = dict()
collectedModeSplits = dict()
for ba in busAllocations:
    optimizer.updateAndRunModel(np.array([ba, 0., 120]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[ba] = allCosts.sum()
        collectedModeSplits[ba] = model.getModeSplit(microtypeID='A')

varyBusAllocation = pd.DataFrame(collectedCosts).transpose()
varyBusAllocationModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_bus = plotModes(varyBusAllocationModes, "Bus ROW allocation")
f1_bus = plotCosts(varyBusAllocation, "Bus ROW allocation")

busHeadway = np.linspace(60, 300, 20)
collectedCosts = dict()
collectedModeSplits = dict()
for bh in busHeadway:
    optimizer.updateAndRunModel(np.array([0., 0., bh]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[bh] = allCosts.sum()
        collectedModeSplits[bh] = model.getModeSplit(microtypeID='A')
        model.getModeSplit()

varyBusHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_headway = plotModes(varyBusHeadwayModes, "Bus headway")
f1_headway = plotCosts(varyBusHeadway, "Bus headway")

bikeAllocations = np.linspace(0, 0.12, 20)
collectedCosts = dict()
collectedModeSplits = dict()
for ba in bikeAllocations:
    optimizer.updateAndRunModel(np.array([0., ba, 300]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[ba] = allCosts.sum()
        collectedModeSplits[ba] = model.getModeSplit(microtypeID='A')

varyBikeAllocation = pd.DataFrame(collectedCosts).transpose()
varyBikeAllocationModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_bike = plotModes(varyBikeAllocationModes, "Bike ROW allocation")
f1_bike = plotCosts(varyBikeAllocation, "Bike ROW allocation")

busCosts = np.linspace(0, 5., 20)
collectedCosts = dict()
collectedModeSplits = dict()
for bc in busCosts:
    model.interact.modifyModel(changeType=('fare', ('A', 'Bus')), value=bc)
    model.interact.modifyModel(changeType=('fareSenior', ('A', 'Bus')), value=bc / 2.0)
    optimizer.updateAndRunModel(np.array([0., 0., 83.]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[bc] = allCosts.sum()
        collectedModeSplits[bc] = model.getModeSplit(microtypeID='A')

varyBusCost = pd.DataFrame(collectedCosts).transpose()
varyBusCostModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_buscost = plotModes(varyBusCostModes, "Bus fare")
f1_buscost = plotCosts(varyBusCost, "Bus fare")

collectedCosts = dict()
collectedModeSplits = dict()
for bh in busHeadway:
    for ba in busAllocations:
        optimizer.updateAndRunModel(np.array([ba, 0., bh]))
        allCosts = optimizer.sumAllCosts()
        if optimizer.model.successful:
            collectedCosts[(ba, bh)] = allCosts.sum()
            collectedModeSplits[(ba, bh)] = model.getModeSplit(microtypeID='A')
varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()
f3_compare = plt.contourf(busHeadway, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
plt.xlabel('Bus headway')
plt.ylabel('Bus ROW allocation')
cbar = plt.colorbar()
cbar.set_label("Total costs")

collectedCosts = dict()
collectedModeSplits = dict()
for bi in bikeAllocations:
    for ba in busAllocations:
        optimizer.updateAndRunModel(np.array([ba, bi, 94.17]))
        allCosts = optimizer.sumAllCosts()
        if optimizer.model.successful:
            collectedCosts[(ba, bi)] = allCosts.sum()
            collectedModeSplits[(ba, bi)] = model.getModeSplit(microtypeID='A')
varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()
f4_compare = plt.contourf(bikeAllocations, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
plt.xlabel('Bike ROW allocation')
plt.ylabel('Bus ROW allocation')
cbar = plt.colorbar()
cbar.set_label("Total costs")

"""
     fun: 1415800.9342278235
     jac: array([ -5386.56424,  36265.10123, -10130.32122])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 396
     nit: 4
  status: 1
 success: True
       x: array([0.00618, 0.02831, 0.09417])
       
reduce step size to 0.005:
     fun: 1415785.1056187206
     jac: array([-5492.58945, -8468.22157, -3182.56466])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 300
     nit: 4
  status: 1
 success: True
       x: array([0.00664, 0.02826, 0.09639])
       
       
     fun: 1407224.8228403067
     jac: array([-5.27984e+03,  6.71734e+04,  1.37346e+01,  4.40610e+04,
        2.78833e+03, -4.41726e+03])
 message: 'Converged (|f_n-f_(n-1)| ~= 0)'
    nfev: 1302
     nit: 15
  status: 1
 success: True
       x: array([0.0495 , 0.03067, 0.     , 0.01491, 0.09642, 0.12636])
"""

print('done')

model = Model(ROOT_DIR + "/../input-data-losangeles-national-params", nSubBins=1)
optimizer = Optimizer(model, modesAndMicrotypes=[('1', 'Bus')],
                      fromToSubNetworkIDs=[('1', 'Bus'), ('1', 'Bike')],
                      method="min")
#
# model.data.updateMicrotypeNetworkLength('1', 0.71)
# model.data.updateMicrotypeNetworkLength('2', 0.71)
# model.data.updateMicrotypeNetworkLength('3', 0.95)
# model.data.updateMicrotypeNetworkLength('4', 0.71)
# model.data.updateMicrotypeNetworkLength('5', 0.55)
# model.data.updateMicrotypeNetworkLength('6', 0.6)

busAllocations = np.linspace(0., 0.5, 20)
collectedCosts = dict()
collectedModeSplits = dict()
for ba in busAllocations:
    optimizer.updateAndRunModel(np.array([ba, 0., 120]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[ba] = allCosts.sum()
        collectedModeSplits[ba] = model.getModeSplit(microtypeID='1')

varyBusAllocation = pd.DataFrame(collectedCosts).transpose()
varyBusAllocationModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_bus = plotModes(varyBusAllocationModes, "Bus ROW allocation")
f1_bus = plotCosts(varyBusAllocation, "Bus ROW allocation")

busHeadway = np.linspace(60, 900, 12)
collectedCosts = dict()
collectedModeSplits = dict()
for bh in busHeadway:
    optimizer.updateAndRunModel(np.array([0., 0., bh]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[bh] = allCosts.sum()
        collectedModeSplits[bh] = model.getModeSplit(microtypeID='1')
        model.getModeSplit()

varyBusHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_headway = plotModes(varyBusHeadwayModes, "Bus headway")
f1_headway = plotCosts(varyBusHeadway, "Bus headway")

bikeAllocations = np.linspace(0, 0.12, 20)
collectedCosts = dict()
collectedModeSplits = dict()
for ba in bikeAllocations:
    optimizer.updateAndRunModel(np.array([0., ba, 300]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[ba] = allCosts.sum()
        collectedModeSplits[ba] = model.getModeSplit(microtypeID='1')

varyBikeAllocation = pd.DataFrame(collectedCosts).transpose()
varyBikeAllocationModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_bike = plotModes(varyBikeAllocationModes, "Bike ROW allocation")
f1_bike = plotCosts(varyBikeAllocation, "Bike ROW allocation")

busCosts = np.linspace(0, 5., 20)
collectedCosts = dict()
collectedModeSplits = dict()
for bc in busCosts:
    model.interact.modifyModel(changeType=('fare', ('1', 'Bus')), value=bc)
    model.interact.modifyModel(changeType=('fareSenior', ('1', 'Bus')), value=bc / 2.0)
    optimizer.updateAndRunModel(np.array([0., 0., 83.]))
    allCosts = optimizer.sumAllCosts()
    if optimizer.model.successful:
        collectedCosts[bc] = allCosts.sum()
        collectedModeSplits[bc] = model.getModeSplit(microtypeID='1')

varyBusCost = pd.DataFrame(collectedCosts).transpose()
varyBusCostModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()

f2_buscost = plotModes(varyBusCostModes, "Bus fare")
f1_buscost = plotCosts(varyBusCost, "Bus fare")

collectedCosts = dict()
collectedModeSplits = dict()
for bh in busHeadway:
    for ba in busAllocations:
        optimizer.updateAndRunModel(np.array([ba, 0., bh]))
        allCosts = optimizer.sumAllCosts()
        if optimizer.model.successful:
            collectedCosts[(ba, bh)] = allCosts.sum()
            collectedModeSplits[(ba, bh)] = model.getModeSplit(microtypeID='1')
varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()
f3_compare = plt.contourf(busHeadway, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
plt.xlabel('Bus headway')
plt.ylabel('Bus ROW allocation')
cbar = plt.colorbar()
cbar.set_label("Total costs")

collectedCosts = dict()
collectedModeSplits = dict()
for bi in bikeAllocations:
    for ba in busAllocations:
        optimizer.updateAndRunModel(np.array([ba, bi, 94.17]))
        allCosts = optimizer.sumAllCosts()
        if optimizer.model.successful:
            collectedCosts[(ba, bi)] = allCosts.sum()
            collectedModeSplits[(ba, bi)] = model.getModeSplit(microtypeID='1')
varyBusBike = pd.DataFrame(collectedCosts).transpose()
varyBusBikeModes = pd.DataFrame(collectedModeSplits, index=model.modeToIdx.keys()).transpose()
f4_compare = plt.contourf(bikeAllocations, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
plt.xlabel('Bike ROW allocation')
plt.ylabel('Bus ROW allocation')
cbar = plt.colorbar()
cbar.set_label("Total costs")
