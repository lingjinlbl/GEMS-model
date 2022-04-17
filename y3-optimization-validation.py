import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Model, Optimizer

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']

colorsModes = ['tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'xkcd:sky blue']


def plotCosts(series: pd.Series, folder, suffix, xLabel=''):
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
    plt.savefig(folder + '/costs' + suffix + '.png')
    return f


def plotModes(series: pd.Series, folder, suffix, xLabel=''):
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
    plt.savefig(folder + '/modes' + suffix + '.png')
    return f


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# model = Model(ROOT_DIR + "/input-data-simpler", nSubBins=2)
# optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'Bus')],
#                       fromToSubNetworkIDs=[('A', 'Bus')],
#                       method="min")
#
# initialDemand = model.data.tripRate().copy()
#
# multipliers = np.linspace(0.8, 1.2, 11)
# #
# # model.data.updateMicrotypeNetworkLength('1', 0.71)
# # model.data.updateMicrotypeNetworkLength('2', 0.71)
# # model.data.updateMicrotypeNetworkLength('3', 0.95)
# # model.data.updateMicrotypeNetworkLength('4', 0.71)
# # model.data.updateMicrotypeNetworkLength('5', 0.55)
# # model.data.updateMicrotypeNetworkLength('6', 0.6)
#
# busAllocations = np.linspace(0., 0.6, 10)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for ba in busAllocations:
#         optimizer.updateAndRunModel(np.array([ba, 180]))
#         allCosts = optimizer.sumAllCosts()
#         if optimizer.model.successful:
#             collectedCosts[ba] = allCosts.sum()
#             collectedModeSplits[ba] = model.getModeSplit(microtypeID='A')
#
#     varyBusAllocation = pd.DataFrame(collectedCosts).transpose()
#     varyBusAllocationModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
#     f2_bus = plotModes(varyBusAllocationModes, "out-optimization/1-microtype", "-busROW-demand" + f'{mul:.2f}',
#                        "Bus ROW allocation")
#     f1_bus = plotCosts(varyBusAllocation, "out-optimization/1-microtype", "-busROW-demand" + f'{mul:.2f}',
#                        "Bus ROW allocation")
#
# multipliers = np.linspace(0.5, 1.2, 11)
# busHeadway = np.linspace(60, 240, 11)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for bh in busHeadway:
#         optimizer.updateAndRunModel(np.array([0., bh]))
#         allCosts = optimizer.sumAllCosts()
#         if optimizer.model.successful:
#             collectedCosts[bh] = allCosts.sum()
#             collectedModeSplits[bh] = model.getModeSplit(microtypeID='A')
#
#     varyBusHeadway = pd.DataFrame(collectedCosts).transpose()
#     varyBusHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
#     f2_bus = plotModes(varyBusHeadwayModes, "out-optimization/1-microtype", "-busHeadway-demand" + f'{mul:.2f}',
#                        "Bus headway")
#     f1_bus = plotCosts(varyBusHeadway, "out-optimization/1-microtype", "-busHeadway-demand" + f'{mul:.2f}',
#                        "Bus headway")
#
# multipliers = np.linspace(0.5, 1.2, 11)
# busCosts = np.linspace(0, 5., 20)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for bc in busCosts:
#         model.interact.modifyModel(changeType=('fare', ('A', 'Bus')), value=bc)
#         model.interact.modifyModel(changeType=('fareSenior', ('A', 'Bus')), value=bc / 2.0)
#         optimizer.updateAndRunModel(np.array([0., 120.]))
#         allCosts = optimizer.sumAllCosts()
#         if optimizer.model.successful:
#             collectedCosts[bc] = allCosts.sum()
#             collectedModeSplits[bc] = model.getModeSplit(microtypeID='A')
#
#     varyBusCost = pd.DataFrame(collectedCosts).transpose()
#     varyBusCostModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
#     f2_buscost = plotModes(varyBusCostModes, "out-optimization/1-microtype", "-busCost-demand" + f'{mul:.2f}',
#                            "Bus fare")
#     f1_buscost = plotCosts(varyBusCost, "out-optimization/1-microtype", "-busCost-demand" + f'{mul:.2f}', "Bus fare")
#
# multipliers = np.linspace(0.8, 1.12, 7)
# busAllocations = np.linspace(0., 0.6, 12)
# busHeadway = np.linspace(60, 240, 12)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for bh in busHeadway:
#         for ba in busAllocations:
#             optimizer.updateAndRunModel(np.array([ba, bh]))
#             allCosts = optimizer.sumAllCosts()
#             if optimizer.model.successful:
#                 collectedCosts[(ba, bh)] = allCosts.sum()
#                 collectedModeSplits[(ba, bh)] = model.getModeSplit(microtypeID='A')
#     varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
#     varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#     f3 = plt.figure(figsize=(5, 4))
#     f3_compare = plt.contourf(busHeadway, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
#     plt.xlabel('Bus headway')
#     plt.ylabel('Bus ROW allocation')
#     cbar = plt.colorbar()
#     cbar.set_label("Total costs")
#     plt.savefig("out-optimization/1-microtype/headway-row-cost-demand" + f'{mul:.2f}' + ".png")
#     # params, outcomes, mask = optimizer.plotConvergence()
#     # plt.plot(params[:, 1] * 1000., params[:, 0], 'ok-')
#
# """
# 'eps': 0.001, 'eta': 0.05
#      fun: 11107.89474150491
#      jac: array([-0.11045,  3.52667])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 276
#      nit: 5
#   status: 1
#  success: True
#        x: array([0.38612, 0.09783])
#
# 'eps': 0.001, 'eta': 0.1
#      fun: 11107.894748087037
#      jac: array([-0.48857,  3.65627])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 273
#      nit: 5
#   status: 1
#  success: True
#        x: array([0.38584, 0.09783])
#
#
# 'eps': 0.001, 'eta': 0.3,
#      fun: 11236.963157226093
#      jac: array([ 0.03367, 22.05223])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 249
#      nit: 7
#   status: 1
#  success: True
#        x: array([0.28059, 0.10105])
#
# 'eps': 0.001, 'eta': 0.5,
#     fun: 11107.887189299972
#      jac: array([-1.40775, 23.3317 ])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 207
#      nit: 10
#   status: 1
#  success: True
#        x: array([0.38501, 0.09795])
# """
#
# print('done')

model = Model(ROOT_DIR + "/input-data-california-A", nSubBins=1)
optimizer = Optimizer(model, modesAndMicrotypes=[('1', 'Bus')],
                      fromToSubNetworkIDs=[('1', 'Bus'), ('1', 'Bike')],
                      method="min")

initialDemand = model.data.tripRate().copy()

multipliers = np.linspace(0.6, 1.2, 7)
# # #
# # # model.data.updateMicrotypeNetworkLength('1', 0.71)
# # # model.data.updateMicrotypeNetworkLength('2', 0.71)
# # # model.data.updateMicrotypeNetworkLength('3', 0.95)
# # # model.data.updateMicrotypeNetworkLength('4', 0.71)
# # # model.data.updateMicrotypeNetworkLength('5', 0.55)
# # # model.data.updateMicrotypeNetworkLength('6', 0.6)
# #
# busAllocations = np.linspace(0., 0.5, 30)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for ba in busAllocations:
#         optimizer.updateAndRunModel(np.array([ba, 0., 400]))
#         allCosts = optimizer.sumAllCosts()
#         if optimizer.model.successful:
#             collectedCosts[ba] = allCosts.sum()
#             collectedModeSplits[ba] = model.getModeSplit(microtypeID='1')
#
#     varyBusAllocation = pd.DataFrame(collectedCosts).transpose()
#     varyBusAllocationModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
#     f2_bus = plotModes(varyBusAllocationModes, "out-optimization/CA", "-busROW-demand" + f'{mul:.2f}',
#                        "Bus ROW allocation")
#     f1_bus = plotCosts(varyBusAllocation, "out-optimization/CA", "-busROW-demand" + f'{mul:.2f}',
#                        "Bus ROW allocation")
#
# busHeadway = np.linspace(60, 300, 30)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for bh in busHeadway:
#         optimizer.updateAndRunModel(np.array([0., 0., bh]))
#         allCosts = optimizer.sumAllCosts()
#         if optimizer.model.successful:
#             collectedCosts[bh] = allCosts.sum()
#             collectedModeSplits[bh] = model.getModeSplit(microtypeID='1')
#             model.getModeSplit()
#
#     varyBusHeadway = pd.DataFrame(collectedCosts).transpose()
#     varyBusHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
#     f2_bus = plotModes(varyBusHeadwayModes, "out-optimization/CA", "-busHeadway-demand" + f'{mul:.2f}',
#                        "Bus headway")
#     f1_bus = plotCosts(varyBusHeadway, "out-optimization/CA", "-busHeadway-demand" + f'{mul:.2f}',
#                        "Bus headway")
#
# bikeAllocations = np.linspace(0, 0.12, 30)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for ba in bikeAllocations:
#         optimizer.updateAndRunModel(np.array([0., ba, 400]))
#         allCosts = optimizer.sumAllCosts()
#         if optimizer.model.successful:
#             collectedCosts[ba] = allCosts.sum()
#             collectedModeSplits[ba] = model.getModeSplit(microtypeID='1')
#
#     varyBikeAllocation = pd.DataFrame(collectedCosts).transpose()
#     varyBikeAllocationModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
#     f2_bus = plotModes(varyBikeAllocationModes, "out-optimization/CA", "-bikeROW-demand" + f'{mul:.2f}',
#                        "Bike ROW allocation")
#     f1_bus = plotCosts(varyBikeAllocation, "out-optimization/CA", "-bikeROW-demand" + f'{mul:.2f}',
#                        "Bike ROW allocation")

busCosts = np.linspace(0, 5., 30)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for bc in busCosts:
#     model.interact.modifyModel(changeType=('fare', ('A', 'Bus')), value=bc)
#     model.interact.modifyModel(changeType=('fareSenior', ('A', 'Bus')), value=bc / 2.0)
#     optimizer.updateAndRunModel(np.array([0., 0., 83.]))
#     allCosts = optimizer.sumAllCosts()
#     if optimizer.model.successful:
#         collectedCosts[bc] = allCosts.sum()
#         collectedModeSplits[bc] = model.getModeSplit(microtypeID='A')
#
# varyBusCost = pd.DataFrame(collectedCosts).transpose()
# varyBusCostModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
# f2_buscost = plotModes(varyBusCostModes, "out-optimization/4-microtype", "-busFare", "Bus fare")
# f1_buscost = plotCosts(varyBusCost, "out-optimization/4-microtype", "-busFare", "Bus fare")
#
# multipliers = np.linspace(0.8, 1.05, 7)
busAllocations = np.linspace(0., 0.5, 20)
busHeadway = np.linspace(60, 300, 20)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for bh in busHeadway:
#         for ba in busAllocations:
#             optimizer.updateAndRunModel(np.array([ba, 0.028, bh]))
#             allCosts = optimizer.sumAllCosts()
#             if optimizer.model.successful:
#                 collectedCosts[(ba, bh)] = allCosts.sum()
#                 collectedModeSplits[(ba, bh)] = model.getModeSplit(microtypeID='A')
#             else:
#                 collectedCosts[(ba, bh)] = np.nan
#                 collectedModeSplits[(ba, bh)] = np.nan
#     varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
#     varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#     plt.figure()
#     f3_compare = plt.contourf(busHeadway, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
#     plt.xlabel('Bus headway')
#     plt.ylabel('Bus ROW allocation')
#     cbar = plt.colorbar()
#     cbar.set_label("Total costs")
#     plt.savefig("out-optimization/4-microtype/headway-row-cost-demand" + f'{mul:.2f}' + ".png")
# # plt.scatter([94.17], [0.00618])
#
# multipliers = np.linspace(0.9, 1.05, 5)
# bikeAllocations = np.linspace(0, 0.10, 20)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for mul in multipliers:
#     model.data.updateTripRate(initialDemand * mul)
#     for bi in bikeAllocations:
#         for ba in busAllocations:
#             optimizer.updateAndRunModel(np.array([ba, bi, 94.17]))
#             allCosts = optimizer.sumAllCosts()
#             if optimizer.model.successful:
#                 collectedCosts[(ba, bi)] = allCosts.sum()
#                 collectedModeSplits[(ba, bi)] = model.getModeSplit(microtypeID='A')
#             else:
#                 collectedCosts[(ba, bi)] = np.nan
#                 collectedModeSplits[(ba, bi)] = np.nan
#     varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
#     varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#     plt.figure()
#     f4_compare = plt.contourf(bikeAllocations, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
#     plt.xlabel('Bike ROW allocation')
# #     plt.ylabel('Bus ROW allocation')
# #     cbar = plt.colorbar()
# #     cbar.set_label("Total costs")
# #     plt.savefig("out-optimization/4-microtype/bikerow-busrow-cost-demand" + f'{mul:.2f}' + ".png")
# # # plt.scatter([0.0283], [0.00618])
# #
# # busHeadway = np.linspace(60, 300, 20)
# # bikeAllocations = np.linspace(0, 0.1, 20)
# # collectedCosts = dict()
# # collectedModeSplits = dict()
# # for bi in bikeAllocations:
# #     for bh in busHeadway:
# #         optimizer.updateAndRunModel(np.array([0.0065, 0.028, bh]))
# #         allCosts = optimizer.sumAllCosts()
# #         if optimizer.model.successful:
# #             collectedCosts[(bi, bh)] = allCosts.sum()
# #             collectedModeSplits[(bi, bh)] = model.getModeSplit(microtypeID='A')
# #         else:
# #             collectedCosts[(bi, bh)] = np.nan
# #             collectedModeSplits[(bi, bh)] = np.nan
# # varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
# # varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
# # f4_compare = plt.contourf(bikeAllocations, busHeadway, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
# # plt.xlabel('Bike ROW allocation')
# # plt.ylabel('Bus headway')
# # cbar = plt.colorbar()
# # cbar.set_label("Total costs")
# # plt.scatter([0.0283], [94.17])
#
# """
#      fun: 1415800.9342278235
#      jac: array([ -5386.56424,  36265.10123, -10130.32122])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 396
#      nit: 4
#   status: 1
#  success: True
#        x: array([0.00618, 0.02831, 0.09417])
#
# reduce step size to 0.005:
#      fun: 1415785.1056187206
#      jac: array([-5492.58945, -8468.22157, -3182.56466])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 300
#      nit: 4
#   status: 1
#  success: True
#        x: array([0.00664, 0.02826, 0.09639])
#
#
#      fun: 1407224.8228403067
#      jac: array([-5.27984e+03,  6.71734e+04,  1.37346e+01,  4.40610e+04,
#         2.78833e+03, -4.41726e+03])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 1302
#      nit: 15
#   status: 1
#  success: True
#        x: array([0.0495 , 0.03067, 0.     , 0.01491, 0.09642, 0.12636])
# """
#
# print('done')
# #
# # model = Model(ROOT_DIR + "/input-data-california-A", nSubBins=1)
# # optimizer = Optimizer(model, modesAndMicrotypes=[('1', 'Bus')],
# #                       fromToSubNetworkIDs=[('1', 'Bus'), ('1', 'Bike')],
# #                       method="min")
# # #
# # model.data.updateMicrotypeNetworkLength('1', 0.71)
# # model.data.updateMicrotypeNetworkLength('2', 0.71)
# # model.data.updateMicrotypeNetworkLength('3', 0.95)
# # model.data.updateMicrotypeNetworkLength('4', 0.71)
# # model.data.updateMicrotypeNetworkLength('5', 0.55)
# # model.data.updateMicrotypeNetworkLength('6', 0.6)
#
# busAllocations = np.linspace(0., 0.3, 20)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for ba in busAllocations:
#     optimizer.updateAndRunModel(np.array([ba, 0., 443]))
#     allCosts = optimizer.sumAllCosts()
#     if optimizer.model.successful:
#         collectedCosts[ba] = allCosts.sum()
#         collectedModeSplits[ba] = model.getModeSplit(microtypeID='1')
#     else:
#         collectedCosts[ba] = np.nan
#         collectedModeSplits[ba] = np.nan
#
# varyBusAllocation = pd.DataFrame(collectedCosts).transpose()
# varyBusAllocationModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
# f2_bus = plotModes(varyBusAllocationModes, "out-optimization/CA", "-busRow", "Bus ROW allocation")
# f1_bus = plotCosts(varyBusAllocation, "out-optimization/CA", "-busRow", "Bus ROW allocation")
#
# busHeadway = np.linspace(240, 900, 20)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for bh in busHeadway:
#     optimizer.updateAndRunModel(np.array([0., 0., bh]))
#     allCosts = optimizer.sumAllCosts()
#     if optimizer.model.successful:
#         collectedCosts[bh] = allCosts.sum()
#         collectedModeSplits[bh] = model.getModeSplit(microtypeID='1')
#         model.getModeSplit()
#     else:
#         collectedCosts[bh] = np.nan
#         collectedModeSplits[bh] = np.nan
#
# varyBusHeadway = pd.DataFrame(collectedCosts).transpose()
# varyBusHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
# f2_headway = plotModes(varyBusHeadwayModes, "out-optimization/CA", "-busHeadway", "Bus headway")
# f1_headway = plotCosts(varyBusHeadway, "out-optimization/CA", "-busHeadway", "Bus headway")
#
# bikeAllocations = np.linspace(0, 0.08, 20)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for ba in bikeAllocations:
#     optimizer.updateAndRunModel(np.array([0., ba, 443]))
#     allCosts = optimizer.sumAllCosts()
#     if optimizer.model.successful:
#         collectedCosts[ba] = allCosts.sum()
#         collectedModeSplits[ba] = model.getModeSplit(microtypeID='1')
#     else:
#         collectedCosts[ba] = np.nan
#         collectedModeSplits[ba] = np.nan
#
# varyBikeAllocation = pd.DataFrame(collectedCosts).transpose()
# varyBikeAllocationModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
# f2_bike = plotModes(varyBikeAllocationModes, "out-optimization/CA", "-bikeROW", "Bike ROW allocation")
# f1_bike = plotCosts(varyBikeAllocation, "out-optimization/CA", "-bikeROW", "Bike ROW allocation")
#
# busCosts = np.linspace(0, 5., 20)
# collectedCosts = dict()
# collectedModeSplits = dict()
# for bc in busCosts:
#     model.interact.modifyModel(changeType=('fare', ('1', 'Bus')), value=bc)
#     model.interact.modifyModel(changeType=('fareSenior', ('1', 'Bus')), value=bc / 2.0)
#     optimizer.updateAndRunModel(np.array([0., 0., 443.]))
#     allCosts = optimizer.sumAllCosts()
#     if optimizer.model.successful:
#         collectedCosts[bc] = allCosts.sum()
#         collectedModeSplits[bc] = model.getModeSplit(microtypeID='1')
#     else:
#         collectedCosts[bc] = np.nan
#         collectedModeSplits[bc] = np.nan
#
# varyBusCost = pd.DataFrame(collectedCosts).transpose()
# varyBusCostModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
#
# f2_buscost = plotModes(varyBusCostModes, "out-optimization/CA", "-busFare", "Bus fare")
# f1_buscost = plotCosts(varyBusCost, "out-optimization/CA", "-busFare", "Bus fare")

busHeadway = np.linspace(240, 900, 20)
busAllocations = np.linspace(0., 0.3, 20)

collectedCosts = dict()
collectedModeSplits = dict()
for bh in busHeadway:
    for ba in busAllocations:
        optimizer.updateAndRunModel(np.array([ba, 0., bh]))
        allCosts = optimizer.sumAllCosts()
        if optimizer.model.successful:
            collectedCosts[(ba, bh)] = allCosts.sum()
            collectedModeSplits[(ba, bh)] = model.getModeSplit(microtypeID='1')
        else:
            collectedCosts[(ba, bh)] = np.nan
            collectedModeSplits[(ba, bh)] = np.nan
varyBusCostHeadway = pd.DataFrame(collectedCosts).transpose()
varyBusCostHeadwayModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
plt.figure()
f3_compare = plt.contourf(busHeadway, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
plt.xlabel('Bus headway')
plt.ylabel('Bus ROW allocation')
cbar = plt.colorbar()
cbar.set_label("Total costs")
plt.savefig("out-optimization/CA/bus-headway-vs-allocation.png")

collectedCosts = dict()
collectedModeSplits = dict()
for bi in bikeAllocations:
    for ba in busAllocations:
        optimizer.updateAndRunModel(np.array([ba, bi, 443.]))
        allCosts = optimizer.sumAllCosts()
        if optimizer.model.successful:
            collectedCosts[(ba, bi)] = allCosts.sum()
            collectedModeSplits[(ba, bi)] = model.getModeSplit(microtypeID='1')
        else:
            collectedCosts[(ba, bi)] = np.nan
            collectedModeSplits[(ba, bi)] = np.nan
varyBusBike = pd.DataFrame(collectedCosts).transpose()
varyBusBikeModes = pd.DataFrame(collectedModeSplits, index=model.passengerModeToIdx.keys()).transpose()
plt.figure()
f4_compare = plt.contourf(bikeAllocations, busAllocations, varyBusCostHeadway.sum(axis=1).unstack().values, 30)
plt.xlabel('Bike ROW allocation')
plt.ylabel('Bus ROW allocation')
cbar = plt.colorbar()
cbar.set_label("Total costs")
plt.savefig("out-optimization/CA/bike-vs-bus.png")
