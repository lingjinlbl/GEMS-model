import os

import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

from model import Model

colors = {'arrivals': '#800080',
          'departures': '#00DBFF',
          'accumulation': '#E56717',
          'autoTime': '#ED4337',
          'transitTime': '#3895D3',
          'walkTime': '#3DED97',
          'autoSplit': '#C21807',
          'transitSplit': '#1338BE',
          'walkSplit': '#3CB043',
          }


def gatherOutputs(model: Model):
    t_timePeriod = np.cumsum(np.array(list(chain(
        *[[0]] + [[dur / nSubBins] * nSubBins for dur in model.scenarioData['timePeriods'].DurationInHours]))))

    output = dict()
    vectorUserCosts, utils = model.collectAllCharacteristics()
    x, y = model.plotAllDynamicStats("delay")
    inflow = y[0, :, :]
    outflow = y[1, :, :]
    autoTravelTime = np.interp(np.squeeze(inflow), np.squeeze(outflow), x)[:-1] * 60. - x[:-1] * 60.

    output['inflow'] = inflow
    output['outflow'] = outflow
    output['time'] = x
    output['autoTravelTime'] = autoTravelTime

    x, y = model.plotAllDynamicStats("n")

    output['accumulation'] = y

    autoTravelTime_timePeriod = np.vstack([6436 / model.getModeSpeeds('0')['auto'] / 60] +
                                          [6436 / model.getModeSpeeds(p)['auto'] / 60 for p in
                                           model.timePeriods().keys()])

    busTravelTime_timePeriod = np.vstack([6436 / model.getModeSpeeds('0')['bus'] / 60] +
                                          [6436 / model.getModeSpeeds(p)['bus'] / 60 for p in
                                           model.timePeriods().keys()])

    walkTravelTime_timePeriod = np.vstack([6436 / model.getModeSpeeds('0')['walk'] / 60] +
                                          [6436 / model.getModeSpeeds(p)['walk'] / 60 for p in
                                           model.timePeriods().keys()])

    autoSpeed_timePeriod = np.vstack([model.getModeSpeeds('0')['auto']] +
                                     [model.getModeSpeeds(p)['auto'] for p in model.timePeriods().keys()])

    busSpeed_timePeriod = np.vstack([model.getModeSpeeds('0')['bus']] +
                                     [model.getModeSpeeds(p)['bus'] for p in model.timePeriods().keys()])

    walkSpeed_timePeriod = np.vstack([model.getModeSpeeds('0')['walk']] +
                                    [model.getModeSpeeds(p)['walk'] for p in model.timePeriods().keys()])

    output['t_timePeriod'] = t_timePeriod
    output['autoTravelTime_timePeriod'] = autoTravelTime_timePeriod
    output['autoSpeed_timePeriod'] = autoSpeed_timePeriod
    output['busTravelTime_timePeriod'] = busTravelTime_timePeriod
    output['busSpeed_timePeriod'] = busSpeed_timePeriod
    output['walkTravelTime_timePeriod'] = walkTravelTime_timePeriod
    output['walkSpeed_timePeriod'] = walkSpeed_timePeriod

    x, y = model.plotAllDynamicStats("v")
    output['autoSpeed'] = y

    output['utils'] = np.vstack([utils[:, :, 0, :].mean(axis=1)[0, :], utils[:, :, 0, :].mean(axis=1)])

    output['modeSplit'] = np.vstack([model.getModeSplit('0')] + [model.getModeSplit(p) for p in model.timePeriods().keys()])

    return output


nSubBins = 2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=nSubBins)
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -1000.0
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "walk", "BetaTravelTime"] = -1000.0
model.scenarioData['modeData']['bus'].Headway = 1e6
model.readFiles()


pops = [800, 1500, 1700]

scenarios = dict()

for ind, pop in enumerate(pops):
    scenario = 'pop-'+str(pop)+'-auto'

    model.scenarioData['populations'].Population = pop
    model.updatePopulation()

    scenarios[scenario] = gatherOutputs(model)

scenarioNames = ['pop-800-auto','pop-1500-auto','pop-1700-auto']
titles = ['Population 800', 'Population 1500', 'Population 1700']

figs = []
axs = []
for i in range(7):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(9,4))
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]

    fig = figs[0]
    ax = axs[0]

    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'], color=colors['departures'], label='Departures')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'], color=colors['arrivals'], label='Arrivals')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Cumulative vehicles')
    ax[ind].set_title(title)

    fig = figs[1]
    ax = axs[1]
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'], axis=0), color=colors['departures'],
             label='Departures')
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'], axis=0), color=colors['arrivals'],
             label='Arrivals')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Rate of change of vehicles')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    ax[ind].plot(scenarios[scenario]['inflow'][1:], np.diff(scenarios[scenario]['inflow'], axis=0), color=colors['departures'],
             label='Departures')
    ax[ind].plot(scenarios[scenario]['outflow'][1:], np.diff(scenarios[scenario]['outflow'], axis=0), color=colors['arrivals'],
             label='Arrivals')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Cumulative vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Rate of change of vehicles')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'], color=colors['accumulation'])
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)

    fig = figs[4]
    ax = axs[4]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'], color='C4', label='Averaged')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'], color='C5', label='Instantaneous')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Speed (m/s)')
    ax[ind].set_title(title)

    fig = figs[5]
    ax = axs[5]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoTravelTime_timePeriod'],
             color=colors['autoTime'], label='Averaged')
    ax[ind].plot(scenarios[scenario]['time'][1:], scenarios[scenario]['autoTravelTime'], color='C6', label='Instantaneous')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Travel time (min)')
    ax[ind].set_title(title)

    fig = figs[6]
    ax = axs[6]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['utils'][:, -1], color='C1', label='Averaged')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Utility')
    ax[ind].set_title(title)

groupName = 'validationForQR/1-microtype-compare-'

figs[0].savefig(groupName + 'queuing.pdf')
figs[1].savefig(groupName + 'inout-t.pdf')
figs[2].savefig(groupName + 'inout-n.pdf')
figs[3].savefig(groupName + 'accumulation.pdf')
figs[4].savefig(groupName + 'speed.pdf')
figs[5].savefig(groupName + 'travelTime.pdf')
figs[6].savefig(groupName + 'util.pdf')


model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=nSubBins)
model.readFiles()

pops = [2000, 2500, 3200]

for ind, pop in enumerate(pops):
    scenario = 'pop-'+str(pop)+'-all'

    model.scenarioData['populations'].Population = pop
    model.updatePopulation()

    scenarios[scenario] = gatherOutputs(model)

scenarioNames = ['pop-2000-all','pop-2500-all','pop-3200-all']
titles = ['Population 2000', 'Population 2500', 'Population 3200']

figs = []
axs = []
for i in range(8):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(9,4))
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]

    fig = figs[0]
    ax = axs[0]

    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'], color=colors['departures'], label='Departures')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'], color=colors['arrivals'], label='Arrivals')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Cumulative vehicles')
    ax[ind].set_title(title)

    fig = figs[1]
    ax = axs[1]
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'], axis=0), color=colors['departures'],
             label='Departures')
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'], axis=0), color=colors['arrivals'],
             label='Arrivals')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Rate of change of vehicles')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    ax[ind].plot(scenarios[scenario]['inflow'][1:], np.diff(scenarios[scenario]['inflow'], axis=0), color=colors['departures'],
             label='Departures')
    ax[ind].plot(scenarios[scenario]['outflow'][1:], np.diff(scenarios[scenario]['outflow'], axis=0), color=colors['arrivals'],
             label='Arrivals')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Cumulative vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Rate of change of vehicles')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'], color=colors['accumulation'])
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)

    fig = figs[4]
    ax = axs[4]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'], color='C4', label='Averaged auto')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['walkSpeed_timePeriod'], color='C2', label='Averaged walk')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busSpeed_timePeriod'], color='C3', label='Averaged bus')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'], color='C5', label='Instantaneous auto')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Speed (m/s)')
    ax[ind].set_title(title)

    fig = figs[5]
    ax = axs[5]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoTravelTime_timePeriod'],
             color=colors['autoTime'], label='Averaged')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busTravelTime_timePeriod'],
                 color=colors['transitTime'], label='Averaged')
    ax[ind].plot(scenarios[scenario]['time'][1:], scenarios[scenario]['autoTravelTime'], color='C6', label='Instantaneous')
    if ind == len(scenarios) - 1:
        ax[ind].legend()
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Travel time (min)')
    ax[ind].set_title(title)

    fig = figs[6]
    ax = axs[6]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['utils'],  label='Averaged')
    ax[ind].set_xlabel('Time (h)')
    if ind == len(scenarios) - 1:
        ax[ind].legend(['Bus','Walk','Auto'])
    if ind == 0:
        ax[ind].set_ylabel('Utility')
    ax[ind].set_title(title)

    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])

    fig = figs[7]
    ax = axs[7]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['modeSplit'])
    ax[ind].set_xlabel('Time (h)')
    if ind == len(scenarios) - 1:
        ax[ind].legend(['Bus', 'Walk', 'Auto'])
    if ind == 0:
        ax[ind].set_ylabel('Mode split')
    ax[ind].set_title(title)
    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])

groupName = 'validationForQR/1-microtype-modes-'

figs[0].savefig(groupName + 'queuing.pdf')
figs[1].savefig(groupName + 'inout-t.pdf')
figs[2].savefig(groupName + 'inout-n.pdf')
figs[3].savefig(groupName + 'accumulation.pdf')
figs[4].savefig(groupName + 'speed.pdf')
figs[5].savefig(groupName + 'travelTime.pdf')
figs[6].savefig(groupName + 'util.pdf')
figs[7].savefig(groupName + 'modeSplit.pdf')

print('done')


