import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


class Interact:
    def __init__(self, model):
        self.__model = model
        self.__fig = go.FigureWidget()
        self.__modeToHandle = dict()
        self.__microtypeToHandle = dict()
        self.addBlankPlots(self.__fig)
        self.__microtypeToMixedNetworkID = dict()
        self.__microtypeToBusNetworkID = dict()
        self.__microtypeToBusService = dict()

        self.__grid = self.generateGridSpec()

    @property
    def model(self):
        return self.__model

    @property
    def data(self):
        return self.__fig.data[0]

    @property
    def layout(self):
        return self.__fig.layout

    @property
    def fig(self):
        return self.__fig

    @property
    def grid(self):
        return self.__grid

    def addBlankPlots(self, fig: go.FigureWidget):
        for mID in self.model.scenarioData['microtypeIDs'].MicrotypeID:
            fig.add_scatter(x=[],y=[],visible=True, name=mID)
            self.__microtypeToHandle[mID] = fig.data[-1]
        for mode in self.model.scenarioData['modeData'].keys():
            fig.add_scatter(x=[], y=[], visible=False, name=mode)
            self.__modeToHandle[mode] = fig.data[-1]

    def generateGridSpec(self):
        button = widgets.Button(description="Calculate Costs")
        button.on_click(self.updateCosts)

        gs = widgets.GridspecLayout(6, 4)

        gs[0, 0] = widgets.HTML(
            value="<center>Bus lane coverage</center>"
        )
        gs[0, 1] = widgets.HTML(
            value="<center>Bus headway </center>"
        )
        gs[0, 2] = widgets.HTML(
            value="<center>Bus route density </center>"
        )
        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            initialAutoData = self.model.scenarioData['subNetworkDataFull'].loc[
                              self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Auto') &
                              (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), :]
            self.__microtypeToMixedNetworkID[mID] = initialAutoData

            initialBusData = self.model.scenarioData['subNetworkDataFull'].loc[
                             self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Bus') &
                             (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID) &
                             self.model.scenarioData['subNetworkDataFull'].Dedicated, :]
            self.__microtypeToBusNetworkID[mID] = initialBusData

            busServiceData = self.model.scenarioData['modeData']['bus'].loc[mID]
            self.__microtypeToBusService = busServiceData

            gs[ind + 1, 0] = widgets.FloatSlider(value=0, min=0, max=1.0, step=0.02, description=mID)
            gs[ind + 1, 0].observe(self.response, names="value")

            gs[ind + 1, 1] = widgets.IntSlider(busServiceData.Headway, 90, 1800, 30, description=mID)
            gs[ind + 1, 1].observe(self.response, names="value")

            gs[ind + 1, 2] = widgets.FloatSlider(value=busServiceData.CoveragePortion, min=0.02, max=1.0, step=0.02,
                                                 description=mID)
            gs[ind + 1, 2].observe(self.response, names="value")
        gs[-1, :] = button
        return gs

    def response(self, change, otherStuff=None):
        print(change.owner)
        print(otherStuff)

    def returnBusLaneRanges(self):
        return self.model.scenarioData['subNetworkDataFull'].loc[
            self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Auto'), ['MicrotypeID',
                                                                                              'Length']].values

    def plotArray(self):
        x, y = self.model.plotAllDynamicStats("delay")
        fig, axs = plt.subplots(4, len(self.model.microtypes), figsize=(8., 6.), dpi=200)
        for ind, m in enumerate(self.model.microtypes):
            y1 = y[0, :, ind]
            y2 = y[1, :, ind]
            axs[0, ind].plot(x, y1, color="#800080")
            axs[0, ind].plot(x, y2, color="#00DBFF")
            axs[1, ind].plot(x, y1 - y2, color="#E56717")
            axs[2, ind].plot(x[:-1], np.interp(y1, y2, x)[:-1] / 60. - x[:-1] / 60., '#ED4337')
            axs[0, ind].set_title("Microtype " + m[0])

            axs[3, ind].clear()
            axs[3, ind].step(np.arange(len(self.model.timePeriods()) + 1), np.vstack(
                [self.model.getModeSplit('0', microtypeID=m[0])] + [self.model.getModeSplit(p, microtypeID=m[0]) for p
                                                                    in self.model.timePeriods().keys()]))
            axs[3, ind].set_ylim([0, 1])

            axs[3, ind].lines[self.model.modeToIdx['auto']].set_color('#C21807')
            axs[3, ind].lines[self.model.modeToIdx['bus']].set_color('#1338BE')
            axs[3, ind].lines[self.model.modeToIdx['walk']].set_color('#3CB043')
            axs[3, ind].lines[self.model.modeToIdx['rail']].set_color('orange')
            axs[3, ind].lines[self.model.modeToIdx['bike']].set_color('blue')

        axs[3, 0].legend(['bus', 'rail', 'walk', 'bike', 'auto'])
        axs[0, 0].set_ylabel('cumulative vehicles')
        axs[1, 0].set_ylabel('accumulation')
        axs[2, 0].set_ylabel('travel time')
        axs[3, 0].set_ylabel('mode split')

    def updateCosts(self, message=None):
        self.model.collectAllCosts()
        time, spds = self.model.plotAllDynamicStats('v')
        for ind, (mode, handle) in enumerate(self.__microtypeToHandle.items()):
            handle.y = spds[:, ind]
            handle.x = time
