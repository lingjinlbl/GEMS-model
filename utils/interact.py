import base64
import os
from math import floor, log10
from zipfile import ZipFile

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express.colors as col
import plotly.graph_objects as go
from ipywidgets import Layout
from plotly.subplots import make_subplots


class Interact:
    def __init__(self, model, figure=False):
        self.__model = model
        self.__optimizer = model.emptyOptimizer()
        self.__colors = self.generateColorDict()
        self.__paramNames = self.generateParamDict()
        self.__showFigure = figure
        if figure:
            self.__fig = widgets.Accordion(children=[])  # widgets.VBox([])
        else:
            self.__fig = None
        self.__modeToHandle = dict()
        self.__dataToHandle = dict()
        if figure:
            self.addBlankPlots(self.__fig)
            self.copyCurrentToRef()
        self.__microtypeToMixedNetworkID = dict()
        self.__microtypeToBusNetworkID = dict()
        self.__microtypeToBikeNetworkID = dict()
        self.__microtypeToBusService = dict()
        self.__widgetIDtoSubNetwork = dict()
        self.__widgetIDtoField = dict()
        self.__utilDropdownStatus = dict()
        self.__widgetIDtoUtil = dict()
        self.__plotStateWidget = None
        self.__loadingWidget = None
        self.__downloadWidget = None
        self.__generateWidget = None
        self.__downloadHTML = None
        self.__generateHTML = None
        self.__out = None  # print(*a, file = sys.stdout)
        if figure:
            self.__grid = self.generateGridSpec()

    def generateParamDict(self):
        out = {'Constant': 'intercept',
               'Wait time': 'wait_time',
               'In vehicle time': 'travel_time',
               'Access time': 'access_time'}
        return out

    def generateColorDict(self):
        modes = col.qualitative.Bold
        microtypes = col.qualitative.Safe
        costs = col.qualitative.Vivid
        groups = col.qualitative.Plotly
        out = {'bus': modes[0],
               'auto': modes[1],
               'walk': modes[5],
               'bike': modes[3],
               'rail': modes[4],
               'A': microtypes[0],
               'B': microtypes[1],
               'C': microtypes[2],
               'D': microtypes[3],
               '1': microtypes[0],
               '2': microtypes[1],
               '3': microtypes[2],
               '4': microtypes[3],
               '5': microtypes[4],
               '6': microtypes[5],
               '7': microtypes[6],
               '8': microtypes[7],
               'user': costs[1],
               'system': costs[1],
               'dedication': costs[1],
               'HighIncVeh': groups[0],
               'HighIncNoVeh': groups[1],
               'HighIncVehSenior': groups[2],
               'HighIncNoVehSenior': groups[3],
               'LowIncVeh': groups[4],
               'LowIncNoVeh': groups[5],
               'LowIncVehSenior': groups[6],
               'LowIncNoVehSenior': groups[7],
               'low-income-senior': groups[0],
               'high-income-senior': groups[1],
               'low-income': groups[3],
               'high-income': groups[2]}
        return out

    @property
    def colors(self):
        return self.__colors

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

    @property
    def out(self):
        return self.__out

    def addBlankPlots(self, figContainer: widgets.Accordion):
        # fig = go.FigureWidget(
        #     make_subplots(rows=7, cols=2, shared_yaxes=False, column_titles=['Current', 'Reference']))
        # figContainer.children = [fig]
        #
        # fig.update_layout(
        #     autosize=False,
        #     width=900,
        #     height=1500, )
        microtypeFigs = []
        diffFigs = []

        for mID in self.model.scenarioData['microtypeIDs'].MicrotypeID:
            currentMicrotypeFig = go.FigureWidget(
                make_subplots(rows=5, cols=2,
                              shared_yaxes=True,
                              column_titles=['Current', 'Reference'])
            )

            currentMicrotypeFig.update_layout(
                {'autosize': False, 'width': 900, 'height': 1200, 'template': 'simple_white'})

            currentMicrotypeFig['layout']['xaxis']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis']['title'] = 'Mode speed (mi/hr)'
            currentMicrotypeFig['layout']['xaxis2']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['xaxis3']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis3']['title'] = 'Mode split'
            currentMicrotypeFig['layout']['xaxis4']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis5']['title'] = 'Auto speed (mi/hr)'
            currentMicrotypeFig['layout']['xaxis5']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['xaxis6']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis7']['title'] = 'Contribution to social cost'
            currentMicrotypeFig['layout']['yaxis9']['title'] = 'Accessibility measure'

            microtypeFigs.append(currentMicrotypeFig)

            currentDiffFig = go.FigureWidget(
                make_subplots(rows=3, cols=1,
                              shared_yaxes=True)
            )

            currentDiffFig['layout']['yaxis']['title'] = 'Difference in mode split'
            currentDiffFig['layout']['yaxis2']['title'] = 'Difference in auto speed (m/s)'
            currentDiffFig['layout']['xaxis2']['title'] = 'Time (hr)'
            currentDiffFig['layout']['yaxis3']['title'] = 'Accessibility measure'

            currentDiffFig.update_layout(
                {'autosize': False, 'width': 900, 'height': 800, 'template': 'simple_white'})

            diffFigs.append(currentDiffFig)

        bothCostFigs = go.FigureWidget(
            make_subplots(rows=1, cols=2,
                          shared_yaxes=True,
                          column_titles=['Current', 'Reference'])
        )

        bothCostFigs['layout']['yaxis']['title'] = 'Contribution to Social Cost'
        bothCostFigs['layout']['xaxis']['title'] = 'Type'
        bothCostFigs['layout']['xaxis2']['title'] = 'Type'

        bothCostFigs.update_layout(
            {'autosize': False, 'width': 900, 'height': 400, 'template': 'simple_white'})

        combinedCostDiffFig = go.FigureWidget(go.Figure())
        combinedCostDiffFig['layout']['yaxis']['title'] = 'Difference in social cost'
        combinedCostDiffFig.update_layout(template='simple_white')

        self.__dataToHandle['speed'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['modeSplit'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['modeSpeed'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['cost'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['accessibility'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['costCombined'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['costDiff'] = dict()
        self.__dataToHandle['modeSplitDiff'] = dict()
        self.__dataToHandle['speedDiff'] = dict()
        self.__dataToHandle['accessibilityDiff'] = dict()
        for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=3, col=1,
                                           showlegend=False,
                                           mode='lines')
            self.__dataToHandle['speed']['current'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].line = {"color": self.colors[mID]}
            microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=3, col=2,
                                           mode='lines', showlegend=False)
            self.__dataToHandle['speed']['ref'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].line = {"color": self.colors[mID]}
            diffFigs[idx].add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=2, col=1,
                                      legendgroup="Speed",
                                      mode='lines', showlegend=False)
            self.__dataToHandle['speedDiff'][mID] = diffFigs[idx].data[-1]
            diffFigs[idx].data[-1].line = {"color": self.colors[mID]}
        for mode in self.model.scenarioData['modeData'].keys():
            self.__dataToHandle['modeSplit']['current'][mode] = dict()
            self.__dataToHandle['modeSplit']['ref'][mode] = dict()
            self.__dataToHandle['modeSplitDiff'][mode] = dict()
            for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=2, col=1, legendgroup=mode,
                                               mode='lines', showlegend=False)
                self.__dataToHandle['modeSplit']['current'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'vh', "color": self.colors[mode]}
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=2, col=2, legendgroup=mode,
                                               mode='lines',
                                               showlegend=False)
                self.__dataToHandle['modeSplit']['ref'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'vh', "color": self.colors[mode]}
                diffFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=1, col=1, legendgroup=mode,
                                          mode='lines',
                                          showlegend=True)
                self.__dataToHandle['modeSplitDiff'][mode][mID] = diffFigs[idx].data[-1]
                diffFigs[idx].data[-1].line = {"shape": 'vh', "color": self.colors[mode]}
        for mode in self.model.scenarioData['modeData'].keys():
            self.__dataToHandle['modeSpeed']['current'][mode] = dict()
            self.__dataToHandle['modeSpeed']['ref'][mode] = dict()
            for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=1, col=1, mode='lines',
                                               hovertext="Microtype " + mID + " " + mode, hoverinfo="text",
                                               showlegend=True, legendgroup=mode)
                self.__dataToHandle['modeSpeed']['current'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'vh', "color": self.colors[mode]}
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=1, col=2, mode='lines',
                                               showlegend=False, hovertext="Microtype " + mID + " " + mode,
                                               hoverinfo="text", legendgroup=mode)
                self.__dataToHandle['modeSpeed']['ref'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'vh', "color": self.colors[mode]}
        for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            """ COSTS """
            microtypeFigs[idx].add_bar(x=['User', 'Freight', 'Transit', 'Revenue', 'Externality', 'Lane dedication'],
                                       y=[0.] * 4,
                                       visible=True, row=4, col=1, name='Microtype ' + mID, legendgroup="Costs",
                                       showlegend=False)
            self.__dataToHandle['cost']['current'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].marker.color = self.colors[mID]
            microtypeFigs[idx].add_bar(x=['User', 'Freight', 'Transit', 'Revenue', 'Externality', 'Lane dedication'],
                                       y=[0.] * 4,
                                       visible=True, row=4, col=2, name='Microtype ' + mID, legendgroup="Costs",
                                       showlegend=False)
            self.__dataToHandle['cost']['ref'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].marker.color = self.colors[mID]
            combinedCostDiffFig.add_bar(x=['User', 'Freight', 'Transit', 'Revenue', 'Externality', 'Lane dedication'],
                                        y=[0.] * 4,
                                        visible=True, name='Microtype ' + mID, showlegend=True)
            self.__dataToHandle['costDiff'][mID] = combinedCostDiffFig.data[-1]
            combinedCostDiffFig.data[-1].marker.color = self.colors[mID]
        purposes = list(set(self.model.scenarioData.tripPurposeToIdx.keys()).difference({'home', 'work'}))
        for di in self.model.diToIdx.keys():
            homeMicrotype = di.homeMicrotype
            populationGroup = di.populationGroupType
            idx = self.model.microtypeIdToIdx[homeMicrotype]
            if (homeMicrotype, populationGroup) not in self.__dataToHandle['accessibility']['current']:
                microtypeFigs[idx].add_bar(
                    x=purposes,
                    y=[0.] * len(purposes),
                    visible=True, row=5, col=1,
                    name=populationGroup,
                    legendgroup="Groups",
                    showlegend=False)
                self.__dataToHandle['accessibility']['current'][(homeMicrotype, populationGroup)] = \
                    microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].marker.color = self.colors[populationGroup]
                microtypeFigs[idx].add_bar(
                    x=purposes,
                    y=[0.] * len(purposes),
                    visible=True, row=5, col=2, name=populationGroup, legendgroup="Groups",
                    showlegend=True)
                self.__dataToHandle['accessibility']['ref'][(homeMicrotype, populationGroup)] = microtypeFigs[idx].data[
                    -1]
                microtypeFigs[idx].data[-1].marker.color = self.colors[populationGroup]
                diffFigs[idx].add_bar(
                    x=purposes,
                    y=[0.] * len(purposes),
                    visible=True, row=3, col=1, name=populationGroup, legendgroup="Groups",
                    showlegend=True)
                self.__dataToHandle['accessibilityDiff'][(homeMicrotype, populationGroup)] = diffFigs[idx].data[
                    -1]
                diffFigs[idx].data[-1].marker.color = self.colors[populationGroup]

        for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            bothCostFigs.add_bar(x=['User', 'Freight', 'Transit', 'Revenue', 'Externality', 'Lane dedication'],
                                 y=[0.] * 4,
                                 visible=True,
                                 name='Microtype ' + mID, showlegend=False, row=1, col=1)
            self.__dataToHandle['costCombined']['current'][mID] = bothCostFigs.data[-1]
            bothCostFigs.data[-1].marker.color = self.colors[mID]
            bothCostFigs.add_bar(x=['User', 'Freight', 'Transit', 'Revenue', 'Externality', 'Lane dedication'],
                                 y=[0.] * 4,
                                 visible=True,
                                 name='Microtype ' + mID, showlegend=True, row=1, col=2)
            self.__dataToHandle['costCombined']['ref'][mID] = bothCostFigs.data[-1]
            bothCostFigs.data[-1].marker.color = self.colors[mID]

        figContainer.children = microtypeFigs + [bothCostFigs] + diffFigs + [combinedCostDiffFig]

        tabTitles = ["Microtype " + mID + ": Outcomes" for mID in self.model.microtypeIdToIdx.keys()] + ["Costs"] + [
            "Microtype " + mID + ": Change from reference" for mID in self.model.microtypeIdToIdx.keys()] + [
                        "Change in costs"]

        for ind, title in enumerate(tabTitles):
            figContainer.set_title(ind, title)

    def generateGridSpec(self):
        rerunModel = widgets.Button(description="Calculate Costs",
                                    tooltip="Click to run the model with your given inputs",
                                    layout=Layout(width='95%', height='0.5in', justify_content='center'))
        rerunModel.on_click(self.updateCosts)

        setRef = widgets.Button(description="Update reference",
                                tooltip='Click to update reference plots on right',
                                layout=Layout(width='95%', height='0.5in'))
        setRef.on_click(self.copyCurrentToRef)

        self.__downloadHTML = '''
        <html>
        <body>
        <a download="output-results.zip" href="data:text/csv;base64,{payload}" download>
        <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-info">Download File</button>
        </a>
        </body>
        </html>
        '''

        html_button = self.__downloadHTML.format(payload="")

        downloadButton = widgets.HTML(html_button, layout=Layout(width='95%', height='0.5in'))

        generateButton = widgets.Button(description="Create download link", layout=Layout(width='95%', height='0.5in'))
        generateButton.on_click(self.createDownloadLink)

        self.__downloadWidget = downloadButton

        self.__generateWidget = generateButton

        networkLengthStack = []
        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            microtypeText = widgets.HTML(
                value="<center><b>Microtype " + mID + "</b></center>"
            )
            slider = widgets.FloatSlider(value=1.0, min=0.2, max=2.0, step=0.01, orientation='horizontal')
            slider.observe(self.response, names="value")
            self.__widgetIDtoField[slider.model_id] = ('networkLength', mID)
            networkLengthStack.append(widgets.HBox([microtypeText, slider]))

        populationStack = []
        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            microtypePopulations = [widgets.HTML(
                value="<center><b>Microtype " + mID + "</b></center>"
            )]
            sub = self.model.scenarioData['populations'].loc[self.model.scenarioData['populations'].MicrotypeID == mID,
                  :]
            upperBound = sub.Population.max() * 1.5
            upperBound = round(upperBound, 3 - int(floor(log10(
                abs(
                    upperBound)))) - 1)  # https://www.kite.com/python/answers/how-to-round-a-number-to-significant-digits-in-python
            popVBox = []
            for row in sub.itertuples():
                popVBox.append(widgets.IntSlider(row.Population, 0, upperBound, upperBound / 100,
                                                 description=row.PopulationGroupTypeID,
                                                 orientation='horizontal'))
                popVBox[-1].observe(self.response, names="value")
                self.__widgetIDtoField[popVBox[-1].model_id] = ('population', (mID, row.PopulationGroupTypeID))
            microtypePopulations.append(widgets.VBox(popVBox))
            populationStack.append(widgets.HBox(microtypePopulations))

        utilStack = []
        allTripTypes = self.model.scenarioData['populationGroups'].TripPurposeID.unique()
        modes = list(self.model.modeToIdx.keys())
        for groupID in self.model.scenarioData['populationGroups'].PopulationGroupTypeID.unique():
            self.__utilDropdownStatus[groupID] = dict()
            label = widgets.HTML(
                value="<center><b>Population group</b><br>" + groupID + "</center>"
            )
            tripTypeDropdown = widgets.Dropdown(
                options=allTripTypes,
                value=allTripTypes[0],
                description='Trip purpose:',
                disabled=False,
                layout={'vertical_align': 'middle', 'width': '2.8in'},
                style={'description_width': '1.25in'}
            )
            self.__utilDropdownStatus[groupID]['tripType'] = tripTypeDropdown
            self.__widgetIDtoUtil[tripTypeDropdown.model_id] = ('tripType', groupID)
            tripTypeDropdown.observe(self.changeUtilDropdown)
            modeDropdown = widgets.Dropdown(
                options=modes,
                value=modes[0],
                description='Mode:',
                layout={'vertical_align': 'middle', 'width': '2.8in'},
                style={'description_width': '1.25in'}
            )
            self.__utilDropdownStatus[groupID]['mode'] = modeDropdown
            self.__widgetIDtoUtil[modeDropdown.model_id] = ('mode', groupID)
            modeDropdown.observe(self.changeUtilDropdown)
            paramDropdown = widgets.Dropdown(
                options=['Constant', 'Wait time', 'In vehicle time', 'Access time'],
                value='Constant',
                description='Utility param:',
                layout={'vertical_align': 'middle', 'width': '2.8in'},
                style={'description_width': '1.25in'}
            )
            self.__utilDropdownStatus[groupID]['param'] = paramDropdown
            self.__widgetIDtoUtil[paramDropdown.model_id] = ('param', groupID)
            paramDropdown.observe(self.changeUtilDropdown)
            valueBox = widgets.FloatText(value=0.0, step=0.05, description="Value: ", disabled=False,
                                         layout={'vertical_align': 'middle', 'width': '1.8in'})
            self.__utilDropdownStatus[groupID]['value'] = valueBox
            self.__widgetIDtoUtil[valueBox.model_id] = ('value', groupID)
            valueBox.observe(self.changeUtilValue)
            utilStack.append(
                widgets.HBox([label, widgets.VBox([tripTypeDropdown, modeDropdown, paramDropdown]), valueBox]))

        MFDstack = []
        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            initialAutoData = self.model.scenarioData['subNetworkDataFull'].loc[
                              (self.model.scenarioData['subNetworkDataFull'].Type == 'Road') &
                              (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), :]
            microtypeRoadNetworks = [widgets.HTML(
                value="<center><b>Microtype " + mID + "</b></center>"
            )]
            autoVBox = []
            for row in initialAutoData.itertuples():
                roadNetworkParameters = [widgets.HTML(
                    value="<center><i>" + row.ModesAllowed + "</i></center>",
                    layout=Layout(width='1.25in', align_content='center'))]
                parameterVBox = []
                parameterVBox.append(widgets.FloatSlider(value=row.vMax, min=0, max=25, step=0.5,
                                                         description="Max speed (m/s)",
                                                         orientation='horizontal',
                                                         style={'description_width': '1.25in'})
                                     )
                parameterVBox[-1].observe(self.response, names="value")
                self.__widgetIDtoField[parameterVBox[-1].model_id] = ('vMax', row.Index)
                if ~np.isnan(row.k_jam):
                    parameterVBox.append(widgets.FloatSlider(value=row.k_jam, min=0.1, max=0.3, step=0.002,
                                                             description="Jam density (veh/m)",
                                                             orientation='horizontal',
                                                             style={'description_width': '1.25in'}))
                    parameterVBox[-1].observe(self.response, names="value")
                    self.__widgetIDtoField[parameterVBox[-1].model_id] = ('densityMax', row.Index)
                if ~np.isnan(row.capacityFlow):
                    parameterVBox.append(widgets.FloatSlider(value=row.capacityFlow, min=0.1, max=0.6, step=0.002,
                                                             description="Capacity flow (veh/s)",
                                                             orientation='horizontal',
                                                             style={'description_width': '1.25in'}))
                    parameterVBox[-1].observe(self.response, names="value")
                    self.__widgetIDtoField[parameterVBox[-1].model_id] = ('capacityFlow', row.Index)
                if ~np.isnan(row.waveSpeed):
                    parameterVBox.append(widgets.FloatSlider(value=row.waveSpeed, min=0.1, max=0.6, step=0.002,
                                                             description="Backwards wave spd (m/s)",
                                                             orientation='horizontal',
                                                             style={'description_width': '1.25in'}))
                    parameterVBox[-1].observe(self.response, names="value")
                    self.__widgetIDtoField[parameterVBox[-1].model_id] = ('waveSpeed', row.Index)
                if ~np.isnan(row.smoothingFactor):
                    parameterVBox.append(widgets.FloatSlider(value=row.smoothingFactor, min=0.1, max=0.6, step=0.002,
                                                             description="Smoothing Factor",
                                                             orientation='horizontal',
                                                             style={'description_width': '1.25in'}))
                    parameterVBox[-1].observe(self.response, names="value")
                    self.__widgetIDtoField[parameterVBox[-1].model_id] = ('smoothingFactor', row.Index)
                roadNetworkParameters.append(widgets.VBox(parameterVBox))
                autoVBox.append(widgets.HBox(roadNetworkParameters))
            microtypeRoadNetworks.append(widgets.VBox(autoVBox))
            MFDstack.append(widgets.HBox(microtypeRoadNetworks))

        dedicatedTitleStack = [widgets.HTML(value="<center><b>Microtype</b></center>")]
        dedicatedBusStack = [widgets.HTML(value="<center><i>Bus</i></center>")]
        dedicatedBikeStack = [widgets.HTML(value="<center><i>Bike</i></center>")]

        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            dedicatedTitleStack.append(widgets.HTML(value="<center><i>{}</i></center>".format(mID)))
            initialAutoData = self.model.scenarioData['subNetworkDataFull'].loc[
                              self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Auto') &
                              (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), :]
            self.__microtypeToMixedNetworkID[mID] = initialAutoData

            initialBusData = self.model.scenarioData['subNetworkDataFull'].loc[
                             self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Bus') &
                             (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID) &
                             self.model.scenarioData['subNetworkDataFull'].Dedicated, :]
            self.__microtypeToBusNetworkID[mID] = initialBusData

            initialBikeData = self.model.scenarioData['subNetworkDataFull'].loc[
                              self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Bike') &
                              (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID) &
                              self.model.scenarioData['subNetworkDataFull'].Dedicated, :]
            self.__microtypeToBikeNetworkID[mID] = initialBikeData

            dedicatedBusStack.append(
                widgets.FloatSlider(value=0, min=0, max=0.75, step=0.01, layout=Layout(width='180px')))
            dedicatedBusStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[dedicatedBusStack[-1].model_id] = ('dedicated', (mID, 'Bus'))

            dedicatedBikeStack.append(
                widgets.FloatSlider(value=0, min=0, max=0.75, step=0.01, layout=Layout(width='180px')))
            dedicatedBikeStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[dedicatedBikeStack[-1].model_id] = ('dedicated', (mID, 'Bike'))

        costTitleStack = [widgets.HTML(value="<center><b>Microtype</b></center>")]
        costBusStack = [widgets.HTML(value="<center><i>Bus</i></center>")]
        costBusSeniorStack = [widgets.HTML(value="<center><i>Bus (Senior)</i></center>")]

        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            costTitleStack.append(widgets.HTML(value="<center><i>{}</i></center>".format(mID)))
            busData = self.model.scenarioData['modeData']['bus'].loc[
                      self.model.scenarioData['modeData']['bus'].index == mID, :]

            costBusStack.append(
                widgets.FloatSlider(value=busData.PerStartCost[0], min=0, max=5.0, step=0.1,
                                    layout=Layout(width='180px')))
            costBusStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[costBusStack[-1].model_id] = ('fare', (mID, 'Bus'))

            costBusSeniorStack.append(
                widgets.FloatSlider(value=busData.PerStartCost[0] * busData.SeniorFareDiscount[0], min=0, max=5.0,
                                    step=0.1,
                                    layout=Layout(width='180px')))
            costBusSeniorStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[costBusSeniorStack[-1].model_id] = ('fareSenior', (mID, 'Bus'))

        headwayStack = []

        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            busServiceData = self.model.scenarioData['modeData']['bus'].loc[mID]
            self.__microtypeToBusService[mID] = busServiceData
            headwayStack.append(widgets.IntSlider(busServiceData.Headway, 90, 1800, 30, description="Microtype " + mID))
            headwayStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[headwayStack[-1].model_id] = ('headway', (mID, 'Bus'))

        coverageStack = []

        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            busServiceData = self.__microtypeToBusService[mID]
            coverageStack.append(widgets.FloatSlider(value=busServiceData.CoveragePortion, min=0.02, max=1.0, step=0.02,
                                                     description="Microtype " + mID))
            coverageStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[coverageStack[-1].model_id] = ('coverage', (mID, "Bus"))

        titleStack = [widgets.HTML(value="<center><b>Microtype</b></center>")]
        userCostStack = [widgets.HTML(value="<center><i>User Costs</i></center>")]
        systemCostStack = [widgets.HTML(value="<center><i>System Costs</i></center>")]
        externalityCostStack = [widgets.HTML(value="<center><i>Externality Costs</i></center>")]
        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            titleStack.append(widgets.HTML(value="<center><i>{}</i></center>".format(mID)))
            userCostStack.append(
                widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1, layout=Layout(width='180px')))
            self.__widgetIDtoField[userCostStack[-1].model_id] = ('cost', (mID, 'User'))
            userCostStack[-1].observe(self.response, names="value")
            systemCostStack.append(
                widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1, layout=Layout(width='180px')))
            self.__widgetIDtoField[systemCostStack[-1].model_id] = ('cost', (mID, 'System'))
            systemCostStack[-1].observe(self.response, names="value")
            externalityCostStack.append(
                widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1, layout=Layout(width='180px')))
            self.__widgetIDtoField[externalityCostStack[-1].model_id] = ('cost', (mID, 'Externality'))
            externalityCostStack[-1].observe(self.response, names="value")
        costAccordion = widgets.HBox(
            [widgets.VBox(titleStack), widgets.VBox(userCostStack), widgets.VBox(systemCostStack),
             widgets.VBox(externalityCostStack)])

        dataAccordion = widgets.Accordion(
            [widgets.VBox(networkLengthStack), widgets.VBox(populationStack), widgets.VBox(utilStack),
             widgets.VBox(MFDstack)])
        for ind, title in enumerate(('Network Length', 'Population', 'Utility parameters', 'MFD parameters')):
            dataAccordion.set_title(ind, title)

        scenarioAccordion = widgets.Accordion(
            [widgets.HBox(
                [widgets.VBox(dedicatedTitleStack), widgets.VBox(dedicatedBusStack), widgets.VBox(dedicatedBikeStack)]),
                widgets.HBox(
                    [widgets.VBox(costTitleStack), widgets.VBox(costBusStack),
                     widgets.VBox(costBusSeniorStack)]),
                widgets.VBox(headwayStack), widgets.VBox(coverageStack)])

        for ind, title in enumerate(('Lane dedication', 'Cost', 'Bus headway (s)', 'Bus service area')):
            scenarioAccordion.set_title(ind, title)

        accordionChildren = [costAccordion, scenarioAccordion, dataAccordion]

        accordion = widgets.Accordion(children=accordionChildren)
        for ind, title in enumerate(('Cost parameters', 'Scenario parameters', 'Input data')):
            accordion.set_title(ind, title)

        # for ind, title in enumerate(
        #         ('Population by group', 'MFD Parameters', 'Bus lane dedication', 'Bus headways (s)',
        #          'Bus service area')):
        #     accordion.set_title(ind, title)

        gs = widgets.GridspecLayout(3, 3)

        self.__loadingWidget = widgets.HTML(
            value="<center><i>Model Running</i></center>"
        )

        self.__plotStateWidget = widgets.Dropdown(
            options=['Auto speed', 'Mode split', 'Auto accumulation'],
            value='Auto speed',
            description='Plot type:',
            disabled=False,
        )
        self.__plotStateWidget.observe(self.updatePlots, names="value")

        # hardReset = widgets.Button(description="Reset model state",
        #                            tooltip="Click if everything is broken",
        #                            layout=Layout(width='100%', height='0.5in'))
        # hardReset.on_click(self.hardReset)

        gs[:, :2] = accordion
        # gs[0, 2] = rerunModel
        gs[0, 2] = self.__loadingWidget
        self.__out = widgets.Output(layout={'border': '1px solid black'})
        # gs[1, 2] = setRef
        gs[1, 2] = widgets.VBox([rerunModel, setRef])
        gs[2, 2] = widgets.VBox([generateButton, downloadButton])

        # self.createDownloadLink()
        return gs

    def response(self, change, otherStuff=None):
        field = self.__widgetIDtoField[change.owner.model_id]
        self.modifyModel(field, change)

    def getDedicationDistance(self, microtype: str, modeName: str, dedicationPortion: float):
        roadDF = self.returnRoadNetworkLengths(microtype)
        modeDF = self.returnModeNetworkLengths(microtype, modeName)
        routeLength = roadDF.sum() * self.model.scenarioData.data['modeData'][modeName.lower()].loc[microtype].get(
            'CoveragePortion', 1)
        newDedicatedLength = routeLength * dedicationPortion
        newMixedLength = modeDF.sum() - newDedicatedLength
        dedicatedIdx = modeDF.index[1]
        mixedIdx = modeDF.index[0]
        return dedicatedIdx, newDedicatedLength, mixedIdx, newMixedLength

    def modifyModel(self, changeType, value):
        if hasattr(value, 'new'):
            newValue = value.new
        elif np.issubdtype(type(value), np.number):
            newValue = value
        else:
            print("BAD INPUT")
            print(value)
            print(changeType)
            raise NotImplementedError
        if changeType[0] == 'dedicated':
            microtype, modeName = changeType[1]
            dedicatedIdx, newDedicatedLength, mixedIdx, newMixedLength = self.getDedicationDistance(
                microtype, modeName, newValue)
            # NOTE: Right now this relies on the ordering of the input csv
            self.model.data.updateNetworkLength(mixedIdx, newMixedLength)
            self.model.data.updateNetworkLength(dedicatedIdx, newDedicatedLength)
            self.model.scenarioData['subNetworkData'].loc[mixedIdx, 'Length'] = newMixedLength
            self.model.scenarioData['subNetworkData'].loc[dedicatedIdx, 'Length'] = newDedicatedLength
        if changeType[0] == 'headway':
            microtype, modeName = changeType[1]
            self.model.scenarioData['modeData'][modeName.lower()].loc[microtype, 'Headway'] = newValue
        if changeType[0] == 'fleetSize':
            microtype, modeName = changeType[1]
            self.model.scenarioData['modeData'][modeName.lower()].loc[microtype, 'Headway'] = newValue
        if changeType[0] == 'perMileCharge':
            microtype, modeName = changeType[1]
            self.model.data.setModePerMileCosts(modeName.lower(), microtype, newValue, public=True)
        if changeType[0] == 'fare':
            microtype, modeName = changeType[1]
            self.model.data.setModeStartCosts(modeName.lower(), microtype, newValue)
        if changeType[0] == 'fareSenior':
            microtype, modeName = changeType[1]
            self.model.data.setModeFleetSize(modeName.lower(), microtype, newValue)
        if changeType[0] == 'coverage':
            microtype, modeName = changeType[1]
            self.model.scenarioData['modeData'][modeName.lower()].loc[microtype, 'CoveragePortion'] = newValue
            self.model.microtypes[microtype].networks.updateModeData()
        if changeType[0] == 'population':
            mask = (self.model.scenarioData['populations']['MicrotypeID'] == changeType[1][0]) & (
                    self.model.scenarioData['populations']['PopulationGroupTypeID'] == changeType[1][1])
            if sum(mask) == 1:
                self.model.scenarioData['populations'].loc[mask, 'Population'] = newValue
                self.model.updatePopulation()
        if changeType[0] == 'vMax':
            self.model.scenarioData['subNetworkData'].loc[changeType[1], 'vMax'] = newValue
            self.model.microtypes.recompileMFDs()  # TODO: simplify to only microtyype
        if changeType[0] == 'densityMax':
            self.model.scenarioData['subNetworkData'].loc[changeType[1], 'k_jam'] = newValue
            self.model.microtypes.recompileMFDs()
        if changeType[0] == 'capacityFlow':
            self.model.scenarioData['subNetworkData'].loc[changeType[1], 'capacityFlow'] = newValue
            self.model.microtypes.recompileMFDs()
        if changeType[0] == 'smoothingFactor':
            self.model.scenarioData['subNetworkData'].loc[changeType[1], 'smoothingFactor'] = newValue
            self.model.microtypes.recompileMFDs()
        if changeType[0] == 'waveSpeed':
            self.model.scenarioData['subNetworkData'].loc[changeType[1], 'waveSpeed'] = newValue
            self.model.microtypes.recompileMFDs()
        if changeType[0] == 'networkLength':
            mID = changeType[1]
            self.model.data.updateMicrotypeNetworkLength(mID, newValue)
        if changeType[0] == 'cost':
            mID, costType = changeType[1]
            if costType == "System":
                self.__optimizer.updateAlpha("Operator", newValue, mID)
                self.__optimizer.updateAlpha("Dedication", newValue, mID)
            else:
                self.__optimizer.updateAlpha(costType, newValue, mID)
            self.updatePlots()

    def returnModeNetworkLengths(self, mID, modeName):
        return self.model.scenarioData['subNetworkData'].loc[
            self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.lower().str.contains(modeName.lower()) & (
                    self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), 'Length']

    def returnRoadNetworkLengths(self, mID):
        return self.model.scenarioData['subNetworkData'].loc[
            (self.model.scenarioData['subNetworkDataFull'].Type == "Road") & (
                    self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), 'Length']

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
        self.__downloadHTML.format(payload="")
        if self.model.choice.broken | (not self.model.successful):
            print("Starting from a bad place so I'll reset")
            self.model.initializeAllTimePeriods(True)
        if self.__showFigure:
            self.__loadingWidget.value = "<center><i>Model Running</i></center>"
            self.__downloadWidget.layout.visibility = "hidden"
            self.__downloadHTML.format(payload="")
        self.model.collectAllCharacteristics()
        if self.__showFigure:
            self.updatePlots()
            # self.createDownloadLink()
            self.__loadingWidget.value = "<center><b>Complete</b></center>"

    def updatePlots(self, message=None):
        time, spds = self.model.plotAllDynamicStats('v')
        for ind, (mode, handle) in enumerate(self.__dataToHandle['speed']['current'].items()):
            handle.y = spds[:, ind]
            handle.x = time
            handle.visible = True

        microtypeModeSplits = dict()
        for mID in self.model.microtypeIdToIdx.keys():
            time, microtypeModeSplits[mID] = self.model.plotAllDynamicStats('modes', microtype=mID)

        for modeID, (mode, group) in enumerate(self.__dataToHandle['modeSplit']['current'].items()):
            for mID, handle in group.items():
                handle.y = microtypeModeSplits[mID][:, modeID]
                handle.x = time
                handle.visible = True

        time, modeSpeeds = self.model.plotAllDynamicStats('modeSpeeds')
        for mode, group in self.__dataToHandle['modeSpeed']['current'].items():
            for mID, handle in group.items():
                handle.y = modeSpeeds[(mID, mode)].values
                handle.x = time
                handle.visible = True

        # mIDs, costs = self.model.plotAllDynamicStats('costs')
        costs = self.__optimizer.sumAllCosts()
        purposes = list(set(self.model.scenarioData.tripPurposeToIdx.keys()).difference({'home', 'work'}))
        accessibility = self.model.calculateAccessibility(normalize=True).loc[:, purposes]

        for ind, ((mID, groupID), handle) in enumerate(self.__dataToHandle['accessibility']['current'].items()):
            handle.y = accessibility.loc[mID, groupID].values

        for (mID, groupID), plot in self.__dataToHandle['accessibilityDiff'].items():
            yRef = np.array(self.__dataToHandle['accessibility']['ref'][mID, groupID].y)
            yCurrent = np.array(self.__dataToHandle['accessibility']['current'][mID, groupID].y)
            plot.y = yCurrent - yRef

        for ind, (mID, handle) in enumerate(self.__dataToHandle['cost']['current'].items()):
            if isinstance(costs, pd.DataFrame):
                handle.y = costs.loc[mID, :].values
            else:
                handle.y = handle.y * 0.0

        for mID, plot in self.__dataToHandle['costDiff'].items():
            yRef = np.array(self.__dataToHandle['cost']['ref'][mID].y)
            yCurrent = np.array(self.__dataToHandle['cost']['current'][mID].y)
            if len(yRef) == 0:
                plot.y = yCurrent * 0.0
            elif len(yRef) < len(yCurrent):
                y = np.zeros_like(yCurrent) * np.nan
                y[:len(yRef)] = yCurrent[:len(yRef)] - yRef
            elif len(yCurrent) < len(yRef):
                y = np.zeros_like(yRef) * np.nan
                y[:len(yCurrent)] = yCurrent - yRef[:len(yRef)]
            else:
                plot.y = yCurrent - yRef

        for mID, plot in self.__dataToHandle['costCombined']['current'].items():
            yRef = np.array(self.__dataToHandle['cost']['ref'][mID].y)
            yCurrent = np.array(self.__dataToHandle['cost']['current'][mID].y)
            if len(yRef) == 0:
                plot.y = yCurrent * 0.0
            elif len(yCurrent) < len(yRef):
                y = np.zeros_like(yRef) * np.nan
                y[:len(yCurrent)] = yCurrent
            else:
                plot.y = yCurrent

        for mID, plot in self.__dataToHandle['costCombined']['ref'].items():
            yRef = np.array(self.__dataToHandle['cost']['ref'][mID].y)
            yCurrent = np.array(self.__dataToHandle['cost']['current'][mID].y)
            if len(yRef) == 0:
                plot.y = yCurrent * 0.0
            else:
                plot.y = yRef

        for mode, microtypeDict in self.__dataToHandle['modeSplitDiff'].items():
            for mID, plot in microtypeDict.items():
                yRef = np.array(self.__dataToHandle['modeSplit']['ref'][mode][mID].y)
                yCurrent = np.array(self.__dataToHandle['modeSplit']['current'][mode][mID].y)
                if len(yRef) == 0:
                    plot.y = yCurrent * 0.0
                elif len(yRef) < len(yCurrent):
                    y = plot.y.copy()
                    y[:len(yRef)] = yCurrent[:len(yRef)] - yRef
                    y[len(yRef):] = np.nan
                    plot.y = y
                elif len(yCurrent) < len(yRef):
                    y = plot.y.copy()
                    y[:len(yCurrent)] = yCurrent - yRef[:len(yCurrent)]
                    y[len(yCurrent):] = np.nan
                    plot.y = y
                else:
                    plot.y = yCurrent - yRef
                plot.x = self.__dataToHandle['modeSplit']['current'][mode][mID].x

        for mID, plot in self.__dataToHandle['speedDiff'].items():
            yRef = np.array(self.__dataToHandle['speed']['ref'][mID].y)
            yCurrent = np.array(self.__dataToHandle['speed']['current'][mID].y)
            if len(yRef) == 0:
                plot.y = yCurrent * 0.0
            elif len(yRef) < len(yCurrent):
                y = plot.y.copy()
                y[:len(yRef)] = yCurrent[:len(yRef)] - yRef
                y[len(yRef):] = np.nan
                plot.y = y
            elif len(yCurrent) < len(yRef):
                y = plot.y.copy()
                y[:len(yCurrent)] = yCurrent - yRef[:len(yCurrent)]
                y[len(yCurrent):] = np.nan
                plot.y = y
            else:
                plot.y = yCurrent - yRef
            plot.x = self.__dataToHandle['speed']['current'][mID].x

    def copyCurrentToRef(self, message=None):
        for plotType, plots in self.__dataToHandle.items():
            if plotType.endswith("Diff"):
                continue
            elif (plotType == "modeSpeed") | (plotType == "modeSplit"):
                for mode, group in plots['current'].items():
                    for line, value in group.items():
                        plots['ref'][mode][line].y = value.y
                        plots['ref'][mode][line].x = value.x
            else:
                for line, value in plots['current'].items():
                    plots['ref'][line].y = value.y
                    plots['ref'][line].x = value.x
        for plotType, plots in self.__dataToHandle.items():
            if plotType == "costDiff":
                for line in plots.keys():
                    # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                    # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                    self.__dataToHandle['costDiff'][line].y = [0] * len(self.__dataToHandle['cost']['current'][line].y)
            if plotType == "accessibilityDiff":
                for line in plots.keys():
                    # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                    # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                    self.__dataToHandle['accessibilityDiff'][line].y = [0] * len(
                        self.__dataToHandle['accessibility']['current'][line].y)
            if plotType == "modeSplitDiff":
                for mode, group in plots.items():
                    for mID in group.keys():
                        # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                        # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                        self.__dataToHandle['modeSplitDiff'][mode][mID].y = [0] * len(
                            self.__dataToHandle['modeSplit']['current'][mode][mID].y)
            if plotType == "speedDiff":
                for line in plots.keys():
                    # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                    # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                    self.__dataToHandle['speedDiff'][line].y = [0] * len(
                        self.__dataToHandle['speed']['current'][line].y)

    def createDownloadLink(self, message=None):
        if not os.path.exists('temp'):
            os.makedirs('temp')
        modeSplit, speed, utility, continuousSpeed = self.model.toPandas()
        modeSplit.to_csv('temp/modeSplitOutput.csv.gz')
        speed.to_csv('temp/speedOutput.csv.gz')
        utility.to_csv('temp/utilityOutput.csv.gz')
        continuousSpeed.to_csv('temp/continuousOutput.csv.gz')

        # create a ZipFile object
        zipObj = ZipFile('temp/sample.zip', 'w')
        # Add multiple files to the zip
        zipObj.write('temp/modeSplitOutput.csv.gz')
        zipObj.write('temp/speedOutput.csv.gz')
        zipObj.write('temp/utilityOutput.csv.gz')
        zipObj.write('temp/continuousOutput.csv.gz')
        # close the Zip File
        zipObj.close()

        with open("temp/sample.zip", "rb") as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes)

        payload = b64.decode()

        html_button = self.__downloadHTML.format(payload=payload)

        self.__downloadWidget.value = html_button
        self.__downloadWidget.layout.visibility = 'visible'
        #
        # out = FileLink(r'temp/sample.zip')
        # # urlopen(out)
        # display(out)
        #
        #
        # print('BOO')

    def changeUtilDropdown(self, message=None):
        menu, popGroup = self.__widgetIDtoUtil[message.owner.model_id]
        tripPurpose = self.__utilDropdownStatus[popGroup]['tripType'].value
        mode = self.__utilDropdownStatus[popGroup]['mode'].value
        param = self.__paramNames[self.__utilDropdownStatus[popGroup]['param'].value]
        mIDs, vals = self.model.getUtilityParam(param, popGroup, tripPurpose, mode)
        self.__utilDropdownStatus[popGroup]['value'].value = vals[0]

    def changeUtilValue(self, message=None):
        menu, popGroup = self.__widgetIDtoUtil[message.owner.model_id]
        tripPurpose = self.__utilDropdownStatus[popGroup]['tripType'].value
        mode = self.__utilDropdownStatus[popGroup]['mode'].value
        param = self.__paramNames[self.__utilDropdownStatus[popGroup]['param'].value]
        if isinstance(message.new, dict):
            if 'value' in message.new:
                val = message.new['value']
                self.model.updateUtilityParam(val, param, popGroup, tripPurpose, mode)
        elif type(message.new) == int or float:
            val = message.new
            self.model.updateUtilityParam(val, param, popGroup, tripPurpose, mode)

    def init(self):
        print("Initialized")
        self.updateCosts()
        print("Costs updated")
        self.copyCurrentToRef()
        print("Copied to Ref")

    def hardReset(self, message=None):
        self.model.scenarioData.loadData()
        self.model.scenarioData.loadModeData()
        self.model.resetNetworks()
        self.model.readFiles()
        self.updateCosts()
