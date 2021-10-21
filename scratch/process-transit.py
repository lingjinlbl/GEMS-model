import pandas as pd
import os
import numpy as np

miles2meters = 1609.34  # TODO: Change back to right number

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "SubNetworks-raw.csv")
subNetworks = pd.read_csv(path)
path = os.path.join(ROOT_DIR, "..", "input-data-production", "TransitService.csv")
transitService = pd.read_csv(path)
transitService["ServicePerMile"] = transitService["veh_revenue_miles"] / transitService["dir_route_miles"] / 365.0

path = os.path.join(ROOT_DIR, "..", "input-data-production", "TransitSystemCosts.csv")
transitCosts = pd.read_csv(path)[["Geotype", "Mode", "Intercept", "BetaFleetSize"]]
transitCosts["costPerVehicleHour"] = transitCosts["BetaFleetSize"] / 365.0 / 18.0

subNetworks["Geotype"] = subNetworks["MicrotypeID"].str[0]
subNetworks["Microtype"] = subNetworks["MicrotypeID"].str[2]

geoDists = subNetworks.groupby("Geotype").agg({"LengthNetwork": sum}).merge(
    transitService.loc[transitService["mode_group"] == "bus"], left_on="Geotype", right_on="geotype")
geoDists = geoDists.merge(
    transitCosts.loc[transitCosts["Mode"] == "bus"], left_on="geotype", right_on="Geotype"
)
geoDists["NetworkFraction"] = geoDists["dir_route_miles"] / geoDists["LengthNetwork"] / miles2meters
interliningFactor = 0.3

railDists = subNetworks.groupby("Geotype").agg({"LengthNetwork": sum}).merge(
    transitService.loc[transitService["mode_group"] == "rail"], left_on="Geotype", right_on="geotype")
railDists = railDists.merge(
    transitCosts.loc[transitCosts["Mode"] == "rail"], left_on="geotype", right_on="Geotype"
)
railDists["NetworkFraction"] = railDists["dir_route_miles"] / railDists["LengthNetwork"] / miles2meters

subNetworks = subNetworks.merge(geoDists.loc[:, ["geotype", "NetworkFraction", "ServicePerMile", "costPerVehicleHour"]],
                                left_on="Geotype",
                                right_on="geotype", how='left')

subNetworks = subNetworks.merge(
    railDists.loc[:, ["geotype", "NetworkFraction", "ServicePerMile", "costPerVehicleHour"]], left_on="Geotype",
    right_on="geotype", suffixes=('', '_rail'), how='left').fillna(0)

# vMax,densityMax,capacityFlow,smoothingFactor,waveSpeed,MFD


AutoBus = subNetworks[['MicrotypeID', 'Microtype']].set_index('MicrotypeID')
AutoBus['Length'] = subNetworks['LengthNetwork'].values * miles2meters
AutoBus['vMax'] = 16
AutoBus['densityMax'] = np.nan
AutoBus['capacityFlow'] = np.nan
AutoBus['smoothingFactor'] = np.nan
AutoBus['waveSpeed'] = np.nan
AutoBus['MFD'] = "bottleneck"
AutoBus['Type'] = "Road"
AutoBus['Dedicated'] = False
AutoBus.loc[AutoBus['Microtype'] == '1', 'vMax'] = 17.0
AutoBus.loc[AutoBus['Microtype'] == '1', 'densityMax'] = 0.15
AutoBus.loc[AutoBus['Microtype'] == '1', 'capacityFlow'] = 0.18
AutoBus.loc[AutoBus['Microtype'] == '1', 'smoothingFactor'] = 0.13
AutoBus.loc[AutoBus['Microtype'] == '1', 'waveSpeed'] = 3.78
AutoBus.loc[AutoBus['Microtype'] == '1', 'MFD'] = "loder"
AutoBus.loc[AutoBus['Microtype'] == '2', 'vMax'] = 17.0
AutoBus.loc[AutoBus['Microtype'] == '2', 'densityMax'] = 0.15
AutoBus.loc[AutoBus['Microtype'] == '2', 'capacityFlow'] = 0.18
AutoBus.loc[AutoBus['Microtype'] == '2', 'smoothingFactor'] = 0.13
AutoBus.loc[AutoBus['Microtype'] == '2', 'waveSpeed'] = 3.78
AutoBus.loc[AutoBus['Microtype'] == '2', 'MFD'] = "loder"
AutoBus.loc[AutoBus['Microtype'] == '3', 'vMax'] = 28.0
AutoBus.loc[AutoBus['Microtype'] == '3', 'capacityFlow'] = 0.4
AutoBus.loc[AutoBus['Microtype'] == '3', 'MFD'] = "bottleneck"
AutoBus.loc[AutoBus['Microtype'] == '4', 'vMax'] = 18.8
AutoBus.loc[AutoBus['Microtype'] == '4', 'capacityFlow'] = 0.38
AutoBus.loc[AutoBus['Microtype'] == '4', 'MFD'] = "bottleneck"
AutoBus.loc[AutoBus['Microtype'] == '5', 'vMax'] = 18.8
AutoBus.loc[AutoBus['Microtype'] == '5', 'capacityFlow'] = 0.38
AutoBus.loc[AutoBus['Microtype'] == '5', 'MFD'] = "bottleneck"
AutoBus.loc[AutoBus['Microtype'] == '6', 'vMax'] = 18.8
AutoBus.loc[AutoBus['Microtype'] == '6', 'capacityFlow'] = 0.38
AutoBus.loc[AutoBus['Microtype'] == '6', 'MFD'] = "bottleneck"
AutoBus.drop(columns='Microtype', inplace=True)
Bus = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
Bus['Length'] = 0.0
Bus['vMax'] = 16
Bus['densityMax'] = np.nan
Bus['capacityFlow'] = 0.6
Bus['smoothingFactor'] = np.nan
Bus['waveSpeed'] = np.nan
Bus['MFD'] = "bottleneck"
Bus['Type'] = "Road"
Bus['Dedicated'] = True
Walk = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
Walk['Length'] = subNetworks['LengthNetwork'].values * 0.8 * miles2meters
Walk['vMax'] = 1.4
Walk['densityMax'] = np.nan
Walk['capacityFlow'] = np.nan
Walk['smoothingFactor'] = np.nan
Walk['waveSpeed'] = np.nan
Walk['MFD'] = "fixed"
Walk['Type'] = "Sidewalk"
Walk['Dedicated'] = True
Rail = subNetworks[['MicrotypeID']].set_index('MicrotypeID').copy()
Rail['Length'] = subNetworks['LengthNetwork'].values * subNetworks['NetworkFraction_rail'].values * miles2meters
Rail['vMax'] = 20
Rail['densityMax'] = np.nan
Rail['capacityFlow'] = np.nan
Rail['smoothingFactor'] = np.nan
Rail['waveSpeed'] = np.nan
Rail['MFD'] = "fixed"
Rail['Type'] = "Rail"
Rail['Dedicated'] = True
Bike = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
Bike['Length'] = 0.0
Bike['vMax'] = 4.2
Bike['densityMax'] = 0.15
Bike['capacityFlow'] = np.nan
Bike['smoothingFactor'] = np.nan
Bike['waveSpeed'] = np.nan
Bike['MFD'] = "fixed"
Bike['Type'] = "Road"
Bike['Dedicated'] = True

joined = pd.concat([AutoBus, Bus, Walk, Rail, Bike], axis=1,
                   keys=['Auto-Bus-Bike', 'Bus', 'Walk', 'Rail', 'Bike'],
                   names=['ModesAllowed', 'Field'])

# subNetworks[pd.MultiIndex.from_tuples([('Length', 'Bus')])] = subNetworks["LengthNetwork"] * miles2meters * interliningFactor * subNetworks["NetworkFraction"]
out = joined.stack('ModesAllowed').reset_index()
out = out.loc[(out['ModesAllowed'] != 'Rail') | (out['Length'] > 0)]
out.index.name = "SubnetworkID"
out['avgLinkLength'] = 50.0

collected = []

# Sub = out.loc[out['ModesAllowed'] == "Auto-Bike", ['ModesAllowed']]
# Sub['SubnetworkID'] = Sub.index.copy().values
# Sub['ModeTypeID'] = "auto"
#
# collected.append(Sub)
#
# Sub = out.loc[out['ModesAllowed'] == "Auto-Bike", ['ModesAllowed']]
# Sub['SubnetworkID'] = Sub.index.copy().values
# Sub['ModeTypeID'] = "bike"
#
# collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Auto-Bus-Bike", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "auto"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Auto-Bus-Bike", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "bus"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Auto-Bus-Bike", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "bike"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Bus", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "bus"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Walk", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "walk"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Rail", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "rail"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Bike", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "bike"

collected.append(Sub)

modeToSubNetwork = pd.concat(collected)

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "SubNetworks.csv")
out.to_csv(path)

path = os.path.join(ROOT_DIR, "..", "input-data-transgeo", "ModeToSubNetwork.csv")
modeToSubNetwork.to_csv(path, index=False)
print("DONE")
