import pandas as pd
import os


miles2meters = 1609.34 # TODO: Change back to right number

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "..", "input-data-production", "SubNetworks-raw.csv")
subNetworks = pd.read_csv(path)
path = os.path.join(ROOT_DIR, "..", "input-data-production", "TransitService.csv")
transitService = pd.read_csv(path)
transitService["ServicePerMile"] = transitService["veh_revenue_miles"] / transitService["dir_route_miles"] / 365.0

path = os.path.join(ROOT_DIR, "..", "input-data-production", "TransitSystemCosts.csv")
transitCosts = pd.read_csv(path)[["Geotype","Mode","Intercept","BetaFleetSize"]]
transitCosts["costPerVehicleHour"] = transitCosts["BetaFleetSize"]/365.0/18.0

subNetworks["Geotype"] = subNetworks["MicrotypeID"].str[0]

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

subNetworks = subNetworks.merge(geoDists.loc[:, ["geotype", "NetworkFraction", "ServicePerMile","costPerVehicleHour"]], left_on="Geotype",
                                right_on="geotype", how='left')

subNetworks = subNetworks.merge(railDists.loc[:, ["geotype", "NetworkFraction", "ServicePerMile","costPerVehicleHour"]], left_on="Geotype",
                                right_on="geotype", suffixes=('', '_rail'), how='left').fillna(0)

Auto = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
Auto['Length'] = subNetworks['LengthNetwork'].values * interliningFactor * subNetworks[
    'NetworkFraction'].values * miles2meters
Auto['vMax'] = 16
Auto['Type'] = "Road"
Auto['Dedicated'] = False

AutoBus = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
AutoBus['Length'] = subNetworks['LengthNetwork'].values * (
            1.0 - interliningFactor * subNetworks['NetworkFraction'].values) * miles2meters
AutoBus['vMax'] = 16
AutoBus['Type'] = "Road"
AutoBus['Dedicated'] = False
Bus = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
Bus['Length'] = 0.0
Bus['vMax'] = 16
Bus['Type'] = "Road"
Bus['Dedicated'] = True
Walk = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
Walk['Length'] = subNetworks['LengthNetwork'].values * 0.8 * miles2meters
Walk['vMax'] = 1.4
Walk['Type'] = "Sidewalk"
Walk['Dedicated'] = True
Rail = subNetworks[['MicrotypeID']].set_index('MicrotypeID').copy()
Rail['Length'] = subNetworks['LengthNetwork'].values * subNetworks['NetworkFraction_rail'].values * miles2meters
Rail['vMax'] = 20
Rail['Type'] = "Rail"
Rail['Dedicated'] = True
Bike = subNetworks[['MicrotypeID']].set_index('MicrotypeID')
Bike['Length'] = 0.0
Bike['vMax'] = 4.2
Bike['Type'] = "BikeLane"
Bike['Dedicated'] = True

joined = pd.concat([Auto, AutoBus, Bus, Walk, Rail, Bike], axis=1,
                   keys=['Auto-Bike', 'Auto-Bus-Bike', 'Bus-Bike', 'Walk', 'Rail', 'Bike'],
                   names=['ModesAllowed', 'Field'])

# subNetworks[pd.MultiIndex.from_tuples([('Length', 'Bus')])] = subNetworks["LengthNetwork"] * miles2meters * interliningFactor * subNetworks["NetworkFraction"]
out = joined.stack('ModesAllowed').reset_index()
out = out.loc[(out['ModesAllowed'] != 'Rail') | (out['Length'] > 0)]
out.index.name = "SubnetworkID"
out['avgLinkLength'] = 50.0
out['densityMax'] = 0.145

collected = []

Sub = out.loc[out['ModesAllowed'] == "Auto-Bike", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "auto"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Auto-Bike", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "bike"

collected.append(Sub)

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

Sub = out.loc[out['ModesAllowed'] == "Bus-Bike", ['ModesAllowed']]
Sub['SubnetworkID'] = Sub.index.copy().values
Sub['ModeTypeID'] = "bus"

collected.append(Sub)

Sub = out.loc[out['ModesAllowed'] == "Bus-Bike", ['ModesAllowed']]
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

path = os.path.join(ROOT_DIR, "..", "input-data-production", "SubNetworks.csv")
out.to_csv(path)

path = os.path.join(ROOT_DIR, "..", "input-data-production", "ModeToSubNetwork.csv")
modeToSubNetwork.to_csv(path, index=False)
print("DONE")
