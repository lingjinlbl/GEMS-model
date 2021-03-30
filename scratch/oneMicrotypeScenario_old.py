import os
import shutil

import pandas as pd

geotype = "A"
inFolder = "input-data-production"
outFolder = "input-data-geotype-" + geotype
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

newDir = os.path.join(ROOT_DIR, "..", outFolder)
if os.path.exists(newDir):
    shutil.rmtree(newDir)
os.makedirs(newDir)

for copiedFile in ["DistanceBins", "ObjectiveFunctionUserCosts", "PopulationGroups", "TimePeriods", "TripGeneration",
                   "TripPurposes", "MicrotypeAssignment", "TransitionMatrices"]:
    oldPath = os.path.join(ROOT_DIR, "..", inFolder, copiedFile + ".csv")
    newPath = os.path.join(ROOT_DIR, "..", outFolder, copiedFile + ".csv")
    shutil.copyfile(oldPath, newPath)

modesDir = os.path.join(ROOT_DIR, "..", outFolder, "modes")
if os.path.exists(modesDir):
    shutil.rmtree(modesDir)
os.makedirs(modesDir)

for mode in ["auto", "bike", "bus", "rail", "walk"]:
    oldPath = os.path.join(ROOT_DIR, "..", inFolder, "modes", mode + ".csv")
    newPath = os.path.join(ROOT_DIR, "..", outFolder, "modes", mode + ".csv")
    df = pd.read_csv(oldPath)
    newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
    newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
    newdf.to_csv(newPath, index=False)

# %% DistanceDistribution
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "DistanceDistribution.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "DistanceDistribution.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.OriginMicrotypeID.str.startswith(geotype) & df.DestinationMicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
newdf.to_csv(newPath, index=False)

# %% LaneDedicationCost
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "LaneDedicationCost.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "LaneDedicationCost.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.to_csv(newPath, index=False)

# %% Microtypes
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "Microtypes.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "Microtypes.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.to_csv(newPath, index=False)

# %% OriginDestination
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "OriginDestination.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "OriginDestination.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.HomeMicrotypeID.str.startswith(geotype) & df.OriginMicrotypeID.str.startswith(geotype) & df.DestinationMicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "HomeMicrotypeID"] = newdf.loc[:, "HomeMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "OriginMicrotypeID"] = newdf.loc[:, "OriginMicrotypeID"].str.split('_').str[1].values
newdf.loc[:, "DestinationMicrotypeID"] = newdf.loc[:, "DestinationMicrotypeID"].str.split('_').str[1].values
newdf.to_csv(newPath, index=False)

# %% Population
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "Population.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "Population.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.to_csv(newPath, index=False)

# %% RoadNetworkCosts
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "RoadNetworkCosts.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "RoadNetworkCosts.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newdf.to_csv(newPath, index=False)

# %% SubNetworks
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "SubNetworks.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "SubNetworks.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.MicrotypeID.str.startswith(geotype), :]
newdf.loc[:, "MicrotypeID"] = newdf.loc[:, "MicrotypeID"].str.split('_').str[1].values
newSubNetworks = set(newdf.SubnetworkID.values)
newdf.to_csv(newPath, index=False)

# %% ModeToSubNetwork
oldPath = os.path.join(ROOT_DIR, "..", inFolder, "ModeToSubNetwork.csv")
newPath = os.path.join(ROOT_DIR, "..", outFolder, "ModeToSubNetwork.csv")
df = pd.read_csv(oldPath)
newdf = df.loc[df.SubnetworkID.isin(newSubNetworks), :]
newdf.to_csv(newPath, index=False)

print("AA")
