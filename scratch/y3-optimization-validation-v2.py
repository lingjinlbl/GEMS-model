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

model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=2)
optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'Bus')],
                      fromToSubNetworkIDs=[('A', 'Bus')],
                      method="min")

initialDemand = model.data.tripRate().copy()

x0 = optimizer.x0()

alphas = np.logspace(-1., 1., 25)
costType = ["User", "Operator", "Externality", "Dedication"]
out = dict()
for alpha in alphas:
    for ct in costType:
        if ct == "User":
            optimizer.updateAlpha(ct, alpha * 20.0)
        else:
            optimizer.updateAlpha(ct, alpha)
        result = optimizer.minimize(x0)
        out[(ct, alpha)] = pd.concat([pd.Series({"alloc": result.x[0], "headway": result.x[1], "fun": result.fun}),
                                      optimizer.sumAllCosts().sum(), optimizer.model.getModeSpeeds().sum()])
        if ct == "User":
            optimizer.updateAlpha(ct, 20.0)
        else:
            optimizer.updateAlpha(ct, 1.0)
        x0 = result.x

df = pd.concat(out)
cols = ["alloc", "headway", "fun"] + costType + list(optimizer.model.getModeSpeeds().columns)
for col in cols:
    sub = df.unstack()[col].unstack().transpose()
    if col in sub.columns:
        sub[col] /= alphas
    elif col == "headway":
        sub *= 1000.0 / 60.0
    plot = sub.reset_index().plot(x='index', logx=True)
    plt.savefig('out-opt-2/1mic/' + col + '.png')
    plt.ylabel(col)
    plt.xlabel('/alpha')
    plt.close('all')

model = Model(ROOT_DIR + "/../input-data", nSubBins=2)
optimizer = Optimizer(model, modesAndMicrotypes=[('A', 'Bus')],
                      fromToSubNetworkIDs=[('A', 'Bus')],
                      method="min")
x0 = optimizer.x0()
initialDemand = model.data.tripRate().copy()

alphas = np.logspace(-1., 1., 25)
costType = ["User", "Operator", "Externality", "Dedication"]
out = dict()
for alpha in alphas:
    for ct in costType:
        if ct == "User":
            optimizer.updateAlpha(ct, alpha * 20.0)
        else:
            optimizer.updateAlpha(ct, alpha)
        result = optimizer.minimize(x0)
        out[(ct, alpha)] = pd.concat([pd.Series({"alloc": result.x[0], "headway": result.x[1], "fun": result.fun}),
                                      optimizer.sumAllCosts().sum(), optimizer.model.getModeSpeeds().sum()])
        if ct == "User":
            optimizer.updateAlpha(ct, 20.0)
        else:
            optimizer.updateAlpha(ct, 1.0)
        x0 = result.x

df = pd.concat(out)
cols = ["alloc", "headway", "fun"] + costType + list(optimizer.model.getModeSpeeds().columns)
for col in cols:
    sub = df.unstack()[col].unstack().transpose()
    if col in sub.columns:
        sub[col] /= alphas
    elif col == "headway":
        sub *= 1000.0 / 60.0
    plot = sub.reset_index().plot(x='index', logx=True)
    plt.savefig('out-opt-2/4mic/' + col + '.png')
    plt.ylabel(col)
    plt.xlabel('/alpha')
    plt.close('all')
