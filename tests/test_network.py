from utils.network import Network, NetworkFlowParams, AutoMode, AutoModeParams, BusMode, BusModeParams, TravelDemand
import pytest
import pandas as pd


@pytest.fixture
def net():
    return Network(pd.DataFrame({"Length": {0: 1000}}), 0., NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50))


def test_mfd(net):
    auto = AutoMode([net], AutoModeParams())
    net.addMode(auto)
    net.MFD()
    bs1 = net.getBaseSpeed()
    net.addDensity('auto', 5.0)
    net.MFD()
    bs2 = net.getBaseSpeed()
    assert bs2 < bs1
    bus = BusMode([net], BusModeParams())
    net.addMode(bus)
    net.updateBlockedDistance()
    net.MFD()
    bs3 = net.getBaseSpeed()
    assert bs3 < bs2
    td = TravelDemand()
    td.tripStartRate = 3
    bus.updateN(td)
    net.updateBlockedDistance()
    net.MFD()
    bs4 = net.getBaseSpeed()
    assert bs4 < bs3
