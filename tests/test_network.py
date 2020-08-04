import pandas as pd
import pytest

from utils.network import Network, NetworkFlowParams, AutoMode, BusMode, BusModeParams, TravelDemand


@pytest.fixture
def net():
    return Network(pd.DataFrame({"Length": {0: 1000}, "Dedicated": {0: True}}), 0.,
                   NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50))


def test_mfd(net):
    auto = AutoMode([net], pd.DataFrame({"VehicleSize": 1}, index=["A"]), "A")
    net.addMode(auto)
    net.MFD()
    bs1 = net.getBaseSpeed()
    net.addDensity('auto', 5.0)
    net.MFD()
    bs2 = net.getBaseSpeed()
    assert bs2 < bs1
    busParams = pd.DataFrame(
        {"VehicleSize": 1, "Headway": 300, "PassengerWait": 5, "PassengerWaitDedicated": 2., "MinStopTime": 15.,
         "PerStartCost": 2.5, "VehicleOperatingCostPerHour": 30., "StopSpacing": 300}, index=["A"])
    bus = BusMode([net], busParams, "A")
    net.addMode(bus)
    net.updateBlockedDistance()
    net.MFD()
    bs3 = net.getBaseSpeed()
    assert bs3 < bs2
    td = TravelDemand()
    td.tripStartRatePerHour = 3
    bus.updateN(td)
    net.updateBlockedDistance()
    net.MFD()
    bs4 = net.getBaseSpeed()
    assert bs4 < bs3
