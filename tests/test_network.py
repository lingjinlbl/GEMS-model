from utils.network import Network, NetworkFlowParams, AutoMode, AutoModeParams, BusMode, BusModeParams, TravelDemand


def test_mfd():
    n = Network(1000.0, NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50))
    auto = AutoMode([n], AutoModeParams())
    n.addMode(auto)
    n.MFD()
    bs1 = n.getBaseSpeed()
    n.addDensity('auto', 5.0)
    n.MFD()
    bs2 = n.getBaseSpeed()
    assert bs2 < bs1
    bus = BusMode([n], BusModeParams())
    n.addMode(bus)
    n.updateBlockedDistance()
    n.MFD()
    bs3 = n.getBaseSpeed()
    assert bs3 < bs2
    td = TravelDemand()
    td.tripStartRate = 3
    bus.updateN(td)
    n.updateBlockedDistance()
    n.MFD()
    bs4 = n.getBaseSpeed()
    assert bs4 < bs3
