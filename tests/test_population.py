import pytest
from utils.population import Population
from utils.choiceCharacteristics import ModalChoiceCharacteristics, ChoiceCharacteristics
import pandas as pd
import os


@pytest.fixture
def pop():
    return Population()


def test_import_population(pop):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    popdata = pd.read_csv(ROOT_DIR + "/../input-data/Population.csv")
    popgroups = pd.read_csv(ROOT_DIR + "/../input-data/PopulationGroups.csv")
    pop.importPopulation(popdata, popgroups)
    assert True
    return pop


def test_update_mode_split(pop):
    pop = test_import_population(pop)
    mcc = ModalChoiceCharacteristics([])
    mcc["auto"] = ChoiceCharacteristics(0.5, 10, 0)  # Expensive, should be preferred by high VOT
    mcc["bus"] = ChoiceCharacteristics(1.0, 0, 0)

    dis = []
    ms = []

    for di, dc in pop:
        dis.append(di)
        ms.append(dc.updateModeSplit(mcc))
        print("AAH")

    assert ms[0]["auto"] > ms[1]["auto"]  # Rich people are more likely to drive
