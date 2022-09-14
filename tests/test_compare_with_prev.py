from autpop.population_threshold_model import Model, FamilyType
from autpop.population_threshold_model_prev import FamilyType as FamilyTypePrev

import numpy as np


def test():
    models = Model.load_models_file("tests/demos/paper_models.yaml")
    model = models[4]
    assert model.model_name.startswith("Example 5")

    def genS2gens(gensS: str):
        return np.array([[int(v) for v in gS.split(",")]
                         for gS in gensS.split(":")])

    def key2Gens(k: str):
        return map(genS2gens, k.split("|"))

    def gen2HomHets(gen):
        assert gen.shape == (2, 2)
        homs = [0.15]*gen[0, 0] + [-15]*gen[1, 0]
        hets = [0.15]*gen[0, 1] + [-15]*gen[1, 1]
        return [homs, hets]

    key = "32,8:0,3|33,7:0,3"
    momG, dadG = key2Gens(key)

    ft = FamilyType(model, momG, dadG)
    st = ft.measure_stats()
    assert st['family_type_key'] == key
    assert np.abs(st['male_risk'] - 0.010880947) < 1E-7
    # assert np.abs(st['female_risk'] - 0.000925541) < 1E-7

    homHets = gen2HomHets(dadG) + gen2HomHets(momG)
    pft = FamilyTypePrev(model.male_threshold,
                         model.female_threshold, *homHets)
    pst = pft.measure_stats()
    assert st['male_risk'] == pst['male_risk']
    assert st['female_risk'] == pst['female_risk']
