from autpop.population_threshold_model import Model, FamilyType, LocusClass
from autpop.population_threshold_model_prev import FamilyType as FamilyTypePrev

import numpy as np


def test():

    model = Model("Example 5", 10.7, 11.24, [
        LocusClass(0.15, 40, 0.9),
        LocusClass(-15, 60, 0.03)
    ])

    ft = FamilyType(model, "32,8:0,3", "33,7:0,3")
    st = ft.measure_stats()

    def gen2HomHets(gen):
        assert gen.shape == (2, 2)
        homs = [0.15]*gen[0, 0] + [-15]*gen[1, 0]
        hets = [0.15]*gen[0, 1] + [-15]*gen[1, 1]
        return [homs, hets]

    homHets = gen2HomHets(ft.mom_genotypes) + gen2HomHets(ft.dad_genotypes)
    pft = FamilyTypePrev(model.male_threshold,
                         model.female_threshold, *homHets)
    pst = pft.measure_stats()

    assert np.abs(st['male_risk'] - pst['male_risk']) < 1E-8
    assert np.abs(st['female_risk'] - pst['female_risk']) < 1E-8
