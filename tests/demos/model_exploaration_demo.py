import numpy as np
import pylab as pl
from autpop.population_threshold_model import Model, LocusClass
from autpop.population_threshold_model import compute_global_stats


if __name__ == "__main__":
    GSB = []

    for f in np.arange(0.01, 1.0, 0.02):
        model = Model(f"{f:.2f}", 1, 2,
                      [LocusClass(1, 1, f)])
        print(model.model_name)

        family_type_set, female_risk, male_risk = model.build_family_types()
        family_stats = family_type_set.compute_family_stats(n_processes=1)
        global_stats = compute_global_stats(family_stats, family_type_set,
                                            female_risk, male_risk)
        global_stats['threshold_model']['my f'] = f

        GSB.append(global_stats)

    pl.figure()
    f = [GS['threshold_model']['my f'] for GS in GSB]
    u_male_risk = [GS['unaffected parents families']['male risk']
                   for GS in GSB]
    paternal_to_maternal_sharing_ratio = [
        GS['concordant families']['paternal net SCLs'] /
        GS['concordant families']['maternal net SCLs']
        for GS in GSB]
    total_concordant_sharing = [
        GS['concordant families']['paternal net SCLs'] +
        GS['concordant families']['maternal net SCLs']
        for GS in GSB]

    pl.subplot(3, 1, 1)
    pl.plot(f, u_male_risk, '.')
    pl.xlabel('f')
    pl.ylabel('male risk')

    pl.subplot(3, 1, 2)
    pl.plot(f, total_concordant_sharing, '.')
    pl.xlabel('f')
    pl.ylabel('total concordant sharing')
    pl.show(block=False)

    pl.subplot(3, 1, 3)
    pl.plot(f, paternal_to_maternal_sharing_ratio, '.')
    pl.xlabel('f')
    pl.ylabel('paternal to maternal sharing ratio')
    pl.show(block=False)
