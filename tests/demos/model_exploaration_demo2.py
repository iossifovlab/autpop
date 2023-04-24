import numpy as np
import pylab as pl
from autpop.population_threshold_model import Model, LocusClass
from autpop.population_threshold_model import compute_global_stats


if __name__ == "__main__":
    GSB = []

    for f in np.arange(0.01, 1.0, 0.1):
        for g in np.arange(0.01, 1.0, 0.1):
            model = Model(f"f:{f:.2f}, g:{g:.2f}", 1, 2,
                          [LocusClass(1, 1, f), LocusClass(-2, 1, g)])

            print(model.model_name)

            family_type_set, female_risk, male_risk = model.build_family_types()
            family_stats = family_type_set.compute_family_stats(n_processes=1)
            global_stats = compute_global_stats(family_stats, family_type_set,
                                                female_risk, male_risk)
            global_stats['threshold_model']['my f'] = f
            global_stats['threshold_model']['my f'] = g

            GSB.append(global_stats)

    paternal_to_maternal_sharing_ratio = [
        GS['concordant families']['paternal net SCLs'] /
        GS['concordant families']['maternal net SCLs']
        for GS in GSB]
    total_concordant_sharing = [
        GS['concordant families']['paternal net SCLs'] +
        GS['concordant families']['maternal net SCLs']
        for GS in GSB]

    pl.figure()
    pl.plot(total_concordant_sharing, paternal_to_maternal_sharing_ratio, '.')
    pl.xlabel('total concordant sharing')
    pl.ylabel('paternal to maternal sharing ratio')
    pl.show(block=False)
