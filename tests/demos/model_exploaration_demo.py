import numpy as np
import pylab as pl
from autpop.population_threshold_model import Model, PopulationVariant


if __name__ == "__main__":
    GSB = []
    # for f in np.arange(0.01, 1.0, 0.02):
    #     for g in np.arange(0.01, 1.0, 0.02):
    #         model = Model(f"f:{f:.2f}, g:{g:.2f}", 1, 2,
    #                       [PopulationVariant(f, 1), PopulationVariant(g, -2)])
    #         print(model.model_name)
    #         family_stats, global_stats = model.compute_stats(n_processes=1)
    #         GSB.append(global_stats)

    for f in np.arange(0.01, 1.0, 0.02):
        model = Model(f"{f:.2f}", 1, 2,
                      [PopulationVariant(f, 1)])
        print(model.model_name)
        family_stats, global_stats = model.compute_stats(n_processes=1)
        global_stats['Model']['my f'] = f
        GSB.append(global_stats)

    pl.figure()
    f = [GS['Model']['my f'] for GS in GSB]
    u_male_risk = [GS['Unascertained']['male risk'] for GS in GSB]
    total_mama_sharing = [
        GS['Families with two affected boys']['sharing of the father'] /
        GS['Families with two affected boys']['sharing of the mother']
        for GS in GSB]
    pl.subplot(2, 1, 1)
    pl.plot(f, u_male_risk, '.')
    pl.xlabel('f')
    pl.ylabel('male risk')

    pl.subplot(2, 1, 2)
    pl.plot(f, total_mama_sharing, '.')
    pl.xlabel('f')
    pl.ylabel('total sharing')
    pl.show(block=False)
