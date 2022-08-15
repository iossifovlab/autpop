import numpy as np
import pylab as pl
from autpop.population_threshold_model import Model, PopulationVariant


if __name__ == "__main__":
    GSB = []
    for f in np.arange(0.01, 1.0, 0.02):
        for g in np.arange(0.01, 1.0, 0.02):
            model = Model(f"f:{f:.2f}, g:{g:.2f}", 1, 2,
                          [PopulationVariant(f, 1), PopulationVariant(g, -2)])
            print(model.model_name)
            family_stats, global_stats = model.compute_stats(n_processes=1)
            GSB.append(global_stats)

    pl.figure()
    u_male_risk = [GS['Unascertained']['male risk'] for GS in GSB]
    total_mama_sharing = [
        GS['Families with two affected boys']['sharing of the father'] +
        GS['Families with two affected boys']['sharing of the mother']
        for GS in GSB]
    pl.plot(u_male_risk, total_mama_sharing, '.')
    pl.show(block=True)
