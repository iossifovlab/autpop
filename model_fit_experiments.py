import scipy.stats as stats
import numpy as np
import pylab as pl
from autpop.population_threshold_model import Model, PopulationVariant


scored_statistics_dists = [
    ("Unascertained", "male risk",
     stats.beta(10, 10*99)),
    ("Unascertained", "female risk",
     stats.beta(10, 10*399)),
    ("Families with two affected boys", "sharing of the father",
     stats.beta(1, 1*4/6)),
    ("Families with two affected boys", "sharing of the mother",
     stats.beta(1, 1*7/3)),
]


def score_global_stats(GS):
    score = sum([dist.logpdf(GS[section][param])
                 for section, param, dist in scored_statistics_dists])
    return score


def score_model(model: Model):
    family_stats, global_stats = model.compute_stats(
        family_mode="all", family_stats_mode="all", n_processes=1)
    return score_global_stats(global_stats)


def threshold_fit_test(m, f):
    model = Model("t", m, f,
                  [PopulationVariant(0.05, 1)] * 10 +
                  [PopulationVariant(0.01, 8)] * 2)

    _, GS = model.compute_stats(
        family_mode="all", family_stats_mode="all", n_processes=1)

    cs = [score_global_stats(GS), m, f]
    for section, param, dist in scored_statistics_dists:
        cs += [GS[section][param], dist.logpdf(GS[section][param])]
    return cs


res = []
for m in range(0, 10):
    for f in range(0, 10):
        cs = threshold_fit_test(m, f)
        print(cs)
        res.append(cs)

# AS = []
# VS = []
# for a in np.arange(0.01, 20, 0.1):
#     b = 99 * a
#     bd = stats.beta(a, b)
#     print(a, b, bd.mean(), bd.var())
#     AS.append(a)
#     VS.append(bd.var())

# pl.clf()
# xs = np.linspace(0.0001, 0.03, 100)
# for a in np.arange(1, 50, 3):
#     b = 99 * a
#     pl.plot(xs, stats.beta.pdf(xs, a, b),
#             label=f"{a} {b} {stats.beta(a,b).std():.3f}")
# pl.legend()
# pl.show(block=True)

# pl.clf()

# pl.plot(AS, np.log(np.sqrt(VS)))
