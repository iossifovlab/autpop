from pprint import pprint
from autpop.population_threshold_model import compute_global_stats
from autpop.population_threshold_model import Model, LocusClass


if __name__ == "__main__":
    model = Model("Example", 9, 11,
                  [LocusClass(1, 40, 0.05),
                   LocusClass(8, 2, 0.01)]
                  )

    family_type_set, female_risk, male_risk = model.build_family_types()
    family_stats = family_type_set.compute_family_stats(n_processes=1)
    global_stats = compute_global_stats(family_stats, family_type_set,
                                        female_risk, male_risk)
    pprint(global_stats)
