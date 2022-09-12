from autpop.population_threshold_model import Model, LocusClass


def test_full_stack():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])
    stats, global_stats = model.compute_stats()
    print(stats)
