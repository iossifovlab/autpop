from autpop.population_threshold_model import Model, LocusClass
from autpop.population_threshold_model import compute_global_stats
from autpop.population_threshold_model import save_stats, save_global_stats


def test_full_stack():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])
    fts, female_ir, male_ir = model.build_family_types(all_families=True)
    all_stats = fts.compute_all_family_specific_stats()
    fts.add_family_set_specific_stats(all_stats)
    global_stats = compute_global_stats(all_stats, fts, female_ir, male_ir)
    assert global_stats
    assert global_stats['prediction_details']['precise']

    save_stats(all_stats)
    save_global_stats(global_stats)
