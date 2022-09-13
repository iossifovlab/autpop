from re import A
from autpop.population_threshold_model import Model, LocusClass


def test_all_families_one_locus_class():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    fts = model.build_family_types(all_families=True)[0]
    assert fts.get_family_type_number() == 9
    assert 'unaffected' not in fts.description

    stats_1 = fts.compute_all_family_specific_stats(n_processes=1)
    stats_2 = fts.compute_all_family_specific_stats(n_processes=2)
    assert stats_1 == stats_2

    fts = model.build_family_types(all_families=False)[0]
    assert fts.get_family_type_number() == 6
    assert 'unaffected' in fts.description


def test_n_processes():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    fts = model.build_family_types(all_families=False)[0]

    stats_1 = fts.compute_all_family_specific_stats(n_processes=1)
    stats_2 = fts.compute_all_family_specific_stats(n_processes=2)
    stats_all = fts.compute_all_family_specific_stats(n_processes=None)

    assert stats_1 == stats_2
    assert stats_1 == stats_all

    assert len(stats_1) == 6


def test_add_family_set_specific_stats():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    fts = model.build_family_types(all_families=True)[0]
    all_stats = fts.compute_all_family_specific_stats()
    fts.add_family_set_specific_stats(all_stats)

    assert 'pU' in all_stats[0]
