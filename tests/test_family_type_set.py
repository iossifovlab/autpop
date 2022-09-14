from autpop.population_threshold_model import Model, LocusClass
from autpop.population_threshold_model import save_stats, save_global_stats
from autpop.population_threshold_model import compute_global_stats
import numpy as np


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

    save_stats(all_stats)
    assert 'pU' in all_stats[0]


def test_global_stats_with_and_without_all_families():
    model = Model("gosho", 8, 9, [LocusClass(1.0, 10, 0.05)])
    # model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    ftA = model.build_family_types(
        family_number=1_000_000, all_families=False)[0]
    AA = ftA.compute_family_stats()
    GA = compute_global_stats(AA, ftA)
    save_global_stats(GA)

    ftB = model.build_family_types(
        family_number=1_000_000, all_families=True)[0]
    AB = ftB.compute_family_stats()
    GB = compute_global_stats(AB, ftB)

    for section in ['unaffected parents families',
                    'concordant families', 'discordant families']:
        for k in GA[section].keys():
            assert np.abs(GA[section][k]-GB[section][k]) < 1E-8
