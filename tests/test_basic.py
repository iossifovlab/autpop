from autpop.population_threshold_model import Model, LocusClass, FamilyType
import numpy as np


def test_build():
    model = Model("gosho", 1, 2, [])
    assert model.model_name == "gosho"


def test_all_individual_genotypes_one_locus_class():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    assert model.get_number_of_individual_genotypes_types() == 3

    G, P = model.generate_all_individual_genotypes()
    assert G.shape == (3, 1, 2)
    assert len(G) == 3
    assert len(P) == 3
    assert np.abs(P.sum()-1.0) < 1E-7


def test_all_individual_genotypes_two_class():
    model = Model("gosho", 1, 2,
                  [LocusClass(2.2, 1, 0.01), LocusClass(-3.1, 3, 0.01)])

    # number of genoypte is (1+1)*(1+2)/2 * (3+1)*(3+2)/2  = 30
    assert model.get_number_of_individual_genotypes_types() == 30

    G, P = model.generate_all_individual_genotypes()
    assert G.shape == (30, 2, 2)
    assert len(G) == 30
    assert len(P) == 30
    assert np.abs(P.sum()-1.0) < 1E-7


def test_sample_individual_genotypes_tree_class():
    model = Model("gosho", 1, 2,
                  [LocusClass(2.2, 1, 0.1),
                   LocusClass(-3.1, 3, 0.2),
                   LocusClass(1.1, 2, 0.6)])

    assert model.get_number_of_individual_genotypes_types() == 180
    G = model.sample_individual_genotypes(100000)
    assert G.shape == (100000, 3, 2)


def test_create_a_no_hets_family_type():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    family_type = FamilyType(model,
                             np.array([[0, 0]]),
                             np.array([[1, 0]]))
    assert family_type.homozygous_damage_mom == 0.0
    assert family_type.homozygous_damage_dad == 1.0

    assert family_type.get_type_key() == "0,0|1,0"

    assert family_type.get_number_of_hets() == 0
    assert family_type.get_number_of_child_types() == 1

    GA, PA = family_type.generate_all_child_types()
    assert (GA == [[]]).all()
    assert PA == 1.0

    GS, PS = family_type.sample_child_types(100)
    assert (GS == [[]]).all()
    assert PS == 1.0


def test_create_a_family_type_one_locus_class():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    family_type = FamilyType(model,
                             np.array([[0, 1]]),
                             np.array([[1, 0]]))
    assert family_type.homozygous_damage_mom == 0.0
    assert family_type.homozygous_damage_dad == 1.0

    assert family_type.get_type_key() == "0,1|1,0"

    assert family_type.get_number_of_hets() == 1
    assert family_type.get_number_of_child_types() == 2


def test_create_a_family_type_three_locus_class():
    model = Model("gosho", 1, 2,
                  [LocusClass(2.2, 1, 0.1),
                   LocusClass(-3.1, 3, 0.2),
                   LocusClass(1.1, 2, 0.6)])

    family_type = FamilyType(model,
                             np.array([
                                 [0, 1],
                                 [1, 2],
                                 [2, 0],
                             ]),
                             np.array([
                                 [0, 0],
                                 [0, 1],
                                 [1, 1],
                             ]))
    assert family_type.homozygous_damage_mom == -3.1 + 2*1.1
    assert family_type.homozygous_damage_dad == 1.1

    assert family_type.get_type_key() == "0,1:1,2:2,0|0,0:0,1:1,1"

    assert family_type.get_number_of_hets() == 5
    assert family_type.get_number_of_child_types() == 24


def test_generate_all_children_one_locus_class():
    model = Model("gosho", 1, 2, [LocusClass(1.0, 1, 0.01)])

    fm_A = FamilyType(model,
                      np.array([[0, 1]]),
                      np.array([[1, 0]]))

    GA_A, PA_A = fm_A.generate_all_child_types()
    assert GA_A.shape == (2, 1)
    assert np.abs(PA_A.sum()-1.0) < 1E-7

    GS_A, PS_A = fm_A.sample_child_types(10000)
    assert GS_A.shape == (2, 1)
    assert np.abs(PS_A.sum()-1.0) < 1E-7

    fm_B = FamilyType(model,
                      np.array([[0, 1]]),
                      np.array([[0, 1]]))

    GA_B, PA_B = fm_B.generate_all_child_types()
    assert GA_B.shape == (4, 2)
    assert np.abs(PA_B.sum()-1.0) < 1E-7

    GS_B, PS_B = fm_B.sample_child_types(10000)
    assert GS_B.shape == (4, 2)
    assert np.abs(PS_B.sum()-1.0) < 1E-7


def test_generate_all_children_three_locus_class():
    model = Model("gosho", 1, 2,
                  [LocusClass(2.2, 1, 0.1),
                   LocusClass(-3.1, 3, 0.2),
                   LocusClass(1.1, 2, 0.6)])

    fm = FamilyType(model,
                    np.array([
                        [0, 1],
                        [1, 2],
                        [2, 0],
                    ]),
                    np.array([
                        [0, 0],
                        [0, 1],
                        [1, 1],
                    ]))

    GA, PA = fm.generate_all_child_types()
    assert GA.shape == (24, 4)
    assert np.abs(PA.sum()-1.0) < 1E-7

    GS, PS = fm.sample_child_types(1000000)
    assert GS.shape == (24, 4)
    assert np.abs(PS.sum()-1.0) < 1E-7

    assert (GA == GS).all()
    assert np.abs(PA-PS).max() < 0.001
