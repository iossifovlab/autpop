from __future__ import annotations
from collections import Counter, defaultdict
from functools import partial
from typing import List, NamedTuple, Dict, Optional, Tuple, Any
from multiprocessing import Pool
import numpy as np
import pylab as pl
import scipy.stats as stats
from matplotlib.patches import Rectangle
import pathlib
import sys
from .cartisian import cartesian_product_pp, cartesian_product_itertools
import yaml
import argparse
import os


class PopulationVariant(NamedTuple):
    f: float
    w: float


class Model:
    def __init__(self, model_name: str,
                 male_threshold: float,
                 female_threshold: float,
                 population_variants: List[PopulationVariant]):
        self.model_name = model_name
        self.male_threshold = male_threshold
        self.female_threshold = female_threshold
        self.population_variants = population_variants
        self.family_types: List[FamilyType] = []

    def describe(self, F=sys.stdout):
        print(f"MODEL {self.model_name}", file=F)
        print(f"\tmale_threshold: {self.male_threshold}", file=F)
        print(f"\tfemale_threshold: {self.female_threshold}", file=F)
        print(
            f"\tpopulation variants: {len(self.population_variants)}", file=F)
        pv_cnts = Counter([(pv.f, pv.w) for pv in self.population_variants])
        for (f, w), n in pv_cnts.items():
            print(f"\t\tf: {f}, w: {w}, n: {n}", file=F)

        NF = sum([ft.unascertained_weight for ft in self.family_types])

        print(
            f"\tNumber of families: {NF}", file=F)
        print(f"\tNumber of family types: {len(self.family_types)}", file=F)

        n_hets_mean = sum([ft.get_number_of_hets()*ft.unascertained_weight
                           for ft in self.family_types]) / NF
        n_hets_max = max([ft.get_number_of_hets() for ft in self.family_types])
        print(f"\tAverage number of hets: {n_hets_mean}", file=F)
        print(f"\tMaximum number of hets: {n_hets_max}", file=F)

    @staticmethod
    def model_def_to_model(model_def) -> Model:
        assert len(model_def) == 1
        assert "threshold_model" in model_def
        data = model_def["threshold_model"]
        pvs = []
        for locus_def in data['loci']:
            n = locus_def.get("n", 1)
            pvs += [PopulationVariant(locus_def["f"], locus_def["w"])] * n

        return Model(data["name"],
                     data["male_threshold"],
                     data["female_threshold"],
                     pvs)

    @staticmethod
    def load_models_file(file_name: str) -> List[Model]:
        with open(file_name) as F:
            model_defs = yaml.safe_load_all(F)
            return [Model.model_def_to_model(model_def)
                    for model_def in model_defs]

    @staticmethod
    def load(model_name: str) -> Model:

        if model_name == "suppressor":
            return Model(model_name, 14, 19,
                         [PopulationVariant(0.025, 1)] * 160 +
                         [PopulationVariant(0.025, 2)] * 40 +
                         [PopulationVariant(0.025, 4)] * 20 +
                         [PopulationVariant(0.01, -10)] * 150)

        if model_name == "suppressor_v2":
            return Model(model_name, 30, 35,
                         [PopulationVariant(0.025, 1)] * 640 +
                         [PopulationVariant(0.01, -10)] * 150)

        if model_name == "no suppressor":
            return Model(model_name, 25, 28,
                         [PopulationVariant(0.025, 1)] * 160 +
                         [PopulationVariant(0.025, 2)] * 40 +
                         [PopulationVariant(0.025, 4)] * 20 +
                         [PopulationVariant(0.01, -0.001)] * 150)

        if model_name == "no suppressor_v2":
            return Model(model_name, 42, 45,
                         [PopulationVariant(0.025, 1)] * 640 +
                         [PopulationVariant(0.01, -0.001)] * 150)

        if model_name == "debug":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.2, 1)] * 5 +
                         [PopulationVariant(0.1, -10)] * 1)

        if model_name == "abba_original":
            return Model(model_name, 12, 13, [PopulationVariant(0.377, 1)] * 10)

        if model_name == "abba_v2":
            return Model(model_name, 10, 11, [PopulationVariant(0.7, 1)] * 6)

        if model_name == "abba_v3":
            return Model(model_name, 5, 8,
                         [PopulationVariant(0.7, 1)] * 6 +
                         [PopulationVariant(0.01, -10)] * 150)

        if model_name == "abba_v4":
            return Model(model_name, 7, 8.5,
                         [PopulationVariant(0.7, 0.5)] * 12 +
                         [PopulationVariant(0.01, -10)] * 75)

        if model_name == "abba_toy":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.33, 1)] * 1)

        if model_name == "abba_toy2":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.66, 1)] * 1)

        if model_name == "abba_toy2_A":
            return Model(model_name, -1, 0,
                         [PopulationVariant(1-0.66, -1)] * 1)

        if model_name == "suppressor_toy":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.33, 1)] * 1 +
                         [PopulationVariant(0.8, -10)] * 1)

        if model_name == "suppressor_toy2":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.66, 1)] * 1 +
                         [PopulationVariant(0.7, -10)] * 1)

        if model_name == "suppressor_toy3":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.66, 1)] * 1 +
                         [PopulationVariant(0.5, -10)] * 1)

        if model_name == "suppressor_toy4":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.4, 1)] * 1 +
                         [PopulationVariant(0.7, -10)] * 1)

        if model_name == "mike_test":
            return Model(model_name, 59, 60,
                         [PopulationVariant(0.001, 2)] * 60 +
                         [PopulationVariant(0.1, 59)] * 1)

        if model_name == "Strong, rare risk":
            return Model(model_name, 9, 11,
                         [PopulationVariant(0.05, 1)] * 40 +
                         [PopulationVariant(0.01, 8)] * 2)

        if model_name == "Uniform low, rare risk":
            return Model(model_name, 7, 8,
                         [PopulationVariant(0.05, 1)] * 40)

        if model_name == "Common risk with protection":
            return Model(model_name, 1, 2,
                         [PopulationVariant(0.4, 1),
                          PopulationVariant(0.7, -2)])

        if model_name == "Common risk with protection, and females":
            return Model(model_name, 1.15, 2,
                         [PopulationVariant(0.4, 1),
                          PopulationVariant(0.7, -15)] +
                         [PopulationVariant(0.5, 0.01)] * 2 +
                         [PopulationVariant(0.5, -0.01)] * 2)

        if model_name == "weird":
            return Model(model_name, 3.1, 4.1,
                         [PopulationVariant(0.98, 1)] * 2 +
                         [PopulationVariant(0.7, -5)] * 2 +
                         [PopulationVariant(0.5, 0.001)] * 0 +
                         [PopulationVariant(0.5, -0.001)] * 0)

        if model_name == "abba simple":
            return Model(model_name, 13.11, 14,
                         [PopulationVariant(0.75, 1)] * 7 +
                         [PopulationVariant(0.2, 0.1)])

        if model_name == "abba simple A":
            return Model(model_name, 13.11-14, 14-14,
                         [PopulationVariant(0.25, -1)] * 7 +
                         [PopulationVariant(0.2, 0.1)])

        if model_name == "wigler another nice":
            return Model(model_name, 1.99, 2,
                         [PopulationVariant(0.7, 1)] * 1 +
                         [PopulationVariant(0.03, 1.6)] * 1 +
                         [PopulationVariant(0.1, -0.01)] * 16)

        raise Exception(f"Unknown model: {model_name}")

    def generate_population_liability(self, number_of_individuals: int):
        fs = np.array([pv.f for pv in self.population_variants])
        ws = np.array([pv.w for pv in self.population_variants])
        genotypes = (np.random.rand(number_of_individuals, len(fs), 2) <
                     fs[np.newaxis, :, np.newaxis]).sum(axis=2)
        liability = genotypes.dot(ws)
        return liability

    def generate_unaffected_individuals_genotypes(self,
                                                  number_of_individuals: int,
                                                  threshold: float):
        fs = np.array([pv.f for pv in self.population_variants])
        ws = np.array([pv.w for pv in self.population_variants])

        n_done = 0
        gen_buff = []
        n_simulated = 0
        while n_done < number_of_individuals:
            genotypes = (np.random.rand(number_of_individuals, len(fs), 2) <
                         fs[np.newaxis, :, np.newaxis]).sum(axis=2)
            liability = genotypes.dot(ws)
            unaffected_idx = liability <= threshold
            gen_buff.append(genotypes[unaffected_idx, :])
            n_done += unaffected_idx.sum()
            n_simulated += number_of_individuals
            print(f"\t{n_done}/{number_of_individuals} done.")
        gens = np.vstack(gen_buff)
        return gens[:number_of_individuals, :], 1.0 - (n_done / n_simulated)

    def sample_family_types(self, number_of_family_types):
        print(f"Sampling {number_of_family_types} families...")
        print("  Generating unaffected fathers...")
        dad_gens_m, dad_initial_risk = \
            self.generate_unaffected_individuals_genotypes(
                number_of_family_types, self.male_threshold)
        print("  Generating unaffected mothers...")
        mom_gens_m, mom_initial_risk = \
            self.generate_unaffected_individuals_genotypes(
                number_of_family_types, self.female_threshold)
        ws = np.array([pv.w for pv in self.population_variants])
        family_types_raw: List[FamilyType] = []
        for dad_gens, mom_gens in zip(dad_gens_m, mom_gens_m):
            family_types_raw.append(
                FamilyType(self.male_threshold, self.female_threshold,
                           list(ws[dad_gens == 2]), list(ws[dad_gens == 1]),
                           list(ws[mom_gens == 2]), list(ws[mom_gens == 1]))
            )

        family_types: Dict[Tuple, FamilyType] = {}
        for ft in family_types_raw:
            ftk = ft.get_type_key()
            if ftk in family_types:
                family_types[ftk].unascertained_weight += 1
            else:
                family_types[ftk] = ft
        self.family_types = sorted(
            family_types.values(),
            key=lambda ft: ft.unascertained_weight,
            reverse=True)
        self.dad_initial_risk = dad_initial_risk
        self.mom_initial_risk = mom_initial_risk
        print(f"Generated {len(self.family_types)} family types.")

    def get_number_of_genotype_types(self):
        ns = np.array(Counter([(pv.w, pv.f)
                      for pv in self.population_variants]).values())
        return ((ns+1)*(ns+2)//2).prod()

    def generate_family_types(self, genotype_type_number=10_000,
                              family_number=100_000, warn=False) -> bool:
        locus_types = np.array(
            [(w, f, n) for (w, f), n in
             Counter([(pv.w, pv.f)
                      for pv in self.population_variants]).items()])
        ws = locus_types[:, 0]
        fs = locus_types[:, 1]
        ns = np.array(locus_types[:, 2], dtype=int)

        n_genotypes = ((ns+1)*(ns+2)//2).prod()
        print(f"  There are {n_genotypes} genotypes.")

        if n_genotypes > genotype_type_number:
            if warn:
                print(f"WARNING: too many genotype types {n_genotypes}.",
                      file=sys.stderr)
            else:
                raise Exception("Too many genotype types {n_genotypes}.")

        p_arrays = []
        g_arrays = []

        W = np.array([(2*w, w) for w in ws])

        for n, f in zip(ns, fs):
            A = []
            for i in range(n+1):
                for j in range(n+1-i):
                    A.append((i, j, n-i-j))
            A = np.array(A)
            P = stats.multinomial.pmf(A, n, [f*f, 2*f*(1-f), (1-f)*(1-f)])
            p_arrays.append(P)
            g_arrays.append(A[:, :2])

        G = cartesian_product_itertools(g_arrays)
        P = cartesian_product_itertools(p_arrays).prod(axis=1)

        liability = (G*W).sum(axis=(1, 2))
        dad_idx = liability <= self.male_threshold
        mom_idx = liability <= self.female_threshold

        dad_ps = P[dad_idx] / P[dad_idx].sum()
        mom_ps = P[mom_idx] / P[mom_idx].sum()

        self.dad_initial_risk = 1.0 - P[dad_idx].sum()
        self.mom_initial_risk = 1.0 - P[mom_idx].sum()

        n_families = dad_idx.sum() * mom_idx.sum()

        print(f"  There are {dad_idx.sum()} unaffected dad genotypes.")
        print(f"  There are {mom_idx.sum()} unaffected mom genotypes.")
        print(f"  There are {n_families} unaffected families.")

        if n_families > family_number:
            if warn:
                print(f"WARNING: too may family types {n_families}",
                      file=sys.stderr)
            else:
                raise Exception(f"Too many family types {n_families}")
        print(f"Generating all {n_families} family types...")

        def g_to_hom_het(g):
            hom = []
            het = []
            for w, (hm, ht) in zip(ws, g):
                hom += [w]*hm
                het += [w]*ht
            return hom, het

        family_types: Dict[Tuple, FamilyType] = {}
        for dad_p, dad_g in zip(dad_ps, G[dad_idx]):
            dad_hom, dad_het = g_to_hom_het(dad_g)
            for mom_p, mom_g in zip(mom_ps, G[mom_idx]):
                mom_hom, mom_het = g_to_hom_het(mom_g)

                ft = FamilyType(self.male_threshold, self.female_threshold,
                                dad_hom, dad_het, mom_hom, mom_het)
                ft.unascertained_weight = dad_p * mom_p
                assert ft.get_type_key() not in family_types
                family_types[ft.get_type_key()] = ft

        self.family_types = sorted(
            family_types.values(),
            key=lambda ft: ft.unascertained_weight,
            reverse=True)

        return True

    def generate_family_type_statistics(self, number_of_children_per_family):
        max_hets = max([ft.get_number_of_hets() for ft in self.family_types])
        transmission = np.random.randint(
            2, size=(number_of_children_per_family, max_hets),
            dtype=np.byte)

        with Pool() as p:
            stats = p.map(
                partial(
                    FamilyType.measure_stats,
                    sim_children_number=number_of_children_per_family,
                    transmission_buff=transmission),
                self.family_types)
        '''
        stats=[]
        for fti, ft in enumerate(self.family_types):
            if (fti % 100) == 0:
                print("Measuring stats for family type "
                      f"{fti}/{len(self.family_types)}...")
            stats.append(
                ft.measure_stats(number_of_children_per_family,
                                 transmission_buff=transmission))
        '''
        return stats

    def compute_stats(self, family_mode="dynamic",
                      genotype_type_number=10_000, family_number=100_000,
                      family_stats_mode="dynamic",
                      children_number=100_000,
                      n_processes=None):
        '''
            family_mode=all
                Generate all family types. Warn if the number of posible
                genotypes is larger than genotype_type_numaber of if the
                number of families is larger than family_number.
            family_mode=sample
                Sampe family_number parameters
            family_mode=dynamic
                If the number of families is less than family_number, generate
                all families.
                Otherwise, sample family_number families
            family_number=1_000_000


            family_stats_mode=all
                Generate all children. Warn if the number of children is
                larger than children number.
            family_stats_mode=sample
                Sample children_number children.
            family_stats_mode=dynamic
                If the number of posible children types is less than
                children_number, generate all children.
                Otherwise, sample children_number children.
        '''

        # generate family types
        if family_mode == "all":
            self.generate_family_types(
                genotype_type_number, family_number, warn=True)
        elif family_mode == "sample":
            self.sample_family_types(family_number)
        elif family_mode == "dynamic":
            try:
                self.generate_family_types(
                    genotype_type_number, family_number, warn=False)
            except Exception:
                self.sample_family_types(family_number)
        else:
            raise Exception(f"Unkown family_mode {family_mode}. The known "
                            f"family_modes are all, sample, and dynamic.")

        if children_number == "all":
            transmission = None
        else:
            max_hets = max([ft.get_number_of_hets()
                            for ft in self.family_types])
            transmission = np.random.randint(
                2, size=(children_number, max_hets),
                dtype=np.byte)
        if n_processes == 1:
            stats = []
            for fti, ft in enumerate(self.family_types):
                if (fti % 100) == 0:
                    print("Measuring stats for family type "
                          f"{fti}/{len(self.family_types)}...")
                stats.append(
                    ft.measure_stats(family_stats_mode, children_number,
                                     transmission_buff=transmission))
        else:
            with Pool() as p:
                stats = p.map(
                    partial(
                        FamilyType.measure_stats,
                        family_stats_mode=family_stats_mode,
                        children_number=children_number,
                        transmission_buff=transmission),
                    self.family_types)
        add_acertainment_stats(stats, family_number)
        global_stats = compute_global_stats(stats, self)
        return stats, global_stats


class FamilyType:
    def __init__(self,
                 male_threshold: float, female_threshold: float,
                 dad_hom_ws: List[float] = [],
                 dad_het_ws: List[float] = [],
                 mom_hom_ws: List[float] = [],
                 mom_het_ws: List[float] = [],
                 ):

        self.male_threshold = male_threshold
        self.female_threshold = female_threshold
        self.dad_hom_ws = dad_hom_ws
        self.dad_het_ws = dad_het_ws
        self.mom_hom_ws = mom_hom_ws
        self.mom_het_ws = mom_het_ws

        self.homozygous_damage_mom = sum(mom_hom_ws)
        self.homozygous_damage_dad = sum(dad_hom_ws)

        self.unascertained_weight = 1

    def get_type_key(self) -> Tuple:
        return tuple([",".join(map(str, ws)) for ws in [self.dad_hom_ws,
                                                        self.dad_het_ws,
                                                        self.mom_hom_ws,
                                                        self.mom_het_ws]])

    def get_number_of_hets(self):
        return len(self.dad_het_ws) + len(self.mom_het_ws)

    def get_number_of_child_types(self):
        types_n = np.array(
            [n for _, n in Counter(self.dad_het_ws).items()] +
            [n for _, n in Counter(self.mom_het_ws).items()])

        return (types_n+1).prod()

    def measure_stats_analytical(self):

        if len(self.mom_het_ws) + len(self.dad_het_ws) == 0:
            liability = self.homozygous_damage_mom + self.homozygous_damage_dad
            male_risk = 1.0 if liability > self.male_threshold else 0.0
            female_risk = 1.0 if liability > self.female_threshold else 0.0

            return {
                'family_type_key': self.get_type_key(),
                'unascertained_weight': self.unascertained_weight,
                'dad_hom_ws': ",".join(map(str, self.dad_hom_ws)),
                'dad_het_ws': ",".join(map(str, self.dad_het_ws)),
                'mom_hom_ws': ",".join(map(str, self.mom_hom_ws)),
                'mom_het_ws': ",".join(map(str, self.mom_het_ws)),
                'male_risk': male_risk,
                'female_risk': female_risk,
                'ma_ma_mom_SLC_mean': 0,
                'ma_ma_dad_SLC_mean': 0,
                'mu_ma_mom_SLC_mean': 0,
                'mu_ma_dad_SLC_mean': 0
            }

        het_types = np.array(
            [(1, w, n) for w, n in Counter(self.dad_het_ws).items()] +
            [(0, w, n) for w, n in Counter(self.mom_het_ws).items()])
        # n_child_types = (types_n+1).prod()

        mom_type = het_types[:, 0] == 0
        dad_type = het_types[:, 0] == 1
        types_ws = het_types[:, 1]
        types_n = np.array(het_types[:, 2], dtype=int)
        p_arrays = []
        g_arrays = []

        for p, w, n in het_types:
            gs = np.arange(n+1)
            g_arrays.append(gs)
            p_arrays.append(stats.binom.pmf(gs, n, 0.5))
        GS = cartesian_product_pp(g_arrays)
        PS = cartesian_product_pp(p_arrays).prod(axis=1)

        liability = (GS * types_ws).sum(axis=1) + \
            self.homozygous_damage_mom + self.homozygous_damage_dad
        male_risk = PS[liability > self.male_threshold].sum()
        female_risk = PS[liability > self.female_threshold].sum()

        def compute_fs(idx):
            if all(np.logical_not(idx)):
                return np.zeros(GS.shape[1])
            GSS = GS[idx, :]
            PSS = PS[idx]

            fs = (GSS / types_n * PSS[np.newaxis].T).sum(axis=0) / PSS.sum()
            return fs

        ma_fs = compute_fs(liability > self.male_threshold)
        mu_fs = compute_fs(liability <= self.male_threshold)

        ma_ma_sharing = (ma_fs**2 + (1-ma_fs)**2) * types_n
        mu_ma_sharing = (ma_fs*mu_fs + (1-ma_fs)*(1-mu_fs)) * types_n

        MLN = (mom_type * types_n).sum()
        DLN = (dad_type * types_n).sum()

        return {
            'family_type_key': self.get_type_key(),
            'unascertained_weight': self.unascertained_weight,
            'dad_hom_ws': ",".join(map(str, self.dad_hom_ws)),
            'dad_het_ws': ",".join(map(str, self.dad_het_ws)),
            'mom_hom_ws': ",".join(map(str, self.mom_hom_ws)),
            'mom_het_ws': ",".join(map(str, self.mom_het_ws)),
            'male_risk': male_risk,
            'female_risk': female_risk,
            'ma_ma_mom_SLC_mean': 2*ma_ma_sharing[mom_type].sum() - MLN,
            'ma_ma_dad_SLC_mean': 2*ma_ma_sharing[dad_type].sum() - DLN,
            'mu_ma_mom_SLC_mean': 2*mu_ma_sharing[mom_type].sum() - MLN,
            'mu_ma_dad_SLC_mean': 2*mu_ma_sharing[dad_type].sum() - DLN
        }

    def measure_stats(self, family_stats_mode="dynamic",
                      children_number: int = 100_000,
                      transmission_buff: Optional[np.ndarray] = None):
        if family_stats_mode == "all":
            if self.get_number_of_child_types() > children_number:
                print(f"WARNING: Family {self.get_type_key()} has "
                      f"{self.get_number_of_child_types()}, more than "
                      f"{children_number}", file=sys.stderr)
            return self.measure_stats_analytical()
        elif family_stats_mode == "sample":
            return self.measure_stats_by_simulation(
                children_number, transmission_buff)
        elif family_stats_mode == "dynamic":
            if self.get_number_of_child_types() < children_number:
                return self.measure_stats_analytical()
            else:
                return self.measure_stats_by_simulation(
                    children_number, transmission_buff)
        else:
            raise Exception(f"Unknown family_stats_mode {family_stats_mode}."
                            f"The family_stats_mode should be all, sample, "
                            f"or dynamic.")

    def measure_stats_by_simulation(
            self, sim_children_number: int,
            transmission_buff: Optional[np.ndarray] = None):
        ws = np.array(self.dad_het_ws + self.mom_het_ws)
        dad_locus = np.zeros(len(ws), dtype=bool)
        dad_locus[:len(self.dad_het_ws)] = True
        mom_locus = np.logical_not(dad_locus)

        if transmission_buff is None:
            transmission = np.random.randint(
                2, size=(sim_children_number, len(ws)))
        else:
            assert transmission_buff.shape[0] == sim_children_number
            assert transmission_buff.shape[1] >= len(ws)
            transmission = transmission_buff[:, :len(ws)]

        liability = (transmission * ws).sum(axis=1) + \
            self.homozygous_damage_mom + self.homozygous_damage_dad

        idx = {
            "MU": liability <= self.male_threshold,
            "MA": liability > self.male_threshold,
            "FU": liability <= self.female_threshold,
            "FA": liability > self.female_threshold
        }
        idx_i = {k: np.where(v)[0] for k, v in idx.items()}

        if len(idx_i['MA']):
            ma_fs = transmission[idx["MA"], :].mean(axis=0)
        else:
            ma_fs = np.zeros(transmission.shape[1])

        # print("ma_fs", ma_fs)
        if len(idx_i['MU']):
            mu_fs = transmission[idx["MU"], :].mean(axis=0)
        else:
            mu_fs = np.zeros(transmission.shape[1])

        male_risk = len(idx_i['MA']) / sim_children_number
        female_risk = len(idx_i['FA']) / sim_children_number

        ma_ma_sharing = ma_fs**2 + (1-ma_fs)**2
        mu_ma_sharing = ma_fs*mu_fs + (1-ma_fs)*(1-mu_fs)

        MLN = mom_locus.sum()
        DLN = dad_locus.sum()

        return {
            'family_type_key': self.get_type_key(),
            'unascertained_weight': self.unascertained_weight,
            'dad_hom_ws': ",".join(map(str, self.dad_hom_ws)),
            'dad_het_ws': ",".join(map(str, self.dad_het_ws)),
            'mom_hom_ws': ",".join(map(str, self.mom_hom_ws)),
            'mom_het_ws': ",".join(map(str, self.mom_het_ws)),
            'male_risk': male_risk,
            'female_risk': female_risk,
            'ma_ma_mom_SLC_mean': 2*ma_ma_sharing[mom_locus].sum() - MLN,
            'ma_ma_dad_SLC_mean': 2*ma_ma_sharing[dad_locus].sum() - DLN,
            'mu_ma_mom_SLC_mean': 2*mu_ma_sharing[mom_locus].sum() - MLN,
            'mu_ma_dad_SLC_mean': 2*mu_ma_sharing[dad_locus].sum() - DLN
        }


def compute_stats_models(models: List[Model], family_mode="dynamic",
                         genotype_type_number=10_000, family_number=100_000,
                         family_stats_mode="dynamic",
                         children_number=100_000,
                         n_processes=None):
    R = []
    for model in models:
        _, global_stats = model.compute_stats(
            family_mode, genotype_type_number, family_number,
            family_stats_mode, children_number, n_processes)
        R.append(global_stats)
    return R


def add_acertainment_stats(all_stats, normlization_sum=100_000):
    ma_ma_ws = np.array([stats['unascertained_weight'] *
                         stats['male_risk']**2 for stats in all_stats])
    ma_ma_ws /= ma_ma_ws.sum()
    ma_ma_ws *= normlization_sum
    for stats, ma_ma_w in zip(all_stats, ma_ma_ws):
        stats['ma_ma_weight'] = ma_ma_w

    mu_ma_ws = np.array([stats['unascertained_weight'] *
                         2*stats['male_risk']*(1-stats['male_risk'])
                         for stats in all_stats])
    mu_ma_ws /= mu_ma_ws.sum()
    mu_ma_ws *= normlization_sum
    for stats, mu_ma_w in zip(all_stats, mu_ma_ws):
        stats['mu_ma_weight'] = mu_ma_w


def save_stats_wigler(all_stats, file_name: Optional[pathlib.Path] = None):
    if file_name:
        F = open(file_name, 'w')
    else:
        F = sys.stdout
    # hcs = list(all_stats[0].keys())
    hcs = "unascertained_weight,male_risk,female_risk,ma_ma_mom_SLC_mean," + \
        "ma_ma_dad_SLC_mean,mu_ma_mom_SLC_mean,mu_ma_dad_SLC_mean"
    # "dad_total_load,dad_pos_load,dad_neg_load,dad_max_pos_het,dad_min_neg_het," + \
    # "mom_total_load,mom_pos_load,mom_neg_load,mom_max_pos_het,mom_min_neg_het"
    hcs = hcs.split(",")

    hcsO = "unascertainedWeight,maleRisk,femaleRisk,mamaMomSLC," + \
        "mamaDadSLC,mumaMomSLC,mumaDadSLC"
    # "dadTotalLoad,dadPosLoad,dadNegLoad,dadMaxPosHet,dadMinNegHet," + \
    # "momTotalLoad,momPosLoad,momNegLoad,momMaxPosHet,momMinNegHet"
    hcsO = hcsO.split(",")

    print("\t".join(hcsO), file=F)

    for stats in all_stats:
        cs: List[Any] = [stats[a] for a in hcs]
        print("\t".join(map(str, cs)), file=F)
    if file_name:
        F.close()


def save_stats(all_stats, file_name: Optional[pathlib.Path] = None):
    if file_name:
        F = open(file_name, 'w')
    else:
        F = sys.stdout
    hcs = list(all_stats[0].keys())
    print("\t".join(hcs), file=F)

    for stats in all_stats:
        cs: List[Any] = [stats[a] for a in hcs]
        print("\t".join(map(str, cs)), file=F)
    if file_name:
        F.close()


def compute_global_stats(all_stats, model: Model):
    global_stats = defaultdict(dict)

    global_stats['Model']['name'] = model.model_name
    global_stats['Model']['male threshold'] = model.male_threshold
    global_stats['Model']['female threshold'] = model.female_threshold
    global_stats['Model']['population variants number'] = len(
        model.population_variants)
    pv_cnts = Counter([(pv.f, pv.w) for pv in model.population_variants])
    for pvi, ((f, w), n) in enumerate(pv_cnts.items()):
        global_stats['Model'][f'population variant class {pvi}'] = f"w={w}, f={f}, n={n}"

    NF = float(sum([ft.unascertained_weight for ft in model.family_types]))
    global_stats['Model']['number of families'] = NF
    global_stats['Model']['number of family types'] = len(model.family_types)

    global_stats['Initial']['male risk'] = float(model.dad_initial_risk)
    global_stats['Initial']['female risk'] = float(model.mom_initial_risk)

    def weighted_average(att, w_att):
        w_sum = sum([stats[w_att] for stats in all_stats])
        w_ave = sum([stats[att]*stats[w_att]
                     for stats in all_stats if stats[w_att] > 0]) / w_sum
        return float(w_ave)

    for ft_str, ft_pref in [
            ("Unascertained", None),
            ("Families with two affected boys", 'ma_ma'),
            ("Families with one affected one unaffected boy", 'mu_ma')]:
        w_att = 'unascertained_weight'if ft_pref is None else f'{ft_pref}_weight'

        male_risk = weighted_average('male_risk', w_att)
        female_risk = weighted_average('female_risk', w_att)
        global_stats[ft_str]['male risk'] = male_risk
        global_stats[ft_str]['female risk'] = female_risk
        if ft_str == "Unascertained":
            global_stats[ft_str]['two boys, both affected, proportion'] = \
                float(male_risk * male_risk)
            global_stats[ft_str]['two boys, one affected, proportion'] = \
                float(2 * male_risk * (1-male_risk))
            global_stats[ft_str]['two boys, none affected, proportion'] = \
                float((1-male_risk) * (1-male_risk))
        else:
            global_stats[ft_str]['sharing of the father'] = weighted_average(
                f'{ft_pref}_dad_SLC_mean', w_att)
            global_stats[ft_str]['sharing of the mother'] = weighted_average(
                f'{ft_pref}_mom_SLC_mean', w_att)
    return global_stats


def draw_fancy_scatter(all_stats, x_att, y_att, w_att, gs, SMI, SMJ):
    fig = pl.gcf()

    main_ax = fig.add_axes(gs.get_spec(SMI, SMJ, tp='main'))
    x_ax = fig.add_axes(gs.get_spec(SMI, SMJ, tp='x_margin'))
    y_ax = fig.add_axes(gs.get_spec(SMI, SMJ, tp='y_margin'))

    X = np.array([stats[x_att] for stats in all_stats])
    Y = np.array([stats[y_att] for stats in all_stats])
    W = np.array([stats[w_att] for stats in all_stats])
    MX = max(X.max(), Y.max())
    MN = min(X.min(), Y.min())
    MX += (MX-MN)/10000
    BN = 30
    D = (MX-MN)/BN
    cnt_2d = np.zeros((BN, BN))
    X_I = (X-MN)//D
    Y_I = (Y-MN)//D
    for x_i, y_i, w in zip(X_I, Y_I, W):
        cnt_2d[int(x_i), int(y_i)] += w

    MMX = cnt_2d.max()
    for x_i, y_i in zip(*np.nonzero(cnt_2d)):
        w = cnt_2d[x_i, y_i]
        # print(x_i, y_i, (MN+x_i*D, MN+y_i*D))

        rp = max(0, 0.2 + 0.8*np.log10(w)/np.ceil(np.log10(MMX)))
        rp = min(rp, 1.0)
        c = [1, 1-rp, 1-rp]
        main_ax.add_patch(Rectangle((MN+x_i*D, MN+y_i*D), D, D, fc=c))
    # pl.scatter(SX, SY, s=SS)
    main_ax.set_xlim([MN, MX])
    main_ax.set_ylim([MN, MX])
    main_ax.set_xlabel(x_att)
    main_ax.set_ylabel(y_att)

    bns = np.linspace(MN, MX, BN+1)

    x_ax.hist(X, bns, weights=W)
    x_ax.set_xticks([])
    x_ax.set_xlim([MN, MX])
    x_ax.set_yticks([1, 10, 100, 1000, 10000])
    x_ax.set_yscale('log')
    X_MN = (X * W).sum() / W.sum()
    x_ax.plot([X_MN, X_MN], x_ax.get_ylim())
    x_ax.text(x_ax.get_xlim()[1], x_ax.get_ylim()
              [1], f"{X_MN:.3f}", ha="right", va="top")

    y_ax.hist(Y, bns, weights=W, orientation='horizontal')
    y_ax.set_ylim([MN, MX])
    y_ax.set_yticks([])
    y_ax.set_xticks([1, 10, 100, 1000, 10000])
    y_ax.set_xscale('log')
    Y_MN = (Y * W).sum() / W.sum()
    y_ax.plot(y_ax.get_xlim(), [Y_MN, Y_MN])
    y_ax.text(y_ax.get_xlim()[1], y_ax.get_ylim()
              [1], f"{Y_MN:.2f}", ha="right", va="top")


class MyGS:
    def __init__(self, NX: int, NY: int,
                 TM: float = 0.1, BM: float = 0.1,
                 LM: float = 0.1, RM: float = 0.1):
        self.NX = NX
        self.NY = NY
        self.TM = TM
        self.BM = BM
        self.LM = LM
        self.RM = RM

    def get_spec(self, X: int, Y: int, tp: str,
                 A: float = 0.3, B: float = 0.3):
        assert X < self.NX
        assert Y < self.NY
        PW = (1 - self.LM - self.RM) / self.NX
        PH = (1 - self.TM - self.BM) / self.NY
        PL = self.LM + X * PW
        PB = self.BM + (self.NY-Y-1) * PH

        # print(f"DDD: PW: {PW}, PH: {PH}, PL: {PL}, PB: {PB}")
        if tp == 'main':
            return (PL + PW*A, PB + PH*A, PW * (1-A-B), PH * (1-A-B))
        if tp == 'x_margin':
            return (PL + PW*A, PB + PH*(1-B), PW * (1-A-B), PH * B)
        if tp == 'y_margin':
            return (PL + PW*(1-B), PB + PH*A, PW * B, PH * (1-A-B))
        else:
            raise Exception("Unknown type " + tp)


def draw_population_liability(model: Model, ax):
    pop_liability = model.generate_population_liability(10000)
    pop_liability_values, pop_liability_counts = np.unique(
        pop_liability, return_counts=True)
    ax.plot(pop_liability_values, pop_liability_counts/len(pop_liability),
            'k.-', label='population liability')
    yl = pl.ylim()
    ax.plot([model.male_threshold]*2, yl, 'b', label='male threshold')
    ax.plot([model.female_threshold]*2, yl, 'r', label='female threshold')
    ax.legend()


def draw_dashboard(model: Model, all_stats,
                   figure_file_name: Optional[pathlib.Path] = None,
                   show: bool = False):
    fig = pl.figure()
    gs = MyGS(2, 3)
    draw_fancy_scatter(all_stats, "male_risk",
                       "female_risk", "unascertained_weight", gs, 0, 0)

    liability_ax = pl.gcf().add_axes(gs.get_spec(1, 0, tp='main', A=0.2, B=0))
    draw_population_liability(model, liability_ax)

    draw_fancy_scatter(all_stats, "male_risk",
                       "female_risk", "ma_ma_weight", gs, 0, 1)
    draw_fancy_scatter(all_stats, "ma_ma_dad_SLC_mean",
                       "ma_ma_mom_SLC_mean", "ma_ma_weight", gs, 1, 1)
    draw_fancy_scatter(all_stats, "male_risk",
                       "female_risk", "mu_ma_weight", gs, 0, 2)
    draw_fancy_scatter(all_stats, "mu_ma_dad_SLC_mean",
                       "mu_ma_mom_SLC_mean", "mu_ma_weight", gs, 1, 2)
    fig.suptitle(f'Model {model.model_name}')
    if figure_file_name:
        fig.set_size_inches(6, 10)
        fig.savefig(figure_file_name)
    if show:
        pl.show(block=False)


def save_global_stats(global_stats,
                      file_name: Optional[pathlib.Path] = None):
    if file_name:
        F = open(file_name, "w")
    else:
        F = sys.stdout

    for section in global_stats.keys():
        print(file=F)
        print(section, file=F)
        for param in global_stats[section].keys():
            print(f"  {param}:\t{global_stats[section][param]}", file=F)

    if file_name:
        F.close()


def save_global_stats_buff(global_stats_buff,
                           file_name: Optional[pathlib.Path] = None):
    if file_name:
        F = open(file_name, "w")
    else:
        F = sys.stdout

    for section in global_stats_buff[0].keys():
        print(file=F)
        print(section, file=F)
        for param in global_stats_buff[0][section].keys():
            cs = [f"  {param}:"] + \
                [f'{gs[section][param]}' for gs in global_stats_buff]
            # [f'{gs[section][param]:.3f}' for gs in global_stats_buff]
            print("\t".join(cs), file=F)

    if file_name:
        F.close()


def save_global_stats_buff_summary(global_stats_buff,
                                   file_name: Optional[pathlib.Path] = None):
    if file_name:
        F = open(file_name, "w")
    else:
        F = sys.stdout

    for section in global_stats_buff[0].keys():
        print(file=F)
        print(section, file=F)
        for param in global_stats_buff[0][section].keys():
            vs = np.array([gs[section][param] for gs in global_stats_buff])
            if isinstance(vs[0], str):
                v, = set(vs)
                print(f"  {param}: {v}", file=F)
            else:
                print(f"  {param}:\t{vs.mean():.3f}\t{vs.std():.5f}", file=F)

    if file_name:
        F.close()


def save_global_stats_table(GSB, file_name: Optional[pathlib.Path] = None,
                            prec: Optional[int] = 3):
    if file_name:
        F = open(file_name, "w")
    else:
        F = sys.stdout

    hls = [
        ",Model Definition,,,,,Initial,,All Families,,Multiplex,,,,Simplex",
        ",threshold,,loci,,,risk,,risk,,risk,,SCL,,risk,,anti-SCL",
        "Model Name,male,female,weight,frequency,number,"
        "male,female,male,female,male,"
        "female,paternal,maternal,male,female,paternal,maternal"
    ]
    for hl in hls:
        print("\t".join(hl.split(",")), file=F)

    for GS in GSB:
        for pvi in range(int(GS['Model']['population variants number'])):
            pvK = f'population variant class {pvi}'
            if pvK not in GS['Model']:
                break
            cs = []
            if pvi == 0:
                cs += [
                    GS['Model']['name'],
                    GS['Model']['male threshold'],
                    GS['Model']['female threshold'],
                ]
            else:
                cs += [""] * 3
            cs += [bb.strip(" ").split("=")[1]
                   for bb in GS['Model'][pvK].split(",")]
            if pvi == 0:
                for section in ['Initial', 'Unascertained',
                                'Families with two affected boys',
                                'Families with one affected one unaffected boy']:
                    for gender in ['male', 'female']:
                        param = f'{gender} risk'
                        if prec:
                            cs.append(f'{GS[section][param]: .{prec}f}')
                        else:
                            cs.append(f'{GS[section][param]}')
                    if not section.startswith('Families'):
                        continue
                    for parent in ['father', 'mother']:
                        param = f'sharing of the {parent}'
                        if prec:
                            cs.append(f'{GS[section][param]: .{prec}f}')
                        else:
                            cs.append(f'{GS[section][param]}')
            print("\t".join(map(str, cs)), file=F)
    if file_name:
        F.close()


def cli(cli_args=None):
    """Provide CLI for threshold model."""
    if not cli_args:
        cli_args = sys.argv[1:]

    desc = "Threshold Model Processor"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("models_file", type=str,
                        help="The models file. It is should be a yaml file.")
    parser.add_argument("-m", "--model", nargs="*",
                        help="the model names to work on")

    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="number of runs per model")

    parser.add_argument("-fm", "--family_mode", default="dynamic", type=str,
                        help='''
            Allowed values are: 'all', 'sample', and 'dynamic' (the default).
            'all' - generate all families, but do warn if the number of posible
            genotypes is larger than <genotype_number> of if the
            number of families is larger than <family_number>.

            'sample' - sampe <family_number> parameters

            'dynamic' - if the number of genotypes is less than
            <genotype_number> and the number of families is less than
            <family_number>, generate all families.
            Otherwise, sample family_number families.''')
    parser.add_argument("-fn", "--family_number", type=int, default=100_000,
                        help="The number of families. See <family_mode> for "
                        "further description")
    parser.add_argument("-gn", "--genotype_number", type=int, default=10_000,
                        help="The number of genotypes. See family_mode for "
                        "further description")
    parser.add_argument("-cm", "--children_mode", default="dynamic", type=str,
                        help='''
            Allowed values are: 'all', 'sample', and 'dynamic' (the default).
            'all' - generate all the children for each family, but do warn
            if the number of children is larger than children_number.

            'sample' - sampe childrent_number children for each family.

            'dynamic' - if the number of children is less than children_number, 
            generate all childrent.
            Otherwise, sample children_number children''')
    parser.add_argument("-cn", "--children_number", type=int, default=1_000_000,
                        help="The number of children. See <children_mode> for "
                        "further description")

    args = parser.parse_args(cli_args)

    models = Model.load_models_file(args.models_file)
    if args.model:
        unknown_models = set(args.model) - set([m.model_name for m in models])
        if unknown_models:
            raise Exception("Unknown models: ", ",".join(unknown_models))
        modelsD = {m.model_name: m for m in models}
        models = [modelsD[mn] for mn in args.model]

    pre, _ = os.path.splitext(args.models_file)
    out_dir = pathlib.Path(pre + "_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    GSB = []
    for model in models:
        print(f"\n\nWorking on {model.model_name}...")
        for run in range(args.runs):
            print(f"RUN {run}")

            res_dir = out_dir / f"{model.model_name}.run{run}"
            res_dir.mkdir(parents=True, exist_ok=True)
            family_stats, global_stats = model.compute_stats(
                family_mode=args.family_mode, family_number=args.family_number,
                genotype_type_number=args.genotype_number,
                family_stats_mode=args.children_mode,
                children_number=args.children_number)

            save_stats(family_stats, res_dir /
                       f"family_types_{model.model_name}.txt")
            save_stats_wigler(family_stats, res_dir /
                              f"family_types_wigler_{model.model_name}.txt")
            save_global_stats(global_stats)
            save_global_stats(global_stats,
                              res_dir / f"global_stats_{model.model_name}")
            GSB.append(global_stats)
    save_global_stats_table(GSB, out_dir / "models_results.txt")
