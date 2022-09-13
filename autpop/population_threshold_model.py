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


class LocusClass(NamedTuple):
    w: float
    n: int = 1
    f: float = 0.01


class Model:
    def __init__(self, model_name: str,
                 male_threshold: float,
                 female_threshold: float,
                 locus_classes: List[LocusClass]):
        self.model_name = model_name
        self.male_threshold = male_threshold
        self.female_threshold = female_threshold
        self.locus_classes = locus_classes
        self.ws = np.array([lc.w for lc in self.locus_classes])
        self.W = np.array([(2*lc.w, lc.w) for lc in self.locus_classes])

    def get_number_of_loci(self):
        return sum([lc.n for lc in self.locus_classes])

    def get_number_of_individual_genotypes_types(self):
        ns = np.array([lc.n for lc in self.locus_classes])
        return ((ns+1)*(ns+2)//2).prod()

    def describe(self, F=sys.stdout):
        print(f"MODEL {self.model_name}", file=F)
        print(f"\tmale_threshold: {self.male_threshold}", file=F)
        print(f"\tfemale_threshold: {self.female_threshold}", file=F)
        print(f"\tpopulation variants: {self.get_number_of_loci()}", file=F)
        for lc in self.locus_classes:
            print(f"\t\tf: {lc.f}, w: {lc.w}, n: {lc.n}", file=F)

        '''
        NF = sum([ft.unascertained_weight for ft in self.family_types])

        print(
            f"\tNumber of families: {NF}", file=F)
        print(f"\tNumber of family types: {len(self.family_types)}", file=F)
        n_hets_mean = sum([ft.get_number_of_hets()*ft.unascertained_weight
                           for ft in self.family_types]) / NF
        n_hets_max = max([ft.get_number_of_hets() for ft in self.family_types])
        print(f"\tAverage number of hets: {n_hets_mean}", file=F)
        print(f"\tMaximum number of hets: {n_hets_max}", file=F)
        '''

    @staticmethod
    def model_def_to_model(model_def) -> Model:
        assert len(model_def) == 1
        assert "threshold_model" in model_def
        data = model_def["threshold_model"]
        locus_classes = []
        for locus_def in data['loci']:
            locus_classes.append(LocusClass(
                locus_def["w"],
                locus_def.get("n", 1),
                locus_def.get("f", 0.01)))

        return Model(data["name"],
                     data["male_threshold"],
                     data["female_threshold"],
                     locus_classes)

    @staticmethod
    def load_models_file(file_name: str) -> List[Model]:
        with open(file_name) as F:
            model_defs = yaml.safe_load_all(F)
            return [Model.model_def_to_model(model_def)
                    for model_def in model_defs]

    def generate_all_individual_genotypes(self):
        p_arrays = []
        g_arrays = []

        for lc in self.locus_classes:
            A = []
            for i in range(lc.n+1):
                for j in range(lc.n+1-i):
                    A.append((i, j, lc.n-i-j))
            A = np.array(A, dtype=int)
            P = stats.multinomial.pmf(A, lc.n, [lc.f*lc.f,
                                                2*lc.f*(1-lc.f),
                                                (1-lc.f)*(1-lc.f)])
            p_arrays.append(P)
            g_arrays.append(A[:, :2])

        G = cartesian_product_itertools(g_arrays)
        P = cartesian_product_itertools(p_arrays).prod(axis=1)

        return G, P

    def sample_individual_genotypes(self, number_of_individuals):
        by_class = []
        for lc in self.locus_classes:
            by_class.append(stats.multinomial.rvs(
                lc.n,
                [lc.f*lc.f, 2*lc.f*(1-lc.f), (1-lc.f)*(1-lc.f)],
                number_of_individuals)[:, :2])
        return np.array(by_class).transpose([1, 0, 2])

    def compute_liabilities(self, GS):
        return (GS*self.W).sum(axis=(1, 2))

    def compute_liability(self, G):
        return (G*self.W).sum()

    def generate_all_family_types(self, genotype_type_number=10_000,
                                  family_number=100_000, all_families=False,
                                  warn=False) \
            -> Tuple[FamilyTypeSet, float, float]:

        n_genotypes = self.get_number_of_individual_genotypes_types()
        if n_genotypes > genotype_type_number:
            if warn:
                print(f"WARNING: too many genotype types {n_genotypes}.",
                      file=sys.stderr)
            else:
                raise Exception("Too many genotype types {n_genotypes}.")

        G, P = self.generate_all_individual_genotypes()

        liability = self.compute_liabilities(G)

        dad_idx = liability <= self.male_threshold
        mom_idx = liability <= self.female_threshold

        dad_initial_risk = 1.0 - P[dad_idx].sum()
        mom_initial_risk = 1.0 - P[mom_idx].sum()

        if all_families:
            description = 'all families'
            momGs, dadGs = G, G
            momPs, dadPs = P, P
        else:
            description = 'all families with unaffected parents'
            momGs = G[mom_idx]
            momPs = P[mom_idx]
            dadGs = G[dad_idx]
            dadPs = P[dad_idx]
            momPs /= momPs.sum()
            dadPs /= dadPs.sum()

        n_families = len(momGs) * len(dadGs)
        print(f"  Generating {description} ({n_families}).")
        if n_families > family_number:
            if warn:
                print(f"WARNING: too may family types {n_families}",
                      file=sys.stderr)
            else:
                raise Exception(f"Too many family types {n_families}")

        family_type_set = FamilyTypeSet(self, description)
        for dad_p, dad_g in zip(dadPs, dadGs):
            for mom_p, mom_g in zip(momPs, momGs):
                ft = FamilyType(self, mom_g, dad_g)
                family_type_set.add_family_type(ft, dad_p * mom_p)

        return family_type_set, mom_initial_risk, dad_initial_risk

    def build_family_types(self, family_mode="dynamic",
                           genotype_type_number=10_000,
                           family_number=100_000,
                           all_families=False,
                           warn=True) -> Tuple[FamilyTypeSet, float, float]:
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
        '''
        if family_mode == "all":
            return self.generate_all_family_types(
                genotype_type_number, family_number,
                all_families=all_families, warn=warn)
        elif family_mode == "sample":
            self.sample_family_types(family_number)
        elif family_mode == "dynamic":
            try:
                return self.generate_all_family_types(
                    genotype_type_number, family_number,
                    all_families=all_families, warn=False)
            except Exception:
                self.sample_family_types(family_number)
        else:
            raise Exception(f"Unkown family_mode {family_mode}. The known "
                            f"family_modes are all, sample, and dynamic.")
        raise Exception('breh')

    def compute_stats(self, family_mode="dynamic",
                      genotype_type_number=10_000, family_number=100_000,
                      family_stats_mode="dynamic",
                      children_number=100_000,
                      n_processes=None):

        # generate family types
        if family_mode == "all":
            self.generate_all_family_types(
                genotype_type_number, family_number, warn=True)
        elif family_mode == "sample":
            self.sample_family_types(family_number)
        elif family_mode == "dynamic":
            try:
                self.generate_all_family_types(
                    genotype_type_number, family_number, warn=False)
            except Exception:
                self.sample_family_types(family_number)
        else:
            raise Exception(f"Unkown family_mode {family_mode}. The known "
                            f"family_modes are all, sample, and dynamic.")

        if n_processes == 1:
            stats = []
            for fti, ft in enumerate(self.family_types):
                if (fti % 100) == 0:
                    print("Measuring stats for family type "
                          f"{fti}/{len(self.family_types)}...")
                stats.append(
                    ft.measure_stats(family_stats_mode, children_number))
        else:
            with Pool() as p:
                stats = p.map(
                    partial(
                        FamilyType.measure_stats,
                        family_stats_mode=family_stats_mode,
                        children_number=children_number),
                    range(self.get_number_of_individual_genotypes_types()))
        add_acertainment_stats(stats, family_number)
        global_stats = compute_global_stats(stats, self)
        return stats, global_stats


class FamilyTypeSet:
    def __init__(self, model, description):
        self.model = model
        self.description = description
        self.family_types: List[FamilyType] = []
        self.family_type_set_probabilities = []
        self.ft_key_to_index: Dict[str, int] = {}

    def add_family_type(self, family_type: FamilyType,
                        family_type_set_probability: float):
        ftk = family_type.get_type_key()
        try:
            idx = self.ft_key_to_index[ftk]
            self.family_type_set_probabilities[idx] += \
                family_type_set_probability
        except KeyError:
            idx = len(self.family_types)
            self.ft_key_to_index[ftk] = idx
            self.family_types.append(family_type)
            self.family_type_set_probabilities.append(
                family_type_set_probability)

    def get_family_type_number(self):
        return len(self.family_types)

    def compute_all_family_specific_stats(self, family_stats_mode="dynamic",
                                          children_number=100_000, warn=True,
                                          n_processes=None):
        '''
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
        if n_processes == 1:
            stats = []
            for fti, ft in enumerate(self.family_types):
                if (fti % 100) == 0:
                    print("Measuring stats for family type "
                          f"{fti}/{self.get_family_type_number()}...")
                stats.append(ft.measure_stats(
                    family_stats_mode=family_stats_mode,
                    children_number=children_number, warn=warn))
        else:
            with Pool(n_processes) as p:
                stats = p.map(
                    partial(
                        FamilyType.measure_stats,
                        family_stats_mode=family_stats_mode,
                        children_number=children_number),
                    self.family_types)
        return stats

    def add_family_set_specific_stats(self, all_stats):

        unaffected_parents = np.array(
            [ft.are_parents_unaffected()
             for ft in self.family_types], dtype=bool)
        pU = np.array(self.family_type_set_probabilities)
        pU[np.logical_not(unaffected_parents)] = 0.0
        pU /= pU.sum()

        male_risk = np.array([stats['male_risk'] for stats in all_stats])
        pC = pU * male_risk**2
        pC /= pC.sum()

        pD = pU * 2 * male_risk * (1-male_risk)
        pD /= pD.sum()

        for fs_stats, U, C, D in zip(all_stats, pU, pC, pD):
            fs_stats['pU'] = U
            fs_stats['pC'] = C
            fs_stats['pD'] = D


class FamilyType:
    def __init__(self, model: Model, mom_genotypes, dad_genotypes):

        self.model = model

        self.mom_genotypes = mom_genotypes
        self.dad_genotypes = dad_genotypes

        self.homozygous_damage_mom = model.ws.dot(self.mom_genotypes[:, 0])
        self.homozygous_damage_dad = model.ws.dot(self.dad_genotypes[:, 0])

        het_types_list = [(1, lc.w, het_n)
                          for lc, het_n in zip(self.model.locus_classes,
                                               self.dad_genotypes[:, 1])
                          if het_n > 0] + \
            [(0, lc.w, het_n)
             for lc, het_n in zip(self.model.locus_classes,
                                  self.mom_genotypes[:, 1])
             if het_n > 0]
        if het_types_list:
            het_types = np.array(het_types_list)
        else:
            het_types = np.array(het_types_list).reshape(0, 3)

        self.het_mom = het_types[:, 0] == 0
        self.het_dad = het_types[:, 0] == 1
        self.het_ws = het_types[:, 1]
        self.het_ns = np.array(het_types[:, 2], dtype=int)

        self.mom_liability = model.compute_liability(self.mom_genotypes)
        self.dad_liability = model.compute_liability(self.dad_genotypes)

    def is_dad_affected(self):
        return self.dad_liability > self.model.male_threshold

    def is_mom_affected(self):
        return self.dad_liability > self.model.male_threshold

    def are_parents_unaffected(self):
        return (not self.is_dad_affected()) and (not self.is_mom_affected())

    def get_type_key(self) -> str:
        def g2s(gens):
            return ":".join([",".join(map(str, g)) for g in gens])
        return g2s(self.mom_genotypes) + "|" + g2s(self.dad_genotypes)

    def get_number_of_hets(self):
        return self.mom_genotypes[:, 1].sum() + self.dad_genotypes[:, 1].sum()

    def get_number_of_child_types(self):
        het_ns = np.array([self.mom_genotypes[:, 1], self.dad_genotypes[:, 1]])
        return (het_ns+1).prod()

    def generate_all_child_types(self):
        if len(self.het_ns) == 0:
            return np.array([[]], dtype=int), np.array([1.0])
        p_arrays = []
        g_arrays = []

        for n in self.het_ns:
            gs = np.arange(n+1, dtype=int)
            g_arrays.append(gs)
            p_arrays.append(stats.binom.pmf(gs, n, 0.5))
        GS = cartesian_product_pp(g_arrays)
        PS = cartesian_product_pp(p_arrays).prod(axis=1)

        return GS, PS

    def sample_child_types(self, number_of_children):
        if len(self.het_ns) == 0:
            return np.array([[]], dtype=int), np.array([1.0])
        buff = [stats.binom.rvs(n, 0.5, size=number_of_children)
                for n in self.het_ns]
        all_GS = np.array(buff).T

        GS, PS = np.unique(all_GS, axis=0, return_counts=True)
        PS = PS / number_of_children
        return GS, PS

    def measure_stats(self, family_stats_mode="dynamic",
                      children_number: int = 100_000, warn=True):

        if self.get_number_of_hets() == 0:
            liability = self.homozygous_damage_mom + self.homozygous_damage_dad
            male_risk = 1.0 if liability > self.model.male_threshold else 0.0
            female_risk = 1.0 if liability > self.model.female_threshold else 0.0

            return {
                'family_type_key': self.get_type_key(),
                'prediction_method': 'precise',
                'male_risk': male_risk,
                'female_risk': female_risk,
                'mC_mom_netSCLs': 0,
                'mC_dad_netSCLs': 0,
                'mD_mom_netSCLs': 0,
                'mD_dad_netSCLs': 0
            }

        if family_stats_mode == "all":
            if self.get_number_of_child_types() > children_number:
                err_str = f"Family {self.get_type_key()} has " + \
                    f"{self.get_number_of_child_types()}, more than " + \
                    f"{children_number}"
                if warn:
                    print(f"WARNING: {err_str}, more than "
                          f"{children_number}", file=sys.stderr)
                else:
                    raise Exception(err_str)
            prediction_method = 'precise'
        elif family_stats_mode == "sample":
            prediction_method = 'sample'
        elif family_stats_mode == "dynamic":
            if self.get_number_of_child_types() < children_number:
                prediction_method = 'precise'
            else:
                prediction_method = 'sample'
        else:
            raise Exception(f"Unknown family_stats_mode {family_stats_mode}."
                            f"The family_stats_mode should be all, sample, "
                            f"or dynamic.")

        if prediction_method == 'precise':
            GS, PS = self.generate_all_child_types()
        else:
            GS, PS = self.sample_child_types(children_number)

        liability = (GS * self.het_ws).sum(axis=1) + \
            self.homozygous_damage_mom + self.homozygous_damage_dad
        male_risk = PS[liability > self.model.male_threshold].sum()
        female_risk = PS[liability > self.model.female_threshold].sum()

        def compute_fs(idx):
            if all(np.logical_not(idx)):
                return np.zeros(GS.shape[1])
            GSS = GS[idx, :]
            PSS = PS[idx]

            fs = (GSS / self.het_ns *
                  PSS[np.newaxis].T).sum(axis=0) / PSS.sum()
            return fs

        ma_fs = compute_fs(liability > self.model.male_threshold)
        mu_fs = compute_fs(liability <= self.model.male_threshold)

        mC_netSCLs = (2*(ma_fs**2 + (1-ma_fs)**2) - 1) * self.het_ns
        mD_netSCLs = (2*(ma_fs*mu_fs + (1-ma_fs)*(1-mu_fs)) - 1) * self.het_ns

        return {
            'family_type_key': self.get_type_key(),
            'prediction_method': prediction_method,
            'male_risk': male_risk,
            'female_risk': female_risk,
            'mC_mom_netSCLs': mC_netSCLs[self.het_mom].sum(),
            'mC_dad_netSCLs': mC_netSCLs[self.het_dad].sum(),
            'mD_mom_netSCLs': mD_netSCLs[self.het_mom].sum(),
            'mD_dad_netSCLs': mD_netSCLs[self.het_dad].sum()
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


def compute_global_stats(all_stats, family_type_set: FamilyTypeSet,
                         female_initial_risk: Optional[float],
                         male_initial_risk: Optional[float]):

    model = family_type_set.model
    global_stats = defaultdict(dict)

    global_stats['Model']['name'] = model.model_name
    global_stats['Model']['male threshold'] = model.male_threshold
    global_stats['Model']['female threshold'] = model.female_threshold
    global_stats['Model']['population variants number'] = \
        model.get_number_of_loci()

    for lci, lc in enumerate(model.locus_classes):
        global_stats['Model'][f'population variant class {lci}'] = \
            f"w={lc.w}, f={lc.f}, n={lc.n}"

    global_stats['Model']['number of family types'] = len(all_stats)

    if female_initial_risk is not None or male_initial_risk is not None:
        global_stats['Initial']['male risk'] = male_initial_risk
        global_stats['Initial']['female risk'] = female_initial_risk

    def weighted_average(att, w_att):
        w_sum = sum([stats[w_att] for stats in all_stats])
        w_ave = sum([stats[att]*stats[w_att]
                     for stats in all_stats if stats[w_att] > 0]) / w_sum
        return float(w_ave)

    for ft_str, ft_pref in [
            ("Unascertained", 'U'),
            ("Families with two affected boys", 'C'),
            ("Families with one affected one unaffected boy", 'D')]:
        w_att = f'p{ft_pref}'

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
                f'm{ft_pref}_dad_netSCLs', w_att)
            global_stats[ft_str]['sharing of the mother'] = weighted_average(
                f'm{ft_pref}_mom_netSCLs', w_att)
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
