from __future__ import annotations
import yaml
import argparse
import os
import sys
import pathlib
from typing import List, NamedTuple, Dict, Optional, Tuple, Any
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import numpy as np
import scipy.stats as stats
from .cartisian import cartesian_product_pp, cartesian_product_itertools
import time


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
        return int(((ns+1)*(ns+2)//2).prod())

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

    @ staticmethod
    def dict_to_model(data) -> Model:
        locus_classes = []
        for locus_def in data['locus_classes']:
            locus_classes.append(LocusClass(
                locus_def["w"],
                locus_def.get("n", 1),
                locus_def.get("f", 0.01)))

        return Model(data["name"],
                     data["male_threshold"],
                     data["female_threshold"],
                     locus_classes)

    def to_dict(self):
        return {
            "name": self.model_name,
            "male_threshold": self.male_threshold,
            "female_threshold": self.female_threshold,
            "locus_classes": [{"w": lc.w, "f": lc.f, "n": lc.n}
                              for lc in self.locus_classes]
        }

    @ staticmethod
    def load_models_file(file_name: str) -> List[Model]:
        with open(file_name) as F:
            model_defs = yaml.safe_load_all(F)
            return [Model.dict_to_model(model_def['threshold_model'])
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
            g_arrays.append(A[:, : 2])

        G = cartesian_product_itertools(g_arrays)
        P = cartesian_product_itertools(p_arrays).prod(axis=1)

        return G, P

    def sample_individual_genotypes(self, number_of_individuals):
        by_class = []
        for lc in self.locus_classes:
            by_class.append(stats.multinomial.rvs(
                lc.n,
                [lc.f*lc.f, 2*lc.f*(1-lc.f), (1-lc.f)*(1-lc.f)],
                number_of_individuals)[:, : 2])
        return np.array(by_class).transpose([1, 0, 2])

    def sample_unaffected_individual_genotypes(self, number_of_individuals,
                                               threshold):
        n_done = 0
        gen_buff = []
        n_simulated = 0
        while n_done < number_of_individuals:
            Gs = self.sample_individual_genotypes(number_of_individuals)
            liability = self.compute_liabilities(Gs)
            unaffected_idx = liability <= threshold
            gen_buff.append(Gs[unaffected_idx, :])
            n_done += unaffected_idx.sum()
            n_simulated += number_of_individuals
            print(f"\t{n_done}/{number_of_individuals} done.")
        gens = np.vstack(gen_buff)
        return gens[:number_of_individuals, :], \
            float(1.0 - (n_done / n_simulated))

    def sample_unaffected_parents_families(self, family_number):
        print(f"    Sampling {family_number} with unaffected parents...")
        print(f"      Mother genotypes:")
        momGs, female_risk = self.sample_unaffected_individual_genotypes(
            family_number, self.female_threshold)
        print(f"      Father genotypes:")
        dadGs, male_risk = self.sample_unaffected_individual_genotypes(
            family_number, self.male_threshold)

        ft_probability = (1 / family_number)**2
        family_type_set = FamilyTypeSet(self,
                                        f"sampling {family_number} "
                                        "unaffected-parents families",
                                        'sample')
        for momG, dadG in zip(momGs, dadGs):
            family_type_set.add_family_type(
                FamilyType(self, momG, dadG, delay_precompute=True),
                ft_probability)
        return family_type_set, female_risk, male_risk

    def sample_all_families(self, family_number):
        print(f"    Sampling {family_number} families...")
        momGs = self.sample_individual_genotypes(family_number)
        dadGs = self.sample_individual_genotypes(family_number)

        def risk(GS, threshold):
            L = self.compute_liabilities(GS)
            return float((L > threshold).sum()/len(L))

        ft_probability = (1 / family_number)**2
        family_type_set = FamilyTypeSet(self,
                                        f"sampling {family_number} families",
                                        'sample')
        for momG, dadG in zip(momGs, dadGs):
            family_type_set.add_family_type(
                FamilyType(self, momG, dadG, delay_precompute=True),
                ft_probability)
        return family_type_set, \
            risk(momGs, self.female_threshold), \
            risk(dadGs, self.male_threshold)

    def compute_liabilities(self, GS):
        return (GS*self.W).sum(axis=(1, 2))

    def compute_liability(self, G):
        return (G*self.W).sum()

    def compute_genotype_probability(self, G):
        multinomial_dists = [
            stats.multinomial(lc.n, [lc.f*lc.f,
                                     2*lc.f*(1-lc.f),
                                     (1-lc.f)*(1-lc.f)])
            for lc in self.locus_classes]
        probs = np.array([
            mult_dist.pmf([hom, het, lc.n-hom-het])
            for (hom, het), lc, mult_dist in
            zip(G, self.locus_classes, multinomial_dists)])
        prob = float(probs.prod())
        return prob

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

        dad_initial_risk = float(1.0 - P[dad_idx].sum())
        mom_initial_risk = float(1.0 - P[mom_idx].sum())

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
        if n_families > family_number:
            if warn:
                print(f"WARNING: too may family types {n_families}",
                      file=sys.stderr)
            else:
                raise Exception(f"Too many family types {n_families}")
        print(f"    Generating {description} ({n_families}).")
        family_type_set = FamilyTypeSet(self, description, 'precise')
        for dad_p, dad_g in zip(dadPs, dadGs):
            for mom_p, mom_g in zip(momPs, momGs):
                ft = FamilyType(self, mom_g, dad_g, delay_precompute=True)
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
            if all_families:
                return self.sample_all_families(family_number)
            else:
                return self.sample_unaffected_parents_families(family_number)
        elif family_mode == "dynamic":
            try:
                return self.generate_all_family_types(
                    genotype_type_number, family_number,
                    all_families=all_families, warn=False)
            except Exception:
                if all_families:
                    return self.sample_all_families(family_number)
                else:
                    return self.sample_unaffected_parents_families(
                        family_number)
        else:
            raise Exception(f"Unkown family_mode {family_mode}. The known "
                            f"family_modes are all, sample, and dynamic.")


class FamilyTypeSet:
    def __init__(self, model: Model, description: str, method: str):
        '''
        method should be 'precise' or 'sample'
        '''
        self.model = model
        self.description = description
        self.method = method
        assert method in ['precise', 'sample']
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

    def compute_family_stats(self, family_stats_mode="dynamic",
                             children_number=100_000, warn=True,
                             n_processes=None):
        stats = self.compute_all_family_specific_stats(
            family_stats_mode, children_number, warn, n_processes)
        self.add_family_set_specific_stats(stats)
        return stats

    def compute_all_family_specific_stats(self, family_stats_mode="dynamic",
                                          children_number=100_000, warn=True,
                                          n_processes=None):

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
            [st['parents_affected'] == 0 for st in all_stats], dtype=bool)
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
    def __init__(self, model: Model, mom_genotypes, dad_genotypes,
                 delay_precompute=False):

        self.model = model

        self.mom_genotypes = mom_genotypes
        self.dad_genotypes = dad_genotypes

        self.precomputed = False
        if not delay_precompute:
            self.precompute()

    def precompute(self):
        if self.precomputed:
            return
        self.precomputed = True

        self.homozygous_damage_mom = self.model.ws.dot(
            self.mom_genotypes[:, 0])
        self.homozygous_damage_dad = self.model.ws.dot(
            self.dad_genotypes[:, 0])

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

        self.mom_liability = self.model.compute_liability(self.mom_genotypes)
        self.dad_liability = self.model.compute_liability(self.dad_genotypes)

    def get_homozygous_damage(self):
        return self.homozygous_damage_dad + self.homozygous_damage_mom

    def is_dad_affected(self):
        return self.dad_liability > self.model.male_threshold

    def is_mom_affected(self):
        return self.mom_liability > self.model.female_threshold

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

    def get_family_probability(self):
        return self.model.compute_genotype_probability(self.mom_genotypes) * \
            self.model.compute_genotype_probability(self.dad_genotypes)

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
        self.precompute()

        family_stats = {
            'family_type_key': self.get_type_key(),
            'parents_affected': int(not self.are_parents_unaffected()),
            'mom_liability': self.mom_liability,
            'mom_affected': int(self.is_mom_affected()),
            'dad_liability': self.dad_liability,
            'dad_affected': int(self.is_dad_affected()),
            'family_probability': self.get_family_probability()
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
            precise_prediction = True
        elif family_stats_mode == "sample":
            precise_prediction = False
        elif family_stats_mode == "dynamic":
            if self.get_number_of_child_types() < children_number:
                precise_prediction = True
            else:
                precise_prediction = False
        else:
            raise Exception(f"Unknown family_stats_mode {family_stats_mode}."
                            f"The family_stats_mode should be all, sample, "
                            f"or dynamic.")

        if precise_prediction:
            GS, PS = self.generate_all_child_types()
        else:
            GS, PS = self.sample_child_types(children_number)

        liability = (GS * self.het_ws).sum(axis=1) + \
            self.homozygous_damage_mom + self.homozygous_damage_dad
        male_risk = PS[liability > self.model.male_threshold].sum()
        female_risk = PS[liability > self.model.female_threshold].sum()

        family_stats['precise_prediction'] = int(precise_prediction)
        family_stats['male_risk'] = male_risk
        family_stats['female_risk'] = female_risk

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

        family_stats['mC_mom_netSCLs'] = mC_netSCLs[self.het_mom].sum()
        family_stats['mC_dad_netSCLs'] = mC_netSCLs[self.het_dad].sum()
        family_stats['mD_mom_netSCLs'] = mD_netSCLs[self.het_mom].sum()
        family_stats['mD_dad_netSCLs'] = mD_netSCLs[self.het_dad].sum()

        return family_stats


def save_stats(all_stats, file_name: Optional[pathlib.Path] = None):
    if file_name:
        F = open(file_name, 'w')
    else:
        F = sys.stdout
    hcs = list(all_stats[0].keys())
    print("\t".join(hcs), file=F)

    for st in all_stats:
        cs: List[Any] = [st[a] for a in hcs]
        print("\t".join(map(str, cs)), file=F)
    if file_name:
        F.close()


def compute_global_stats(all_stats, family_type_set: FamilyTypeSet,
                         female_initial_risk: Optional[float] = None,
                         male_initial_risk: Optional[float] = None):

    model = family_type_set.model
    global_stats = defaultdict(dict)

    global_stats['threshold_model'] = model.to_dict()

    n_families_with_sample_based_predictions = \
        len([1 for st in all_stats
             if not st['precise_prediction']])

    n_unaffected_parents_families_with_sample_based_predictions = \
        len([1 for st in all_stats
             if not st['precise_prediction'] and
             not st['parents_affected']])
    global_stats['prediction'] = {

        'precise': family_type_set.method == 'precise' and
        n_unaffected_parents_families_with_sample_based_predictions == 0,
        'number of possible genotypes':
        model.get_number_of_individual_genotypes_types(),
        'family set description': family_type_set.description,
        'number of family types': len(all_stats),
        'number of family types with sampling':
        n_families_with_sample_based_predictions,
        'number of unaffected-parents family types':
        len([1 for st in all_stats if not st['parents_affected']]),
        'number of unaffected-parents family types with sampling':
        n_unaffected_parents_families_with_sample_based_predictions
    }

    if female_initial_risk is not None or male_initial_risk is not None:
        global_stats['initial risks'] = {
            'male risk': male_initial_risk,
            'female risk': female_initial_risk
        }

    def weighted_average(att, w_att):
        w_sum = sum([st[w_att] for st in all_stats])
        w_ave = sum([st[att]*st[w_att]
                     for st in all_stats if st[w_att] > 0]) / w_sum
        return float(w_ave)

    for ft_str, ft_pref in [
            ("unaffected parents families", 'U'),
            ("concordant families", 'C'),
            ("discordant families", 'D')]:
        w_att = f'p{ft_pref}'

        male_risk = weighted_average('male_risk', w_att)
        female_risk = weighted_average('female_risk', w_att)
        global_stats[ft_str]['male risk'] = male_risk
        global_stats[ft_str]['female risk'] = female_risk
        if ft_str != "unaffected parents families":
            global_stats[ft_str]['paternal net SCLs'] = weighted_average(
                f'm{ft_pref}_dad_netSCLs', w_att)
            global_stats[ft_str]['maternal net SCLs'] = weighted_average(
                f'm{ft_pref}_mom_netSCLs', w_att)
    return dict(global_stats)


def save_global_stats(global_stats,
                      file_name: Optional[pathlib.Path] = None):
    if file_name:
        F = open(file_name, "w")
    else:
        F = sys.stdout

    yaml.dump(global_stats, F, sort_keys=False)

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
                for section in \
                    ['Initial', 'Unascertained',
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
    parser.add_argument("-cn", "--children_number", type=int,
                        default=1_000_000,
                        help="The number of children. See <children_mode> for "
                        "further description")
    parser.add_argument("-w", "--warn", type=bool, default=False,
                        help="This argumet works when precise predictions "
                        "are requested at family set or family level "
                        "(see --family_mode and --children_mode arguments)."
                        "In these cases, the --warn controlls the behavious "
                        "when the number of famililes or children exceed the "
                        "limits set by the --family_number, --children_number,"
                        " --genotype_number aguments. If --warn is True, a "
                        "a warning will be printed; if --warn is False, an "
                        "error message will be printed and the tool will "
                        "terminate.")
    parser.add_argument("-j", "--n_processes", type=int, default=None,
                        help="The number of processes to be used for "
                        "parallelizing the computations. By default, all "
                        "available cores are used.")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="number of runs per model")
    parser.add_argument("-af", "--all_families", action="store_true",
                        help="When --all_families is True, all families, "
                        "including the ones with affected parents will be "
                        "included in the in family stats files. Otherwise, "
                        "only families with unaffected parents will be "
                        "included.")

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

            time_beg = time.time()

            def step(msg):
                time_now = time.time()
                print(f"  STEP {msg} at {time_now-time_beg:.1f} seconds...")

            res_dir = out_dir / f"{model.model_name}.run{run}"
            res_dir.mkdir(parents=True, exist_ok=True)

            step("Building families")
            family_type_set, female_risk, male_risk = model.build_family_types(
                args.family_mode, args.genotype_number, args.family_number,
                args.all_families, args.warn)

            step(f"Computing {len(family_type_set.family_types)} family stats")
            family_stats = family_type_set.compute_family_stats(
                args.children_mode, args.children_number,
                args.warn, args.n_processes)

            step("Computing global stats")
            global_stats = compute_global_stats(family_stats, family_type_set,
                                                female_risk, male_risk)

            step("Saving results")
            save_stats(family_stats, res_dir /
                       f"family_types_{model.model_name}.txt")
            save_global_stats(global_stats)
            save_global_stats(
                global_stats,
                res_dir / f"global_stats_{model.model_name}.yaml")
            GSB.append(global_stats)
            step("DONE.")
            if global_stats['prediction']['precise']:
                assert run == 0
                break
    # save_global_stats_table(GSB, out_dir / "models_results.txt")
