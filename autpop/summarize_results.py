from autpop.population_threshold_model import save_global_stats_table
from collections import defaultdict
import numpy as np
import re
import glob
import sys


def load_model_stats(model_stats_file):

    measureRE = re.compile("^  ([^:]+):\\s(.+)$")

    global_stats = defaultdict(dict)
    F = open(model_stats_file)
    section = None
    for line in F:
        line = line.strip("\n\r")
        if not line:
            continue
        match = measureRE.match(line)
        if match:
            key = match[1]
            vstr = match[2]
            try:
                v = int(vstr)
            except ValueError:
                try:
                    v = float(vstr)
                except ValueError:
                    v = vstr

            global_stats[section][key] = v
        else:
            section = line
    F.close()
    return global_stats


def method1(vls, MXP):
    for p in range(MXP, -1, -1):
        sts = {f'%.{p}f' % v for v in vls}
        if len(sts) == 1:
            r, = sts
            return r


def method2(vls, MXP):
    mn = np.mean(vls)
    for p in range(MXP, -1, -1):
        mvs = f'%.{p}f' % mn
        mv = float(f'%.{p}f' % mn)
        md = max([abs(mv-v) for v in vls])
        dst_cutoff = (10**-(p-1)) / 2
        # print("\t", p, mvs, mv, md, dst_cutoff)
        if md < dst_cutoff:
            return mvs


def summarize(GSB):
    r = []
    by_models = defaultdict(list)
    for GS in GSB:
        by_models[GS['Model']['name']].append(GS)
    for _, GSS in by_models.items():
        GS = GSS[0]
        for sec, dd in GS.items():
            if sec == 'Model':
                continue
            for key, v in dd.items():
                if isinstance(v, float):
                    vs = [G[sec][key] for G in GSS]
                    GS[sec][key] = method2(vs, 4)
        r.append(GS)
    return r


def cli(cli_args=None):
    if not cli_args:
        cli_args = sys.argv[1:]

    res_dir = "paper_model_table_models_wigler_v2_results"
    if cli_args:
        res_dir = cli_args[0]

    stat_files = sorted(glob.glob(f"{res_dir}/*/global_stats_*"))

    GSB = [load_model_stats(gsf) for gsf in stat_files]
    GSBS = summarize(GSB)
    save_global_stats_table(GSBS, prec=None)
