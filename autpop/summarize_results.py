from autpop.population_threshold_model import save_global_stats_table
from autpop.population_threshold_model import average_multiple_runs
import glob
import yaml
import sys


def cli(cli_args=None):
    if not cli_args:
        cli_args = sys.argv[1:]

    res_dir = "paper_model_table_models_wigler_v2_results"
    prec = 3

    if cli_args:
        res_dir = cli_args[0]
    if len(cli_args) > 1:
        prec = int(cli_args[1])

    stat_files = sorted(glob.glob(f"{res_dir}/*/global_stats_*.yaml"))

    GSB = []
    for gsf in stat_files:
        with open(gsf) as F:
            GSB.append(yaml.safe_load(F))
    GSBS = average_multiple_runs(GSB, prec)
    save_global_stats_table(GSBS, prec=None)
