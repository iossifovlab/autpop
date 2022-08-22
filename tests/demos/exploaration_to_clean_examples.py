from autpop.population_threshold_model import Model
from autpop.population_threshold_model import draw_population_liability
from autpop.population_threshold_model import save_global_stats
import pylab as pl

if __name__ == "__main__":
    models = Model.load_models_file("paper_model_table_models_wigler_v2.yaml")
    modelsD = {m.model_name: m for m in models}
    model = modelsD["Example 5: Complex frequent risk with protection"]
    fm_stats, gl_stats = model.compute_stats(family_number=200_000)
    save_global_stats(gl_stats)
    pl.figure()
    draw_population_liability(model, pl.gca())
    pl.show(block=False)
