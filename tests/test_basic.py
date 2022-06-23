from autpop.population_threshold_model import Model

def test_build():
	model = Model("gosho", 1, 2, [])
	assert model.model_name == "gosho"
