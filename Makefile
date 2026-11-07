PY ?= python
export PYTHONPATH := src

.PHONY: test paper figures data train dagger bee viewer benchmark explain video serve pipeline numeros word

test:
	$(PY) -m pytest -q

data:
	$(PY) scripts/generate_data.py

train:
	$(PY) scripts/train_fly.py

dagger:
	$(PY) scripts/train_dagger.py

bee:
	$(PY) scripts/train_bee.py

viewer:
	$(PY) scripts/build_viewer.py

benchmark:
	$(PY) scripts/run_benchmark.py

explain:
	$(PY) scripts/explain.py

video:
	$(PY) scripts/record_video.py

serve:
	$(PY) scripts/serve.py

pipeline:
	$(PY) scripts/run_pipeline.py

numeros:
	$(PY) scripts/collect_results.py

word: numeros
	$(PY) scripts/paper_to_word.py

figures:
	$(PY) scripts/show_scenarios.py
	$(PY) scripts/show_sensors.py
	$(PY) scripts/show_episode.py
	$(PY) scripts/show_baselines.py
	$(PY) scripts/show_dataset.py
	$(PY) scripts/show_fly_models.py
	$(PY) scripts/plot_fly_training.py
	$(PY) scripts/plot_dagger.py
	$(PY) scripts/show_bee_homing.py
	$(PY) scripts/show_controllers.py

paper: numeros
	cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
