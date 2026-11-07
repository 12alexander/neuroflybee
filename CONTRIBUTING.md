# Cómo trabajamos en equipo (v18)

1. **Una rama por versión:** `git switch -c feat/v21-dryden`. Cada quien commitea con **su
   propio usuario de git** (el enunciado pide un historial que refleje el trabajo de todos).
2. **Sólo se editan las carpetas propias** (ver `CODEOWNERS`). Lo nuevo se conecta por los
   contratos, sin tocar el núcleo:
   | Quiero… | Implemento… | En… |
   |---|---|---|
   | un modelo de viento | `contracts.WindModel` | `disturbances/…` |
   | una falla de sensor | `contracts.SensorPerturbation` | `sensors_real/`, `robustness/` |
   | una condición con nombre | `conditions_registry.register(Conditions(...))` | mi módulo |
   | una ruta del servidor | `register(routes)` | `app/extensions/mi_modulo.py` |
   | algo en el visor | `NFB.use({...})` | `viz/viewer/plugins/mi_plugin.js` |
   | texto del artículo | mi archivo | `paper/secciones/` |
3. **Pruebas:** cada módulo trae `tests/test_<modulo>.py` y, si implementa un contrato, llama a
   `contracts.conformance.check_wind_model` / `check_perturbation`. `python -m pytest -q`
   debe pasar antes del pull request (GitHub Actions lo vuelve a correr).
4. **Historial:** un archivo `changelog.d/vNN.md` por versión (no editar `CHANGELOG.md`).
5. **Algo visible por versión:** una figura `results/figures/vNN_*.png` o una demo.
6. **Pull request** hacia `main`; P1 integra en orden de versión y se respalda la carpeta
   `neuroflybee_vNN`.
