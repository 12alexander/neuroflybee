"""Números del artículo generados desde los resultados (v19).

Cada cifra del texto es una macro LaTeX (`\\nfbExitoAstarMoviles`, …) escrita en
`paper/generado/numeros.tex` por `scripts/collect_results.py`, con su archivo de origen en
`results/v19_numeros.csv`. Así se cumple «cada número del artículo se puede regenerar con un
comando del README». Lo que aún no existe (p. ej. resultados neuronales antes de entrenar)
se escribe como `\\pendiente{...}`, que el PDF muestra en rojo: nunca se inventa un número.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional


# Las configuraciones dicen QUÉ números existen aunque aún no haya resultados (así cada
# macro queda definida, en rojo si falta, y el artículo siempre compila).
CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def pct(x: float, decimals: int = 0) -> str:
    return f"{100 * x:.{decimals}f}\\,\\%"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "\\,")


def mean_sd(values: list[float], fmt: Callable[[float], str]) -> str:
    if len(values) == 1:
        return fmt(values[0])
    return f"{fmt(statistics.mean(values))} $\\pm$ {fmt(statistics.stdev(values))}"


def _csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@dataclass
class NumberSet:
    values: dict[str, tuple[str, str]] = field(default_factory=dict)   # macro -> (valor, fuente)

    def put(self, name: str, value: str, source: str) -> None:
        if not name.isalpha():
            raise ValueError(f"Nombre de macro inválido: {name} (sólo letras)")
        self.values[name] = (value, source)

    def pending(self, name: str, what: str) -> None:
        safe = what.replace("\\", "").replace("_", "\\_").replace("%", "\\%")
        self.values[name] = (f"\\pendiente{{{safe}}}", "pendiente")

    def is_pending(self, name: str) -> bool:
        return self.values.get(name, ("", "pendiente"))[1] == "pendiente"

    def latex(self) -> str:
        lines = ["% GENERADO por scripts/collect_results.py — no editar a mano.",
                 "% Fuente de cada número: results/v19_numeros.csv"]
        lines += [f"\\newcommand{{\\nfb{n}}}{{{v}}}  % {src}" for n, (v, src) in sorted(self.values.items())]
        return "\n".join(lines) + "\n"

    def write(self, tex_path: Path, csv_path: Path) -> None:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(self.latex(), encoding="utf-8")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["macro", "valor", "fuente"])
            for n, (v, src) in sorted(self.values.items()):
                w.writerow([f"\\nfb{n}", v, src])


# ------------------------------------------------------------------ lectores (uno por fuente)
CLASSICAL = {"Directo": "Directo", "Campos de potencial": "Campos", "A* (clásico)": "Astar",
             "A* + ALV (clásico)": "AstarAlv", "A* + supervisor": "AstarSup"}
NEURAL = {"Mosca": "Mosca", "Abeja": "Abeja", "Híbrido abeja+mosca": "Hibrido"}
FAMILY = {"nominal": "Nominal", "unknown_obstacles": "Desconocidos", "dynamic": "Moviles",
          "wind": "Viento", "gps_loss": "SinGPS", "combined": "Combinado", "total": "Total"}


def datasets(ns: NumberSet, results: Path) -> None:
    src = "v7_datasets.csv"
    rows = {(r["modulo"], r["particion"]): r for r in _csv(results / src)}
    part_names = {"train": "Train", "val": "Val", "test": "Test"}
    for (mod, part), r in rows.items():
        key = ("Mosca" if mod == "mosca" else "Abeja") + part_names[part]
        ns.put(f"Muestras{key}", thousands(int(r["muestras"])), src)
        ns.put(f"Episodios{key}", thousands(int(r["episodios_o_escenarios"])), src)
    fly = rows[("mosca", "train")]
    total = int(fly["muestras"])
    ns.put("PctAvanzar", pct(int(fly["Avanzar"]) / total), src)
    ns.put("PctMantener", pct(int(fly["Mantener posición"]) / total, 1), src)


def expert(ns: NumberSet, results: Path) -> None:
    rows = _csv(results / "v7_experto.csv")
    eps = sum(int(r["episodios"]) for r in rows)
    ok = sum(int(r["episodios"]) * float(r["exito_experto"]) for r in rows)
    ns.put("ExitoExperto", pct(ok / eps), "v7_experto.csv")
    ns.put("EpisodiosExperto", str(eps), "v7_experto.csv")


def parameters(ns: NumberSet, results: Path) -> None:
    names = {"fly_mlp": "MLP", "fly_cnn": "CNN", "fly_cnn_gru": "GRU", "fly_cnn_transformer": "Transformer"}
    for r in _csv(results / "v8_parametros.csv"):
        if r["modelo"] in names:
            ns.put(f"Params{names[r['modelo']]}", thousands(int(r["parametros_entrenables"])), "v8_parametros.csv")


def alv(ns: NumberSet, results: Path) -> None:
    for r in _csv(results / "v11_abeja_validacion.csv"):
        if r["metodo"] == "alv":
            ns.put("AlvErrorMediano", f"{float(r['error_angular_mediano_grados']):.1f}\\textdegree", "v11_abeja_validacion.csv")
            ns.put("AlvFraccionTreinta", pct(float(r["fraccion_error_menor_30"])), "v11_abeja_validacion.csv")


def main_table(ns: NumberSet, results: Path) -> None:
    src = "v14_tabla_principal.csv"
    rows = {r["controlador"]: r for r in _csv(results / src)}
    for label, slug in CLASSICAL.items():
        r = rows.get(label)
        if r is None:
            continue
        for fam, short in FAMILY.items():
            ns.put(f"Exito{slug}{short}", pct(float(r[f"exito_{fam}"])), src)
        ns.put(f"Choques{slug}", pct(float(r["choques_total"])), src)
    gps = [float(rows[l]["exito_gps_loss"]) for l in CLASSICAL if l in rows]
    moving = [float(rows[l]["exito_dynamic"]) for l in CLASSICAL if l in rows]
    ns.put("MejorClasicoSinGPS", pct(max(gps)), src)
    ns.put("MejorClasicoMoviles", pct(max(moving)), src)
    any_row = next(iter(rows.values()))
    ns.put("EpisodiosPorCondicion", str(int(any_row["episodios"]) // len(FAMILY.keys() - {"total"})), src)
    for prefix, slug in NEURAL.items():
        r = next((v for k, v in rows.items() if k.startswith(prefix)), None)
        for fam, short in FAMILY.items():
            if r is None:
                ns.pending(f"Exito{slug}{short}", f"{prefix.lower()}: entrenar en Colab")
            else:
                ns.put(f"Exito{slug}{short}", pct(float(r[f"exito_{fam}"])), src)
                if fam == "total" and int(r.get("semillas") or 1) > 1:
                    ns.put(f"Exito{slug}TotalDE", pct(float(r["exito_total_de"])), src)


def error_shares(ns: NumberSet, results: Path) -> None:
    src = "v15_errores.csv"
    rows = _csv(results / src)
    by: dict[str, dict[str, int]] = {}
    for r in rows:
        by.setdefault(r["controlador"], {}).setdefault(r["categoria"], 0)
        by[r["controlador"]][r["categoria"]] += 1

    def share_range(category: str, controllers: Iterable[str]) -> str:
        vals = [by[c].get(category, 0) / sum(by[c].values()) for c in controllers if c in by]
        return f"{100 * min(vals):.0f}--{100 * max(vals):.0f}\\,\\%"
    ns.put("ErrorDerivaRango", share_range("deriva sin GPS (aterriza lejos)", by), src)
    ns.put("ErrorMovilRango", share_range("choque: obstáculo móvil", ["astar", "astar_alv", "astar_supervised"]), src)


def supervisor(ns: NumberSet, results: Path) -> None:
    src = "v12_supervisor.csv"
    rows = _csv(results / src)
    def col(ctrl: str, families: set[str]) -> float:
        vals = [float(r["choques"]) for r in rows if r["controlador"] == ctrl and r["condicion"] in families]
        return sum(vals) / len(vals)
    fixed = {"nominal", "unknown_obstacles"}
    ns.put("ChoquesDirectoFijos", pct(col("direct", fixed)), src)
    ns.put("ChoquesDirectoSupFijos", pct(col("direct_supervised", fixed)), src)


def fly_training(ns: NumberSet, results: Path) -> None:
    names = {"fly_mlp": "MLP", "fly_cnn": "CNN", "fly_cnn_gru": "GRU", "fly_cnn_transformer": "Transformer"}
    for model, short in names.items():
        files = sorted(results.glob(f"v9_historial_{model}_s*.json"))
        if not files:
            ns.pending(f"FUno{short}", "F1 macro: entrenar en Colab")
            ns.pending(f"Auroc{short}", "AUROC: entrenar en Colab")
            continue
        histories = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        ns.put(f"FUno{short}", mean_sd([h["best_score"] for h in histories], lambda v: f"{v:.3f}"),
               f"v9_historial_{model}_s*.json")
        aurocs = [best_epoch_metric(h, "auroc_macro") for h in histories]
        if any(a is None for a in aurocs):          # entrenado antes de v22
            ns.pending(f"Auroc{short}", "AUROC: reentrenar la mosca con v22")
        else:
            ns.put(f"Auroc{short}", mean_sd(aurocs, lambda v: f"{v:.3f}"), f"v9_historial_{model}_s*.json")


def best_epoch_metric(history: dict, name: str) -> Optional[float]:
    """Métrica de validación en la mejor época de un historial (None si no se registró)."""
    for epoch in history.get("history", []):
        if epoch["epoch"] == history.get("best_epoch"):
            value = epoch["metrics"].get(name)
            return None if value is None else float(value)
    return None


def dagger(ns: NumberSet, results: Path) -> None:
    files = sorted(results.glob("v10_dagger_*_s*.json"))
    if not files:
        ns.pending("DaggerExitoInicial", "DAgger: entrenar"); ns.pending("DaggerExitoFinal", "DAgger: entrenar")
        return
    first, last = [], []
    for f in files:
        its = json.loads(f.read_text(encoding="utf-8"))["iterations"]
        first.append(its[0]["closed_loop"]["all"]["success"]); last.append(its[-1]["closed_loop"]["all"]["success"])
    ns.put("DaggerExitoInicial", mean_sd(first, pct), "v10_dagger_*.json")
    ns.put("DaggerExitoFinal", mean_sd(last, pct), "v10_dagger_*.json")


def bee_training(ns: NumberSet, results: Path) -> None:
    files = sorted(results.glob("v11_historial_bee_s*.json"))
    if not files:
        ns.pending("AbejaErrorMediano", "abeja: entrenar en Colab")
        return
    vals = [json.loads(f.read_text(encoding="utf-8"))["best_score"] for f in files]
    ns.put("AbejaErrorMediano", mean_sd(vals, lambda v: f"{v:.1f}\\textdegree"), "v11_historial_bee_s*.json")


BEE_ABLATION = {"alv": "Alv", "completa": "Completa", "vista_unica": "VistaUnica",
                "sin_diferencia": "SinResta", "compacta": "Compacta"}


def bee_variant_parameters(config_path: Path = CONFIGS / "bee_ablation.yaml") -> dict[str, int]:
    """Parámetros de cada variante de la abeja según su especificación (no requiere entrenar)."""
    from ..evaluation.bee_ablation import load_bee_ablation_config
    from ..models.bee_spec import BeeNetConfig, count_bee_parameters

    if not config_path.exists():
        return {}
    cfg = load_bee_ablation_config(config_path)
    return {v.name: count_bee_parameters(v.config(BeeNetConfig())) for v in cfg.variants}


def bee_ablation(ns: NumberSet, results: Path, config_path: Path = CONFIGS / "bee_ablation.yaml") -> None:
    """v20: error angular mediano (media ± d.e. entre semillas) de cada variante en PRUEBA."""
    src = "v20_ablacion_abeja_resumen.csv"
    path = results / src
    rows = {r["variante"]: r for r in _csv(path)} if path.exists() else {}
    params = bee_variant_parameters(config_path)
    for variant, slug in BEE_ABLATION.items():
        if variant in params:
            ns.put(f"ParamsAbeja{slug}", thousands(params[variant]), "configs/bee_ablation.yaml")
        r = rows.get(variant)
        if r is None or (variant != "alv" and int(r["semillas"] or 0) == 0):
            ns.pending(f"AblAbeja{slug}", "ablación de la abeja: correr run_bee_ablation.py")
            if variant != "alv":
                ns.pending(f"AblAbeja{slug}Treinta", "ablación de la abeja")
            continue
        seeds = int(r["semillas"] or 0)
        value = f"{float(r['error_mediano_media']):.1f}\\textdegree"
        if seeds > 1:
            value = f"{float(r['error_mediano_media']):.1f} $\\pm$ {float(r['error_mediano_de']):.1f}\\textdegree"
        ns.put(f"AblAbeja{slug}", value, src)
        if variant != "alv":
            ns.put(f"AblAbeja{slug}Treinta", pct(float(r["fraccion_menor_30_media"])), src)
    if "completa" in params and "compacta" in params:
        ns.put("AbejaReduccionParams", f"{params['completa'] / max(params['compacta'], 1):.0f}",
               "configs/bee_ablation.yaml")
    else:
        ns.pending("AbejaReduccionParams", "ablación de la abeja")
    if "MuestrasAbejaTest" not in ns.values:     # datos generados antes de v20
        ns.pending("MuestrasAbejaTest", "regenerar datos (v20 agrega la prueba de la abeja)")
        ns.pending("EpisodiosAbejaTest", "regenerar datos")


def sensor_validation(ns: NumberSet, results: Path) -> None:
    """v21: validación del sensor panorámico (IDF, rumbo y área de captura del ALV)."""
    src = "v21_validacion_resumen.json"
    path = results / src
    names = ("IdfSpearman", "IdfMonotonas", "IdfEscenarios", "IdfPisoRuido", "IdfRadioLlegada",
             "RumboErrorDos", "RumboErrorCuatro", "AlvCaptura", "AlvErrorCampo")
    if not path.exists():
        for n in names:
            ns.pending(n, "validación del sensor: correr validate_sensor.py")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    ns.put("IdfSpearman", f"{d['spearman_mean']:.2f}", src)
    ns.put("IdfMonotonas", pct(d["monotonic_fraction"]), src)
    ns.put("IdfEscenarios", str(d["scenarios"]), src)
    ns.put("IdfPisoRuido", f"{d['noise_floor']:.3f}", src)
    ns.put("IdfRadioLlegada", f"{d['arrival_radius_m']:.2f}", src)
    for radius, name in (("2", "RumboErrorDos"), ("4", "RumboErrorCuatro")):
        value = d["heading_error"].get(radius)
        if value is None:
            ns.pending(name, f"rumbo a {radius} m: agregarlo a rotation_radii")
        else:
            ns.put(name, f"{value:.1f}\\textdegree", src)
    ns.put("AlvCaptura", pct(d["alv_catchment_mean"]), src)
    ns.put("AlvErrorCampo", f"{d['alv_error_median']:.1f}\\textdegree", src)


SIG_CONDITION = {"total": "Total", "nominal": "Nominal", "unknown_obstacles": "Desconocidos",
                 "dynamic": "Moviles", "wind": "Viento", "gps_loss": "SinGPS", "combined": "Combinado"}


def significance(ns: NumberSet, results: Path, config_path: Path = CONFIGS / "significance.yaml") -> None:
    """v22: diferencias pareadas y valores p (Holm) de las comparaciones con `macro`."""
    from ..evaluation.significance import load_significance_config

    src = "v22_significancia.csv"
    path = results / src
    rows = _csv(path) if path.exists() else []
    if config_path.exists():                      # primero todo pendiente; luego se llena lo medido
        for comp in load_significance_config(config_path).comparisons:
            for cond in comp.conditions if comp.macro else ():
                name = f"Sig{comp.macro}{SIG_CONDITION.get(cond, 'Otra')}"
                ns.pending(f"{name}Dif", f"{comp.a} vs {comp.b}: correr el benchmark")
                ns.pending(f"{name}P", "significancia")
    if not rows:
        ns.pending("SigSignificativas", "significancia: correr el benchmark y significance.py")
        ns.pending("SigComparaciones", "significancia")
        return
    ns.put("SigSignificativas", str(sum(int(r["significativa"]) for r in rows)), src)
    ns.put("SigComparaciones", str(len(rows)), src)
    for r in rows:
        if not r["macro"]:
            continue
        name = f"Sig{r['macro']}{SIG_CONDITION.get(r['condicion'], 'Otra')}"
        diff, lo, hi = (100 * float(r[k]) for k in ("diferencia", "ic_inf", "ic_sup"))
        ns.put(f"{name}Dif", f"{diff:+.0f}\\,pp [{lo:+.0f}, {hi:+.0f}]", src)
        p = float(r["p_holm"])
        ns.put(f"{name}P", "$p_{\\text{Holm}}<0.001$" if p < 0.001 else f"$p_{{\\text{{Holm}}}}={p:.3f}$", src)


def cost(ns: NumberSet, results: Path) -> Optional[dict]:
    path = results / "v19_costo.json"
    if not path.exists():
        for n in ("HorasGPU", "HorasTotales", "GPU"):
            ns.pending(n, "costo: correr scripts/run_pipeline.py")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ns.put("HorasGPU", f"{data['horas_gpu']:.2f}", "v19_costo.json")
    ns.put("HorasTotales", f"{data['horas_totales']:.2f}", "v19_costo.json")
    ns.put("GPU", data["hardware"].get("gpu", "ninguna").replace("_", "\\_"), "v19_costo.json")
    return data


def cost_table(data: Optional[dict]) -> str:
    head = ["\\begin{table}[t]", "\\caption{Costo computacional por etapa (generado desde \\texttt{v19\\_costo.csv})}",
            "\\label{tab:costo}", "\\centering", "\\begin{tabular}{lrr}", "\\toprule",
            "Etapa & Minutos & Horas de GPU \\\\", "\\midrule"]
    if data is None:
        body = ["\\multicolumn{3}{c}{\\pendiente{ejecutar scripts/run\\_pipeline.py en Colab}} \\\\"]
    else:
        body = [f"{s['stage']} & {s['seconds'] / 60:.1f} & {(s['seconds'] / 3600 if s['uses_gpu'] else 0):.2f} \\\\"
                for s in data["etapas"] if s["status"] == "ok"]
        body += ["\\midrule", f"Total & {data['horas_totales'] * 60:.1f} & {data['horas_gpu']:.2f} \\\\"]
    return "\n".join(head + body + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n"


READERS = (datasets, expert, parameters, alv, main_table, error_shares, supervisor, fly_training, dagger, bee_training,
           bee_ablation, sensor_validation, significance)


def collect(results: Path) -> tuple[NumberSet, Optional[dict]]:
    ns = NumberSet()
    for reader in READERS:
        reader(ns, results)
    return ns, cost(ns, results)
