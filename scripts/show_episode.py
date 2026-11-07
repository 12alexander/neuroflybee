"""Figura de la v5: el dron vuela episodios completos en el entorno.

Usa un controlador de demostración (línea recta a cada punto, sin evasión) sólo para
mostrar el bucle observación → decisión → acción. Las líneas base reales llegan en v6.

Uso:
    python scripts/show_episode.py [--seed 20000]
Genera:
    results/figures/v5_episodios.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from neuroflybee.config import load_config  # noqa: E402
from neuroflybee.control.base import Decision  # noqa: E402
from neuroflybee.domain.drone import Action  # noqa: E402
from neuroflybee.domain.geometry import wrap_angle  # noqa: E402
from neuroflybee.domain.mission import Phase  # noqa: E402
from neuroflybee.sim.env import NavigationEnv  # noqa: E402
from neuroflybee.sim.observation import Observation  # noqa: E402
from neuroflybee.sim.rollout import rollout  # noqa: E402
from neuroflybee.sim.scenarios import FAMILY_LABELS_ES, ScenarioGenerator  # noqa: E402
from neuroflybee.viz.style import apply_style  # noqa: E402
from neuroflybee.viz.trajectory_plot import draw_trajectory, outcome_text  # noqa: E402
from neuroflybee.viz.world_plot import draw_world  # noqa: E402


class StraightLineDemo:
    """Demostración mínima: vuela en línea recta a cada punto y vuelve. Usa el GPS y,
    si se pierde, integra la odometría. No evade obstáculos."""

    name = "demo"

    def reset(self, obs: Observation) -> None:
        self.estimate = obs.base.astype(float).copy()
        self.targets = [*obs.waypoints, obs.base]
        self.index = 0

    def act(self, obs: Observation) -> Decision:
        self.estimate = obs.gps if obs.gps is not None else self.estimate + obs.odometry
        target = self.targets[self.index]
        if np.linalg.norm(target - self.estimate) < 1.0:
            if self.index == len(self.targets) - 1:
                return Decision(Action.HOVER, land=True, phase=Phase.LANDED, estimate=self.estimate.copy())
            self.index += 1
            target = self.targets[self.index]
        d = target - self.estimate
        rel = np.degrees(wrap_angle(np.arctan2(d[1], d[0]) - obs.heading))
        action = (Action.FORWARD if abs(rel) < 8 else
                  (Action.LEFT if rel > 0 else Action.RIGHT) if abs(rel) < 30 else
                  (Action.SHARP_LEFT if rel > 0 else Action.SHARP_RIGHT))
        phase = Phase.RETURN if self.index == len(self.targets) - 1 else Phase.INSPECT
        return Decision(action, phase=phase, goal=target, estimate=self.estimate.copy())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--families", default="nominal,unknown_obstacles,gps_loss")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args(argv)

    apply_style()
    cfg = load_config(args.config)
    gen = ScenarioGenerator(cfg)
    families = args.families.split(",")
    fig, axes = plt.subplots(1, len(families), figsize=(5.2 * len(families), 5.8))
    for ax, family in zip(np.atleast_1d(axes), families):
        env = NavigationEnv(cfg, gen.generate(family, args.seed))
        traj = rollout(env, StraightLineDemo())
        draw_world(ax, env.world, t=env.t, title=FAMILY_LABELS_ES[family],
                   gps_loss=env.scenario.gps_loss_on_return)
        draw_trajectory(ax, traj)
        ax.set_xlabel("x (m)\n" + outcome_text(traj, env.tracker.home_error(env.state.position), env.tracker.coverage))
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle("v5 — Episodios completos con un controlador de demostración (línea recta, sin evasión)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "v5_episodios.png")
    plt.close(fig)
    print(f"Figura guardada en {out / 'v5_episodios.png'}")


if __name__ == "__main__":
    main()
