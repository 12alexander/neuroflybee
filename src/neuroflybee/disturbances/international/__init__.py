"""Estándar internacional de turbulencia (Dryden, MIL-F-8785C / MIL-HDBK-1797) — dueño: P2 (v21).

Contrato: una clase con `reset(rng)` y `velocity(t, position, rng) -> np.ndarray(2,)` que
cumpla `contracts.conformance.check_wind_model`. Registrar las condiciones con
`contracts.conditions_registry.register(Conditions("dryden_moderada", wind=...))`.
"""
