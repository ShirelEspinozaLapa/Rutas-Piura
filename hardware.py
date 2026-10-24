"""
¿Cuánto costaría correr esto en un QPU real?

1. Recursos lógicos del circuito (desde el IR, sin depender de Qiskit).
2. Si Qiskit y qiskit-ibm-runtime están instalados: transpilación a un backend
   falso de IBM (FakeTorino / FakeSherbrooke / FakeBrisbane) para obtener
   profundidad y compuertas de dos qubits ya enrutadas en la topología heavy-hex.
3. Estimación del tiempo de QPU para todo el entrenamiento y de la probabilidad
   de que un disparo corra sin error (modelo de error por compuerta).

Los tiempos y tasas de error por defecto son de orden de magnitud y están en
config.py; no son los de un dispositivo específico.
"""
from __future__ import annotations

import numpy as np

# tasas de error típicas de orden de magnitud (no de un equipo específico)
ERROR_1Q = 3e-4
ERROR_2Q = 3e-3
ERROR_MEDIDA = 1.5e-2
# sin transpilador, sobrecosto de enrutamiento supuesto para interacciones densas en heavy-hex
FACTOR_ENRUTAMIENTO = 3.0


def recursos_logicos(ir) -> dict:
    n1, n2_rzz, n2_cx = 0, 0, 0
    capa = np.zeros(ir.n_qubits, int)
    capa2 = np.zeros(ir.n_qubits, int)
    for nombre, qs, _ in ir.ops:
        t = max(capa[q] for q in qs) + 1
        for q in qs:
            capa[q] = t
        if len(qs) == 2:
            t2 = max(capa2[q] for q in qs) + 1
            for q in qs:
                capa2[q] = t2
            if nombre == "rzz":
                n2_rzz += 1
            else:
                n2_cx += 1
        else:
            n1 += 1
    # cada RZZ se descompone en 2 CNOT + 1 RZ en hardware sin RZZ nativo
    return dict(qubits=ir.n_qubits, capas_p=ir.p, compuertas_1q=n1, rzz=n2_rzz, cx=n2_cx,
                cx_equivalentes=2 * n2_rzz + n2_cx, profundidad=int(capa.max()),
                profundidad_2q=int(capa2.max()))


def transpilar_ibm(ir, gammas, betas, semilla: int = 7) -> dict | None:
    """Transpila a un backend falso de IBM. Devuelve None si no hay Qiskit / runtime."""
    try:
        from qiskit import transpile

        from .cuantico import ir_a_qiskit
    except ImportError:
        return None
    fake = None
    try:
        from qiskit_ibm_runtime import fake_provider as fp
        for nombre in ("FakeTorino", "FakeSherbrooke", "FakeBrisbane", "FakeKyiv"):
            if hasattr(fp, nombre):
                fake = getattr(fp, nombre)()
                break
    except ImportError:
        return None
    if fake is None:
        return None
    qc = ir_a_qiskit(ir, gammas, betas, medir=True)
    tqc = transpile(qc, backend=fake, optimization_level=3, seed_transpiler=semilla)
    ops = {k: int(v) for k, v in tqc.count_ops().items()}
    n2 = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rzz", "swap"))
    prof2 = tqc.depth(lambda ins: ins.operation.num_qubits == 2)
    duracion = None
    try:
        duracion = float(tqc.estimate_duration(fake.target, unit="s"))
    except Exception:
        try:
            t2 = None
            for k in ("cz", "ecr", "cx"):
                if k in fake.target.operation_names:
                    props = [p for p in fake.target[k].values() if p is not None and p.duration]
                    t2 = float(np.median([p.duration for p in props]))
                    break
            if t2 is not None:
                duracion = prof2 * t2 + (tqc.depth() - prof2) * 5e-8 + 1.5e-6
        except Exception:
            duracion = None
    return dict(backend=fake.name if hasattr(fake, "name") else str(fake), operaciones=ops,
                compuertas_2q=int(n2), profundidad=int(tqc.depth()), profundidad_2q=int(prof2),
                duracion_circuito_s=duracion)


def estimar_qpu(rec: dict, cfg, n_evaluaciones: int, shots: int, transpilado: dict | None = None) -> dict:
    if transpilado and transpilado.get("compuertas_2q"):
        n2 = transpilado["compuertas_2q"]
        dur = transpilado.get("duracion_circuito_s")
        fuente = f"transpilado a {transpilado['backend']}"
    else:
        n2 = int(rec["cx_equivalentes"] * FACTOR_ENRUTAMIENTO)
        dur = None
        fuente = f"estimación sin transpilador (enrutamiento x{FACTOR_ENRUTAMIENTO:g})"
    if dur is None:
        prof2 = rec["profundidad_2q"] * FACTOR_ENRUTAMIENTO
        dur = prof2 * cfg.t_2q_s + max(rec["profundidad"] - rec["profundidad_2q"], 0) * cfg.t_1q_s
        dur += cfg.t_medida_s
    por_disparo = dur + cfg.t_reinicio_s
    por_eval = shots * por_disparo + cfg.latencia_job_s
    fidelidad = ((1 - ERROR_2Q) ** n2) * ((1 - ERROR_1Q) ** rec["compuertas_1q"]) * \
                ((1 - ERROR_MEDIDA) ** rec["qubits"])
    return dict(fuente=fuente, compuertas_2q_fisicas=int(n2), duracion_circuito_s=dur,
                tiempo_por_evaluacion_s=por_eval, tiempo_entrenamiento_qpu_s=n_evaluaciones * por_eval,
                prob_disparo_sin_error=fidelidad)
