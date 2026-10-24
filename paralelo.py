"""Ejecución concurrente y paralela de subproblemas independientes.

Dado que cada uno de los 7 subproblemas resultantes del clustering es matemáticamente desacoplado,
pueden resolverse simultáneamente utilizando múltiples hilos o procesos de CPU,
o enviando lotes paralelos asíncronos a servicios cuánticos en la nube.
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import Callable, Any


def resolver_subproblemas_en_paralelo(
    subproblemas: list,
    fn_resolver: Callable[[Any], tuple[str, dict]],
    max_workers: int = 4,
    usar_procesos: bool = False
) -> tuple[dict[str, dict], float]:
    """Ejecuta la función resolutora sobre cada subproblema en paralelo."""
    t0 = time.perf_counter()
    resultados = {}

    ExecutorClass = concurrent.futures.ProcessPoolExecutor if usar_procesos else concurrent.futures.ThreadPoolExecutor

    with ExecutorClass(max_workers=max_workers) as executor:
        futuros = {executor.submit(fn_resolver, sp): sp.id for sp in subproblemas}
        for futuro in concurrent.futures.as_completed(futuros):
            sp_id = futuros[futuro]
            try:
                id_ret, datos = futuro.result()
                resultados[id_ret] = datos
            except Exception as e:
                resultados[sp_id] = {"error": str(e)}

    tiempo_total_s = time.perf_counter() - t0
    return resultados, tiempo_total_s
