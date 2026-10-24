"""
Q-Rutas Piura: ruteo híbrido clásico-cuántico de ayuda humanitaria (Fenómeno El Niño).

Uso:
    python main.py                 # corrida completa (unos minutos en una laptop)
    python main.py --rapido        # versión liviana para probar que todo corre
    python main.py --motor qiskit  # entrena y muestrea el QAOA con Qiskit (más lento)

Salidas en resultados/: figuras PNG, rutas.csv, metricas_*.csv, resultados.json y resumen.md
"""
from __future__ import annotations

import argparse
import os
import time

from qrutas.config import DIR_RESULTADOS, Config
from qrutas.pipeline import ejecutar, guardar
from qrutas.reporte import escribir_resumen
from qrutas.visualizacion import generar_figuras


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="menos iteraciones (prueba rápida)")
    ap.add_argument("--motor", choices=["auto", "numpy", "qiskit"], default="auto",
                    help="auto: entrena con el simulador numpy y muestrea con Qiskit si está instalado")
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--salida", default=DIR_RESULTADOS)
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args()

    cfg = Config(semilla=args.semilla, motor_cuantico=args.motor)
    if args.rapido:
        cfg.aplicar_modo_rapido()
    t0 = time.perf_counter()
    R = ejecutar(cfg)
    guardar(R, args.salida)
    figuras = {} if args.sin_figuras else generar_figuras(R, args.salida)
    escribir_resumen(R, figuras, os.path.join(args.salida, "resumen.md"))
    print(f"\nListo en {time.perf_counter() - t0:.1f} s. Resultados en {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
