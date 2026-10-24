"""
Parámetros del escenario y de los algoritmos.

Todo lo que es supuesto (capacidades, velocidades, consumos, riesgos, pesos de la
función objetivo) vive aquí para que se pueda discutir y cambiar en un solo lugar.
Los valores de flota son de orden de magnitud para un helicóptero tipo Mi-17/Mi-171
y un camión 4x4 de 8 t; no son datos oficiales de la FAP ni del Ejército.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(RAIZ, "datos")
DIR_RESULTADOS = os.path.join(RAIZ, "resultados")


@dataclass(frozen=True)
class TipoVehiculo:
    nombre: str
    modo: str                 # "aereo" | "terrestre"
    capacidad_kg: float
    velocidad_kmh: float      # crucero (aéreo) o en vía sin exposición (terrestre)
    consumo_l_h: float        # aéreo: litros por hora de vuelo
    consumo_l_km: float       # terrestre: litros por km recorrido
    servicio_h: float         # descarga en cada parada (incluye aterrizaje/despegue)
    carga_h: float            # carga inicial en el depósito
    recarga_h: float          # entre dos viajes del mismo vehículo
    riesgo_h: float           # aéreo: índice de riesgo por hora de vuelo
    riesgo_km: float          # terrestre: índice de riesgo por km en vía normal
    riesgo_parada: float      # por aterrizaje o parada


HELICOPTERO = TipoVehiculo(
    nombre="helicoptero", modo="aereo", capacidad_kg=4000.0, velocidad_kmh=200.0,
    consumo_l_h=800.0, consumo_l_km=0.0, servicio_h=1 / 3, carga_h=0.5, recarga_h=0.5,
    riesgo_h=0.004, riesgo_km=0.0, riesgo_parada=0.0015,
)

CAMION = TipoVehiculo(
    nombre="camion", modo="terrestre", capacidad_kg=8000.0, velocidad_kmh=40.0,
    consumo_l_h=0.0, consumo_l_km=0.35, servicio_h=0.5, carga_h=0.75, recarga_h=0.75,
    riesgo_h=0.0, riesgo_km=0.0004, riesgo_parada=0.0005,
)

TIPOS = {"helicoptero": HELICOPTERO, "camion": CAMION}


@dataclass
class Config:
    # ---------------- flota: (id, tipo, depósito base) ----------------
    flota: list = field(default_factory=lambda: [
        ("H1", "helicoptero", "D1"), ("H2", "helicoptero", "D1"),
        ("T1", "camion", "D1"), ("T2", "camion", "D1"),
        ("T3", "camion", "D2"), ("T4", "camion", "D2"),
    ])
    deposito_aereo: str = "D1"

    # ---------------- demanda ----------------
    kg_por_familia: float = 10.0          # víveres 4 + agua 4 + higiene 1.5 + medicinas 0.5
    kg_medicina_por_familia: float = 0.5
    ventana_medicina_h: dict = field(default_factory=lambda: {"aereo": 2.5, "terrestre": 3.0})
    ventana_general_h: float = 24.0       # comunidades sin prioridad médica

    # ---------------- función objetivo ----------------
    # J = wT*sum(llegadas)/T_ref + wF*combustible/F_ref + wR*riesgo/R_ref + lambda_tw*sum(atraso)
    w_tiempo: float = 0.50
    w_combustible: float = 0.25
    w_riesgo: float = 0.25
    T_ref_h: float = 1.0
    F_ref_l: float = 100.0
    R_ref: float = 0.01
    lambda_ventana: float = 10.0          # por hora de atraso en medicinas
    M_bloqueo: float = 1.0e4              # costo de un arco sin ruta terrestre (puente caído, vía cortada)

    # ---------------- red vial sintética ----------------
    sinuosidad: float = 1.30              # km de vía por km en línea recta
    largo_max_tramo_km: float = 11.0
    escala_exposicion_km: float = 1.0     # exposición = exp(-d_rio / escala)
    reduccion_velocidad: float = 0.5      # v = v0 * (1 - reduccion * exposicion)
    factor_riesgo_exposicion: float = 4.0 # r = r0 * (1 + factor * exposicion)
    offset_puente_grados: float = 0.0045  # separación de los extremos del puente respecto al cauce
    puente_restringido_factor_t: float = 2.0
    puente_restringido_riesgo: float = 0.01

    # ---------------- clustering (fase clásica) ----------------
    max_paradas: int = 4                  # n paradas -> n^2 qubits (presupuesto del simulador / QPU)
    peso_urgencia: float = 2.0            # distancia espacio-temporal: d + peso * |u_i - u_j| * d_media
    iter_kmedoides: int = 25

    # ---------------- QAOA ----------------
    p_max: int = 3
    alpha_cvar: float = 0.15
    rejilla_p1: int = 16
    maxiter_cobyla: int = 250
    shots: int = 4096
    motor_cuantico: str = "auto"          # auto | numpy | qiskit  (ver qrutas/cuantico.py)

    # ---------------- clásicos ----------------
    sa_lecturas: int = 64
    sa_barridos: int = 300
    ga_poblacion: int = 24
    ga_generaciones: int = 40
    puro_sa_iter: int = 8000
    puro_ga_poblacion: int = 60
    puro_ga_generaciones: int = 150
    repeticiones: int = 5

    # ---------------- estimación en hardware real ----------------
    t_1q_s: float = 50e-9
    t_2q_s: float = 300e-9
    t_medida_s: float = 1.5e-6
    t_reinicio_s: float = 250e-6          # rep_delay típico entre disparos
    latencia_job_s: float = 2.0

    semilla: int = 2026
    rapido: bool = False

    def aplicar_modo_rapido(self) -> "Config":
        """Versión liviana para pruebas: menos iteraciones, mismos modelos."""
        self.rapido = True
        self.p_max = 2
        self.rejilla_p1 = 10
        self.maxiter_cobyla = 120
        self.shots = 2048
        self.sa_lecturas = 32
        self.sa_barridos = 150
        self.puro_sa_iter = 2500
        self.puro_ga_generaciones = 60
        self.repeticiones = 2
        return self
