"""Configuración tipada con validación Pydantic y soporte de escenarios dinámicos.

Permite cargar y exportar escenarios completos de emergencia (en JSON o YAML),
modificando el estado de puentes, nivel de severidad de lluvia y capacidades de flota
con validación estricta de tipos y rangos físicos.
"""
from __future__ import annotations

import json
import os
from typing import Literal
from pydantic import BaseModel, Field


class EstadoPuente(BaseModel):
    nombre: str
    estado: Literal["operativo", "restringido", "cerrado", "colapsado"]


class VehiculoConfig(BaseModel):
    tipo: Literal["aereo", "terrestre"]
    deposito_id: str
    capacidad_kg: int = Field(gt=0, description="Capacidad máxima de carga en kg")
    cantidad: int = Field(default=1, ge=1)


class PesosObjetivo(BaseModel):
    peso_tiempo: float = Field(default=0.50, ge=0.0, le=1.0)
    peso_combustible: float = Field(default=0.25, ge=0.0, le=1.0)
    peso_riesgo: float = Field(default=0.25, ge=0.0, le=1.0)
    penalizacion_ventana: float = Field(default=10.0, ge=0.0)


class ParametrosCuanticos(BaseModel):
    motor: Literal["auto", "qiskit", "numpy"] = "auto"
    capas_p: int = Field(default=1, ge=1, le=5)
    shots: int = Field(default=4096, ge=100)
    cvar_alpha: float = Field(default=0.15, gt=0.0, le=1.0)


class EscenarioEmergencia(BaseModel):
    nombre: str = "Escenario Piura FEN"
    descripcion: str = "Inundación por crecida del río Piura"
    severidad_rio_factor: float = Field(default=1.0, ge=0.5, le=3.0)
    puentes: list[EstadoPuente] = Field(default_factory=list)
    flota: list[VehiculoConfig] = Field(default_factory=list)
    pesos: PesosObjetivo = Field(default_factory=PesosObjetivo)
    cuantico: ParametrosCuanticos = Field(default_factory=ParametrosCuanticos)


def cargar_escenario_json(ruta: str) -> EscenarioEmergencia:
    """Carga y valida un archivo de escenario en JSON."""
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return EscenarioEmergencia.model_validate(datos)


def guardar_escenario_json(escenario: EscenarioEmergencia, ruta: str):
    """Exporta el escenario validado a un archivo JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(escenario.model_dump_json(indent=2))
