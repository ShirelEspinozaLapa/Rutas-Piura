"""Carga del escenario: nodos (depósitos y comunidades), puentes, río y límites."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import geo
from .config import DIR_DATOS, Config


NOMBRES_DISTRITO = {"EL TALLAN": "El Tallán", "LA UNION": "La Unión", "TAMBO GRANDE": "Tambo Grande",
                    "CURA MORI": "Cura Mori", "LA ARENA": "La Arena", "CATACAOS": "Catacaos",
                    "CASTILLA": "Castilla", "PIURA": "Piura", "LAS LOMAS": "Las Lomas",
                    "VEINTISEIS DE OCTUBRE": "Veintiséis de Octubre"}


@dataclass
class Nodo:
    idx: int
    id: str
    nombre: str
    tipo: str                 # "deposito" | "comunidad"
    distrito: str
    margen: str               # "MD" | "MI" | "NA"
    lat: float
    lon: float
    familias: int
    prioridad_medica: int
    acceso_terrestre: str     # "operativo" | "cortado"
    codigo_inei: str
    motivo: str
    demanda_kg: float = 0.0
    medicina_kg: float = 0.0
    xy: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def distrito_nombre(self) -> str:
        return NOMBRES_DISTRITO.get(self.distrito, self.distrito.title())


@dataclass
class Puente:
    id: str
    nombre: str
    lat: float
    lon: float
    estado: str               # "operativo" | "restringido" | "cerrado" | "colapsado"
    nota: str
    xy: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def transitable(self) -> bool:
        return self.estado in ("operativo", "restringido")


@dataclass
class Escenario:
    nodos: list
    puentes: list
    rio_latlon: np.ndarray    # (k, 2) lat, lon en sentido aguas abajo
    rio_xy: np.ndarray        # (k, 2) km
    limites: dict

    @property
    def depositos(self):
        return [n for n in self.nodos if n.tipo == "deposito"]

    @property
    def comunidades(self):
        return [n for n in self.nodos if n.tipo == "comunidad"]

    def por_id(self, id_: str) -> Nodo:
        for n in self.nodos:
            if n.id == id_:
                return n
        raise KeyError(id_)

    @property
    def lat(self):
        return np.array([n.lat for n in self.nodos])

    @property
    def lon(self):
        return np.array([n.lon for n in self.nodos])


def cargar_escenario(cfg: Config, dir_datos: str = DIR_DATOS) -> Escenario:
    nodos = []
    with open(os.path.join(dir_datos, "nodos.csv"), newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            n = Nodo(idx=i, id=r["id"], nombre=r["nombre"], tipo=r["tipo"], distrito=r["distrito"],
                     margen=r["margen"], lat=float(r["lat"]), lon=float(r["lon"]),
                     familias=int(r["familias"]), prioridad_medica=int(r["prioridad_medica"]),
                     acceso_terrestre=r["acceso_terrestre"], codigo_inei=r["codigo_inei"],
                     motivo=r["motivo"])
            n.demanda_kg = n.familias * cfg.kg_por_familia
            n.medicina_kg = n.familias * cfg.kg_medicina_por_familia * (1 if n.prioridad_medica else 0.5)
            n.xy = geo.a_km(n.lat, n.lon)
            nodos.append(n)

    puentes = []
    with open(os.path.join(dir_datos, "puentes.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Puente(id=r["id"], nombre=r["nombre"], lat=float(r["lat"]), lon=float(r["lon"]),
                       estado=r["estado"], nota=r["nota"])
            p.xy = geo.a_km(p.lat, p.lon)
            puentes.append(p)

    rio = []
    with open(os.path.join(dir_datos, "rio_piura_aprox.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rio.append((float(r["lat"]), float(r["lon"])))
    rio = np.array(rio)

    ruta_lim = os.path.join(dir_datos, "limites_piura.geojson")
    limites = {"features": []}
    if os.path.exists(ruta_lim):
        with open(ruta_lim, encoding="utf-8") as f:
            limites = json.load(f)

    return Escenario(nodos=nodos, puentes=puentes, rio_latlon=rio,
                     rio_xy=geo.a_km(rio[:, 0], rio[:, 1]), limites=limites)
