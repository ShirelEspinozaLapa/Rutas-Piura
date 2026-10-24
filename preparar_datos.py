"""
Genera los archivos de datos/ a partir de fuentes abiertas.

No hace falta correrlo para usar el proyecto: los CSV y el GeoJSON ya vienen
generados en datos/. Se deja para que cualquiera pueda reproducir de dónde sale
cada coordenada.

Fuentes (clonar antes, en una carpeta aparte):
    git clone --depth 1 https://github.com/jmcastagnetto/ubigeo-peru-aumentado
    git clone --depth 1 https://github.com/juaneladio/peru-geojson

    python herramientas/preparar_datos.py --ext /ruta/a/la/carpeta/con/los/clones

- ubigeo-peru-aumentado (MIT): padrón de centros poblados del INEI con lat/lon
  (archivo ubigeo_ccpp.csv). De aquí salen las coordenadas de las comunidades.
- peru-geojson (MPL-2.0): límites distritales simplificados. Solo se extraen
  los distritos del departamento de Piura para el mapa base.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import unicodedata

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "..", "datos")

# ---------------------------------------------------------------------------
# Comunidades del escenario. Nombre INEI exacto + atributos del escenario.
# familias: familias afectadas (valor de escenario, orden de magnitud)
# prioridad_medica: 1 si la comunidad necesita medicinas con ventana de tiempo
# acceso_terrestre: "cortado" = vía anegada o cortada por quebrada (escenario)
# evento: referencia corta a la fuente periodística que motiva incluirla
# ---------------------------------------------------------------------------
COMUNIDADES = [
    # id, nombre visible, distrito, nombre INEI, margen, familias, prioridad, acceso, evento
    ("C01", "Simbilá", "CATACAOS", "SIMBILA", "MI", 220, 0, "operativo",
     "RPP 2017: punto vulnerable margen izquierda"),
    ("C02", "La Legua - San Jacinto", "CATACAOS", "LA LEGUA-SAN JACINTO (LA LEGUA)", "MI", 180, 0, "operativo",
     "Correo: erosión del dique San Jacinto-Catacaos"),
    ("C03", "Viduque", "CATACAOS", "VIDUQUE", "MI", 60, 1, "cortado",
     "Mongabay 2017: agua estancada"),
    ("C04", "Catacaos (ciudad)", "CATACAOS", "CATACAOS", "MI", 450, 1, "operativo",
     "Mongabay 2017: distrito totalmente inundado"),
    ("C05", "Narihualá", "CATACAOS", "NARIHUALA", "MI", 90, 1, "cortado",
     "Mongabay 2017 / Andina 2018: centro de salud inhabilitado"),
    ("C06", "Pedregal Chico", "CATACAOS", "PEDREGAL CHICO", "MI", 50, 0, "cortado",
     "Mongabay 2017: agua estancada"),
    ("C07", "Pedregal Grande", "CATACAOS", "PEDREGAL GRANDE", "MI", 110, 1, "cortado",
     "Mongabay 2017 / UDEP 2023: ayuda humanitaria"),
    ("C08", "Santa Rosa (Cura Mori)", "CURA MORI", "SANTA ROSA", "MI", 90, 1, "cortado",
     "Andina: dique reforzado; El Tiempo: sector más golpeado"),
    ("C09", "Cucungará", "CURA MORI", "CUCUNGARA", "MI", 120, 0, "cortado",
     "Mongabay 2017: Cura Mori totalmente inundado"),
    ("C10", "Chato Grande", "CURA MORI", "CHATO GRANDE", "MI", 70, 0, "cortado",
     "Mongabay 2017: caseríos más afectados"),
    ("C11", "Chato Chico", "CURA MORI", "CHATO CHICO", "MI", 50, 1, "cortado",
     "CENEPRED: inundación margen izquierda"),
    ("C12", "Pueblo Nuevo (Cura Mori)", "CURA MORI", "PUEBLO NUEVO", "MI", 60, 0, "operativo",
     "Cura Mori, sector oriental"),
    ("C13", "San Pablo", "CATACAOS", "SAN PABLO", "MI", 70, 0, "operativo",
     "UDEP 2017: proyecto de ayuda humanitaria"),
    ("C14", "Paredones", "CATACAOS", "PAREDONES", "MD", 90, 0, "operativo", "Catacaos, margen derecha"),
    ("C15", "Monte Castillo", "CATACAOS", "MONTE CASTILLO", "MD", 80, 0, "operativo", "Catacaos, margen derecha"),
    ("C16", "La Arena", "LA ARENA", "LA ARENA", "MD", 380, 1, "operativo",
     "CENEPRED: área urbana inundable"),
    ("C17", "Casagrande", "LA ARENA", "CASAGRANDE", "MD", 120, 0, "operativo", "La Arena"),
    ("C18", "Las Malvinas", "LA ARENA", "LAS MALVINAS", "MD", 100, 0, "operativo", "La Arena"),
    ("C19", "Chatito", "LA ARENA", "CHATITO", "MD", 70, 0, "operativo", "La Arena, ribereño"),
    ("C20", "La Unión", "LA UNION", "LA UNION", "MD", 300, 0, "operativo", "Mongabay 2017: zona de riesgo"),
    ("C21", "Sinchao (El Tallán)", "EL TALLAN", "SINCHAO", "MD", 130, 1, "operativo",
     "CENEPRED: El Tallán afectado"),
    ("C22", "Locuto (Tambo Grande)", "TAMBO GRANDE", "LOCUTO", "NA", 80, 1, "cortado",
     "RPP 2017: desborde en Locuto"),
    ("C23", "San Martín de Malingas", "TAMBO GRANDE", "SAN MARTIN DE MALINGAS", "NA", 100, 0, "cortado",
     "El Tiempo 2023: sin paso a Tambogrande ni a Piura"),
]

# Centros de acopio / bases. Coordenadas de Wikipedia (verificadas el 2026-10-24).
DEPOSITOS = [
    ("D1", "Aeropuerto de Piura / base aérea", "CASTILLA", "MI", -5.20556, -80.61667,
     "Wikipedia: PAF Captain Guillermo Concha Iberico International Airport",
     "Puente aéreo 2017 y helicópteros 2023 (Grupo Aéreo N.° 7)"),
    ("D2", "Campus UDEP (centro de acopio)", "PIURA", "MD", -5.17669, -80.63464,
     "Wikipedia: University of Piura",
     "UDEP 2017: centro de acopio Todos somos Piura"),
]


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").upper().strip()


def leer_ccpp(ruta: str) -> dict:
    """Devuelve {(distrito, ccpp): (codigo, lat, lon)} para la provincia de Piura."""
    out = {}
    with open(ruta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["departamento"] == "PIURA" and r["provincia"] == "PIURA":
                out[(_norm(r["distrito"]), _norm(r["ccpp"]))] = (
                    r["inei_ccpp"], float(r["latitude"]), float(r["longitude"]))
    return out


def main(ext: str) -> None:
    os.makedirs(DATOS, exist_ok=True)
    ccpp = leer_ccpp(os.path.join(ext, "ubigeo-peru-aumentado", "ubigeo_ccpp.csv"))

    filas = []
    for d in DEPOSITOS:
        id_, nombre, distrito, margen, lat, lon, fuente, evento = d
        filas.append(dict(id=id_, nombre=nombre, tipo="deposito", distrito=distrito, margen=margen,
                          lat=f"{lat:.6f}", lon=f"{lon:.6f}", codigo_inei="", familias=0,
                          prioridad_medica=0, acceso_terrestre="operativo",
                          fuente_coordenada=fuente, motivo=evento))
    for c in COMUNIDADES:
        id_, nombre, distrito, nombre_inei, margen, fam, prio, acceso, evento = c
        clave = (_norm(distrito), _norm(nombre_inei))
        if clave not in ccpp:
            raise KeyError(f"No encontré {clave} en el padrón INEI")
        cod, lat, lon = ccpp[clave]
        filas.append(dict(id=id_, nombre=nombre, tipo="comunidad", distrito=distrito, margen=margen,
                          lat=f"{lat:.6f}", lon=f"{lon:.6f}", codigo_inei=cod, familias=fam,
                          prioridad_medica=prio, acceso_terrestre=acceso,
                          fuente_coordenada="INEI centros poblados (ubigeo_ccpp.csv)", motivo=evento))

    with open(os.path.join(DATOS, "nodos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    # Límites distritales: solo departamento de Piura (MPL-2.0, ver datos/LEEME_DATOS.md)
    with open(os.path.join(ext, "peru-geojson", "peru_distrital_simple.geojson"), encoding="utf-8") as f:
        g = json.load(f)
    feats = []
    for ft in g["features"]:
        p = ft["properties"]
        if p.get("NOMBDEP") != "PIURA" or ft["geometry"] is None:
            continue
        prov = "PIURA" if p.get("NOMBPROV") == "PUIRA" else p.get("NOMBPROV")  # errata en la fuente
        feats.append({"type": "Feature",
                      "properties": {"distrito": p["NOMBDIST"], "provincia": prov, "ubigeo": p["IDDIST"]},
                      "geometry": ft["geometry"]})
    with open(os.path.join(DATOS, "limites_piura.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    print(f"nodos.csv: {len(filas)} filas | limites_piura.geojson: {len(feats)} distritos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True, help="carpeta con los repos clonados")
    main(ap.parse_args().ext)
