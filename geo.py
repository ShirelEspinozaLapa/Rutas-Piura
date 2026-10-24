"""Utilidades geográficas: distancias, proyección local y geometría plana."""
from __future__ import annotations

import numpy as np

R_TIERRA_KM = 6371.0088
LAT0 = -5.25          # latitud de referencia para la proyección local (Bajo Piura)
LON0 = -80.65


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia de gran círculo en km (acepta escalares o arrays)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_TIERRA_KM * np.arcsin(np.sqrt(a))


def matriz_haversine(lat, lon):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def a_km(lat, lon):
    """Proyección equirectangular local (x hacia el este, y hacia el norte) en km.
    Error < 0.1 % en la escala de la provincia: suficiente para geometría de red."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = np.radians(lon - LON0) * R_TIERRA_KM * np.cos(np.radians(LAT0))
    y = np.radians(lat - LAT0) * R_TIERRA_KM
    return np.stack([x, y], axis=-1)


def a_latlon(xy):
    """Inversa de a_km. Devuelve (lat, lon)."""
    xy = np.asarray(xy, float)
    lat = LAT0 + np.degrees(xy[..., 1] / R_TIERRA_KM)
    lon = LON0 + np.degrees(xy[..., 0] / (R_TIERRA_KM * np.cos(np.radians(LAT0))))
    return lat, lon


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segmentos_se_cruzan(p1, p2, q1, q2) -> bool:
    """True si los segmentos abiertos p1p2 y q1q2 se cruzan propiamente."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def cruza_polilinea(p1, p2, polilinea) -> bool:
    for k in range(len(polilinea) - 1):
        if segmentos_se_cruzan(p1, p2, polilinea[k], polilinea[k + 1]):
            return True
    return False


def dist_punto_segmento(p, a, b) -> float:
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = 0.0 if not np.any(ab) else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def dist_punto_polilinea(p, polilinea) -> float:
    return min(dist_punto_segmento(p, polilinea[k], polilinea[k + 1]) for k in range(len(polilinea) - 1))


def dist_segmento_polilinea(a, b, polilinea, muestras: int = 9) -> float:
    """Distancia mínima aproximada entre un tramo y el río (muestreo del tramo)."""
    a, b = np.asarray(a), np.asarray(b)
    return min(dist_punto_polilinea(a + t * (b - a), polilinea) for t in np.linspace(0, 1, muestras))


def lado_de_polilinea(p, polilinea) -> int:
    """+1 si el punto queda a la izquierda del sentido de la polilínea (aguas abajo),
    -1 si queda a la derecha. Con la polilínea orientada aguas abajo, izquierda = margen
    izquierda del río."""
    p = np.asarray(p)
    mejor, signo = np.inf, 0
    for k in range(len(polilinea) - 1):
        a, b = np.asarray(polilinea[k]), np.asarray(polilinea[k + 1])
        d = dist_punto_segmento(p, a, b)
        if d < mejor:
            mejor, signo = d, np.sign(_orient(a, b, p))
    return int(signo)
