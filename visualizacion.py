"""Mapas y gráficos de resultados (matplotlib, PNG).

Paleta: la paleta categórica de referencia (validada para daltonismo en el orden
usado aquí) + colores de estado reservados para puentes y tramos bloqueados.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from . import cuantico  # noqa: E402
from .datos import NOMBRES_DISTRITO  # noqa: E402
from . import qubo as qb  # noqa: E402

# ---------------------------------------------------------------- paleta
SUPERFICIE, PLANO = "#fcfcfb", "#f2f1ec"
TINTA, TINTA2, TENUE = "#0b0b0b", "#52514e", "#898781"
GRILLA, BASE = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OK, AVISO, CRITICO = "#0ca30c", "#fab219", "#d03b3b"
RIO, AGUA = "#9ec5f4", "#d9e8fb"
COLOR_METODO = {"QAOA": SERIES[0], "QAOA-XY": SERIES[1], "SA": SERIES[2], "GA": SERIES[3]}

EXT_GENERAL = (-80.80, -80.22, -5.46, -4.86)
EXT_DETALLE = (-80.775, -80.592, -5.428, -5.160)


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": BASE, "axes.labelcolor": TINTA2, "xtick.color": TENUE, "ytick.color": TENUE,
        "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
        "axes.grid": False, "grid.color": GRILLA, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
        "legend.frameon": False, "legend.fontsize": 8, "text.color": TINTA,
        "font.family": ["DejaVu Sans"],
    })


def _halo(lw=2.6):
    return [pe.withStroke(linewidth=lw, foreground=SUPERFICIE)]


def _color_grupo(inst, tipo, deposito):
    if tipo == "helicoptero":
        return SERIES[1]
    deps = sorted({g.deposito for g in inst.grupos if g.tipo == "camion"})
    return SERIES[0] if deposito == deps[0] else SERIES[2]




def _nombre_corto(n):
    return (n.nombre.replace(" (Cura Mori)", "").replace(" (Tambo Grande)", "").replace(" (ciudad)", "")
            .replace(" - San Jacinto", "").replace(" (El Tallán)", ""))


# ---------------------------------------------------------------- colocación de etiquetas sin solapes
_CANDIDATOS = [((6, -1), "left", "center"), ((-6, -1), "right", "center"), ((0, 7), "center", "bottom"),
               ((0, -7), "center", "top"), ((6, 6), "left", "bottom"), ((-6, 6), "right", "bottom"),
               ((6, -8), "left", "top"), ((-6, -8), "right", "top"), ((12, -1), "left", "center"),
               ((-12, -1), "right", "center"), ((0, 14), "center", "bottom"), ((0, -14), "center", "top")]


def _solape(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def colocar_etiquetas(ax, items, obstaculos):
    """items: dicts (x, y, texto, kw). obstaculos: lista de (x, y, radio_px) en datos.
    Prueba posiciones alrededor de cada punto y se queda con la de menor solape."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox

    ocupados = []
    for x, y, rad in obstaculos:
        px = ax.transData.transform((x, y))
        ocupados.append(Bbox.from_extents(px[0] - rad, px[1] - rad, px[0] + rad, px[1] + rad))
    caja_ax = ax.get_window_extent(r)
    for it in items:
        mejor = None
        for (dx, dy), ha, va in it.get("candidatos", _CANDIDATOS):
            t = ax.annotate(it["texto"], (it["x"], it["y"]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, annotation_clip=False, **it.get("kw", {}))
            bb = t.get_window_extent(r).expanded(1.04, 1.10)
            t.remove()
            fuera = (bb.x0 < caja_ax.x0) or (bb.x1 > caja_ax.x1) or (bb.y0 < caja_ax.y0) or (bb.y1 > caja_ax.y1)
            pen = sum(_solape(bb, o) for o in ocupados) + (1e7 if fuera else 0.0)
            if mejor is None or pen < mejor[0]:
                mejor = (pen, (dx, dy), ha, va, bb)
            if pen == 0:
                break
        _, (dx, dy), ha, va, bb = mejor
        ax.annotate(it["texto"], (it["x"], it["y"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, annotation_clip=True, **it.get("kw", {}))
        ocupados.append(bb)


# ---------------------------------------------------------------- mapa base
def _mapa_base(ax, inst, extent, detalle: bool, rotular_distritos=False):
    esc, red = inst.esc, inst.red
    x0, x1, y0, y1 = extent
    for ft in esc.limites.get("features", []):
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            xy = np.array(poly[0])
            if xy[:, 0].max() < x0 - 0.3 or xy[:, 0].min() > x1 + 0.3 or \
               xy[:, 1].max() < y0 - 0.3 or xy[:, 1].min() > y1 + 0.3:
                continue
            ax.add_patch(Polygon(xy, closed=True, facecolor=PLANO, edgecolor="#d9d8d0", lw=0.8, zorder=0))
        if rotular_distritos and ft["properties"].get("provincia") == "PIURA":
            xy = np.array(polys[0][0])
            cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
            if x0 + 0.02 < cx < x1 - 0.02 and y0 + 0.02 < cy < y1 - 0.02:
                nombre = NOMBRES_DISTRITO.get(ft["properties"]["distrito"], ft["properties"]["distrito"].title())
                ax.text(cx, cy, nombre, fontsize=6.6, color=TENUE, ha="center", va="center", zorder=1,
                        clip_on=True, style="italic")
    for n in esc.comunidades:
        if n.acceso_terrestre == "cortado" and n.lat < -5.1:
            ax.add_patch(Circle((n.lon, n.lat), 0.0075, facecolor=AGUA, edgecolor="none", zorder=1))
    ax.plot(esc.rio_latlon[:, 1], esc.rio_latlon[:, 0], color=RIO, lw=2.6 if detalle else 1.9,
            solid_capstyle="round", zorder=2)
    for u, v, d in red.G.edges(data=True):
        xs, ys = [red.lon[u], red.lon[v]], [red.lat[u], red.lat[v]]
        if d["puente"] is not None:
            col = {"operativo": OK, "restringido": AVISO}.get(d["estado"], CRITICO)
            ax.plot(xs, ys, color=col, lw=2.8 if detalle else 1.6, zorder=4, solid_capstyle="butt")
            continue
        if d["bloqueada"]:
            ax.plot(xs, ys, color=CRITICO, lw=1.1, ls=(0, (3, 2)), alpha=0.9, zorder=3)
        else:
            lw = {"carretera": 1.8, "urbana": 1.2, "afirmado": 1.0, "trocha": 1.1}[d["tipo"]]
            ls = (0, (1, 1.6)) if d["tipo"] == "trocha" else "-"
            ax.plot(xs, ys, color=TENUE, lw=lw, ls=ls, alpha=0.8, zorder=3, solid_capstyle="round")
    for p in esc.puentes:
        col, mk = {"operativo": (OK, "^"), "restringido": (AVISO, "D")}.get(p.estado, (CRITICO, "X"))
        ax.scatter(p.lon, p.lat, s=80 if detalle else 38, marker=mk, color=col, edgecolor=SUPERFICIE,
                   linewidth=1.2, zorder=8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRILLA)


def _dibujar_nodos(ax, inst, detalle):
    esc = inst.esc
    for d in esc.depositos:
        ax.scatter(d.lon, d.lat, s=260 if detalle else 150, marker="*", color=TINTA, edgecolor=SUPERFICIE,
                   linewidth=1.5, zorder=12)
    for n in esc.comunidades:
        i = n.idx
        col = _color_grupo(inst, inst.tipo_por_nodo[i], inst.deposito_por_nodo[i])
        ax.scatter(n.lon, n.lat, s=(22 if detalle else 12) + inst.demanda[i] / (40 if detalle else 70), color=col,
                   edgecolor=TINTA if n.prioridad_medica else SUPERFICIE,
                   linewidth=1.5 if n.prioridad_medica else 1.0, zorder=10)


def _obstaculos(inst, ext):
    x0, x1, y0, y1 = ext
    obs = [(n.lon, n.lat, 6) for n in inst.esc.nodos if x0 < n.lon < x1 and y0 < n.lat < y1]
    obs += [(p.lon, p.lat, 6) for p in inst.esc.puentes if x0 < p.lon < x1 and y0 < p.lat < y1]
    return obs


def _items_nombres(inst, ext, solo=None, puentes=False):
    x0, x1, y0, y1 = ext
    items = []
    for d in inst.esc.depositos:
        items.append(dict(x=d.lon, y=d.lat, texto="Aeropuerto" if d.id == "D1" else "UDEP",
                          kw=dict(fontsize=7.4, fontweight="bold", color=TINTA, path_effects=_halo(3), zorder=13)))
    for n in inst.esc.comunidades:
        if not (x0 < n.lon < x1 and y0 < n.lat < y1):
            continue
        if solo is not None and n.idx not in solo:
            continue
        texto = _nombre_corto(n) + ("\n(quebrada activa)" if n.acceso_terrestre == "cortado" and n.lat > -5.1 else "")
        items.append(dict(x=n.lon, y=n.lat, texto=texto,
                          kw=dict(fontsize=6.6, color=TINTA2, path_effects=_halo(), zorder=11)))
    if puentes:
        for p in inst.esc.puentes:
            if x0 < p.lon < x1 and y0 < p.lat < y1:
                items.append(dict(x=p.lon, y=p.lat, texto=p.id,
                                  kw=dict(fontsize=6.8, fontweight="bold", color=TINTA, path_effects=_halo(),
                                          zorder=11)))
    return items


def _leyenda_escenario():
    return [Line2D([], [], color=RIO, lw=2.6, label="Río Piura (trazado aprox.)"),
            Line2D([], [], color=TENUE, lw=1.8, label="Vía transitable"),
            Line2D([], [], color=TENUE, lw=1.1, ls=(0, (1, 1.6)), label="Trocha de desvío"),
            Line2D([], [], color=CRITICO, lw=1.1, ls=(0, (3, 2)), label="Tramo bloqueado"),
            Line2D([], [], color=OK, marker="^", ls="", ms=8, label="Puente operativo"),
            Line2D([], [], color=AVISO, marker="D", ls="", ms=7, label="Puente restringido"),
            Line2D([], [], color=CRITICO, marker="X", ls="", ms=8, label="Puente cerrado o colapsado"),
            Line2D([], [], color=AGUA, marker="o", ls="", ms=11, label="Zona anegada (escenario)"),
            Line2D([], [], color=TINTA, marker="*", ls="", ms=12, label="Centro de acopio"),
            Line2D([], [], color=SERIES[1], marker="o", ls="", ms=8, label="Atención por helicóptero"),
            Line2D([], [], color=SERIES[0], marker="o", ls="", ms=8, label="Camión desde el aeropuerto"),
            Line2D([], [], color=SERIES[2], marker="o", ls="", ms=8, label="Camión desde UDEP"),
            Line2D([], [], color=SUPERFICIE, marker="o", ls="", ms=8, markeredgecolor=TINTA, markeredgewidth=1.5,
                   label="Prioridad médica (ventana)"),
            Line2D([], [], color=TENUE, marker="o", ls="", ms=4, label="tamaño ∝ demanda (kg)")]


def _texto_puentes(inst):
    return "   ".join(f"{p.id} {p.nombre.split(' (')[0]}: {p.estado}" for p in inst.esc.puentes)


def mapa_escenario(inst, ruta):
    estilo()
    fig = plt.figure(figsize=(15.5, 11))
    ax_g = fig.add_axes([0.015, 0.30, 0.43, 0.62])
    ax_d = fig.add_axes([0.46, 0.06, 0.53, 0.86])
    _mapa_base(ax_g, inst, EXT_GENERAL, False, rotular_distritos=True)
    _dibujar_nodos(ax_g, inst, False)
    x0, x1, y0, y1 = EXT_DETALLE
    ax_g.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=TINTA2, lw=1.0, zorder=14)
    ax_g.set_title("Medio y Bajo Piura", loc="left")
    _mapa_base(ax_d, inst, EXT_DETALLE, True)
    _dibujar_nodos(ax_d, inst, True)
    ax_d.set_title("Bajo Piura (recuadro de la izquierda)", loc="left")
    fig.canvas.draw()
    medio = {n.idx for n in inst.esc.comunidades if n.lat > -5.1}
    colocar_etiquetas(ax_g, [it for it in _items_nombres(inst, EXT_GENERAL, solo=medio)
                             if it["texto"] not in ("Aeropuerto", "UDEP")], _obstaculos(inst, EXT_GENERAL))
    colocar_etiquetas(ax_d, _items_nombres(inst, EXT_DETALLE, puentes=True), _obstaculos(inst, EXT_DETALLE))
    fig.legend(handles=_leyenda_escenario(), loc="lower left", bbox_to_anchor=(0.015, 0.035), ncol=2,
               fontsize=8, frameon=False, handletextpad=0.6, columnspacing=1.4, labelspacing=0.7)
    fig.text(0.015, 0.012, "Puentes: " + _texto_puentes(inst), fontsize=7.6, color=TINTA2)
    fig.suptitle("Escenario: desborde del río Piura · comunidades con coordenadas del padrón INEI; red vial y "
                 "estado de puentes son supuestos del escenario", x=0.015, ha="left", fontsize=11.5,
                 fontweight="bold", y=0.985)
    fig.savefig(ruta, dpi=165)
    plt.close(fig)


def mapa_clusters(inst, clusters, ruta):
    estilo()
    fig = plt.figure(figsize=(15.5, 11))
    ax_g = fig.add_axes([0.015, 0.30, 0.43, 0.62])
    ax_d = fig.add_axes([0.46, 0.06, 0.53, 0.86])
    items_d, items_g = [], []
    for ax, ext, det in ((ax_g, EXT_GENERAL, False), (ax_d, EXT_DETALLE, True)):
        _mapa_base(ax, inst, ext, det)
        for c in clusters:
            col = _color_grupo(inst, c.tipo, c.deposito)
            lon = np.array([inst.esc.nodos[i].lon for i in c.nodos])
            lat = np.array([inst.esc.nodos[i].lat for i in c.nodos])
            cx, cy = lon.mean(), lat.mean()
            for x, y in zip(lon, lat):
                ax.plot([cx, x], [cy, y], color=col, lw=1.6, alpha=0.95, zorder=9, solid_capstyle="round")
            ax.scatter(cx, cy, s=30, marker="s", color=col, edgecolor=SUPERFICIE, linewidth=1.0, zorder=11)
            it = dict(x=cx, y=cy, texto=f"{c.id} ({len(c.nodos)}·{len(c.nodos) ** 2}q)",
                      kw=dict(fontsize=7.6, fontweight="bold", color=TINTA, path_effects=_halo(3), zorder=14,
                              bbox=dict(boxstyle="round,pad=0.18", facecolor=SUPERFICIE, edgecolor=col, lw=1.1)))
            (items_d if det else items_g).append(it)
        _dibujar_nodos(ax, inst, det)
    fig.canvas.draw()
    x0, x1, y0, y1 = EXT_DETALLE
    ax_g.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=TINTA2, lw=1.0, zorder=14)
    colocar_etiquetas(ax_g, [i for i in items_g if i["y"] > -5.12], _obstaculos(inst, EXT_GENERAL))
    colocar_etiquetas(ax_d, items_d + _items_nombres(inst, EXT_DETALLE), _obstaculos(inst, EXT_DETALLE))
    ax_g.set_title("Vista general", loc="left")
    ax_d.set_title("Bajo Piura: cada cluster es un viaje y un subproblema QAOA (paradas · qubits)", loc="left")
    h = [Line2D([], [], color=SERIES[1], lw=2, marker="s", label="Viaje de helicóptero (Aéreo-n)"),
         Line2D([], [], color=SERIES[0], lw=2, marker="s", label="Camión desde el aeropuerto (Terr-An)"),
         Line2D([], [], color=SERIES[2], lw=2, marker="s", label="Camión desde UDEP (Terr-Un)"),
         Line2D([], [], color=SUPERFICIE, marker="o", ls="", ms=8, markeredgecolor=TINTA, markeredgewidth=1.5,
                label="Prioridad médica"),
         Line2D([], [], color=TINTA, marker="*", ls="", ms=12, label="Centro de acopio")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.015, 0.06), fontsize=8.5, frameon=False,
               labelspacing=0.8)
    fig.suptitle("Fase clásica: triage modal + k-medoides capacitado + refinamiento local "
                 "(máx. 4 paradas por viaje = 16 qubits)", x=0.015, ha="left", fontsize=11.5,
                 fontweight="bold", y=0.985)
    fig.savefig(ruta, dpi=165)
    plt.close(fig)


# ---------------------------------------------------------------- rutas
def _dibujar_ruta(ax, inst, ej, col, detalle):
    red = inst.red
    seq = [ej.plan.deposito] + list(ej.plan.nodos) + [ej.plan.deposito]
    for a, b in zip(seq[:-1], seq[1:]):
        if ej.plan.tipo == "camion":
            cam = inst.mats["camion"].caminos.get((a, b))
            xs = red.lon[cam] if cam else np.array([red.lon[a], red.lon[b]])
            ys = red.lat[cam] if cam else np.array([red.lat[a], red.lat[b]])
            ax.plot(xs, ys, color=col, lw=3.0 if detalle else 2.0, alpha=0.85, zorder=6,
                    solid_capstyle="round", solid_joinstyle="round")
            k = max(0, len(xs) // 2 - 1)
            ax.annotate("", xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0, mutation_scale=10), zorder=7)
        else:
            x0, y0, x1, y1 = red.lon[a], red.lat[a], red.lon[b], red.lat[b]
            t = np.linspace(0, 1, 40)
            cxp, cyp = (x0 + x1) / 2 - 0.13 * (y1 - y0), (y0 + y1) / 2 + 0.13 * (x1 - x0)
            xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cxp + t ** 2 * x1
            ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cyp + t ** 2 * y1
            ax.plot(xs, ys, color=col, lw=2.1 if detalle else 1.6, alpha=0.95, zorder=6)
            ax.annotate("", xy=(xs[21], ys[21]), xytext=(xs[19], ys[19]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0, mutation_scale=10), zorder=7)


def _etiquetas_rutas(sol):
    cuenta, out = {}, []
    for r in sorted(sol.rutas, key=lambda r: (r.vehiculo, r.inicio_h)):
        cuenta[r.vehiculo] = cuenta.get(r.vehiculo, 0) + 1
        varios = sum(1 for x in sol.rutas if x.vehiculo == r.vehiculo) > 1
        out.append((r, f"{r.vehiculo}·{cuenta[r.vehiculo]}" if varios else r.vehiculo))
    return out


def mapa_rutas(inst, soluciones: dict, ruta):
    estilo()
    nombres = list(soluciones)
    ncol = len(nombres)
    ancho = 0.96 / ncol
    fig = plt.figure(figsize=(7.6 * ncol, 17.2))
    for j, nombre in enumerate(nombres):
        sol = soluciones[nombre]
        ax_d = fig.add_axes([0.02 + j * ancho, 0.375, ancho - 0.02, 0.555])
        ax_g = fig.add_axes([0.02 + j * ancho, 0.055, ancho - 0.02, 0.29])
        items = {id(ax_d): [], id(ax_g): []}
        for ax, ext, det in ((ax_d, EXT_DETALLE, True), (ax_g, EXT_GENERAL, False)):
            _mapa_base(ax, inst, ext, det)
            for ej, et in _etiquetas_rutas(sol):
                col = _color_grupo(inst, ej.plan.tipo, ej.plan.deposito)
                _dibujar_ruta(ax, inst, ej, col, det)
                n0 = inst.esc.nodos[ej.plan.nodos[0]]
                x0, x1, y0, y1 = ext
                if det and x0 < n0.lon < x1 and y0 < n0.lat < y1 or (not det and n0.lat > -5.1):
                    items[id(ax)].append(dict(
                        x=n0.lon, y=n0.lat, texto=et,
                        kw=dict(fontsize=7.6 if det else 6.8, fontweight="bold", color=TINTA, zorder=15,
                                bbox=dict(boxstyle="round,pad=0.18", facecolor=SUPERFICIE, edgecolor=col, lw=1.2))))
                if det:
                    for k, i in enumerate(ej.plan.nodos):
                        n = inst.esc.nodos[i]
                        if x0 < n.lon < x1 and y0 < n.lat < y1:
                            items[id(ax)].append(dict(
                                x=n.lon, y=n.lat, texto=str(k + 1),
                                candidatos=[((7, 7), "center", "center"), ((-7, 7), "center", "center"),
                                            ((7, -7), "center", "center"), ((-7, -7), "center", "center"),
                                            ((10, 0), "center", "center"), ((-10, 0), "center", "center")],
                                kw=dict(fontsize=6.4, fontweight="bold", color="#ffffff", zorder=16,
                                        bbox=dict(boxstyle="circle,pad=0.18", facecolor=col, edgecolor=SUPERFICIE,
                                                  lw=0.8))))
            _dibujar_nodos(ax, inst, det)
        fig.canvas.draw()
        colocar_etiquetas(ax_d, items[id(ax_d)] + _items_nombres(inst, EXT_DETALLE), _obstaculos(inst, EXT_DETALLE))
        colocar_etiquetas(ax_g, items[id(ax_g)], _obstaculos(inst, EXT_GENERAL))
        c = sol.componentes
        ax_d.set_title(f"{nombre}\nJ = {c['J']:.2f} · entrega acumulada {c['tiempo_total_entrega_h']:.1f} h · "
                       f"combustible {c['combustible_l']:.0f} L · ventanas incumplidas {c['violaciones_ventana']}",
                       loc="left", fontsize=10)
        ax_g.set_title("Vista general (vuelos a Tambo Grande)", loc="left", fontsize=9)
    h = [Line2D([], [], color=SERIES[1], lw=2.1, label="Helicóptero (vuelo directo)"),
         Line2D([], [], color=SERIES[0], lw=3, label="Camión desde el aeropuerto (por la red vial)"),
         Line2D([], [], color=SERIES[2], lw=3, label="Camión desde UDEP (por la red vial)"),
         Line2D([], [], color=CRITICO, lw=1.1, ls=(0, (3, 2)), label="Tramo bloqueado"),
         Line2D([], [], color=CRITICO, marker="X", ls="", ms=8, label="Puente cerrado o colapsado"),
         Line2D([], [], color=TINTA, marker="*", ls="", ms=12, label="Centro de acopio"),
         Line2D([], [], color=TENUE, marker="o", ls="", ms=9, markerfacecolor=SERIES[0],
                markeredgecolor=SUPERFICIE, label="1, 2, 3… orden de visita en el viaje")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Rutas seleccionadas: clásico puro frente al híbrido clásico-cuántico "
                 "(recuadro = vehículo·número de viaje)", x=0.02, ha="left", fontsize=12.5,
                 fontweight="bold", y=0.99)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- gráficos de métricas
def convergencia(R, ruta):
    estilo()
    subs = [s for s in R.subproblemas if s.n > 1]
    ncol = 3
    nfil = int(np.ceil(len(subs) / ncol))
    fig, axs = plt.subplots(nfil, ncol, figsize=(13.5, 3.6 * nfil + 0.8), squeeze=False)
    piso = 1e-3
    for k, sp in enumerate(subs):
        ax = axs[k // ncol, k % ncol]
        d = R.sub[sp.id]
        E_opt = d["E_opt"]

        def brecha(e):
            return np.maximum(100 * (np.asarray(e) - E_opt) / abs(E_opt), piso)

        for var, et in (("qaoa_estandar", "QAOA"), ("qaoa_xy", "QAOA-XY")):
            h = d[var]["historia"]
            ax.plot(np.arange(1, len(h) + 1), np.minimum.accumulate(brecha(h)), color=COLOR_METODO[et], lw=2,
                    label=f"{et}: CVaR de la distribución")
        sa = d["sa"]["curva"]
        ax.plot(np.arange(1, len(sa) + 1) * sp.nq * R.cfg.sa_lecturas, brecha(sa), color=COLOR_METODO["SA"], lw=2,
                label="SA: mejor lectura")
        ga = d["ga"]["curva"]
        ax.plot(np.arange(1, len(ga) + 1) * R.cfg.ga_poblacion, brecha(ga), color=COLOR_METODO["GA"], lw=2,
                label="GA: mejor individuo")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(piso * 0.6, None)
        ax.grid(True, which="major", axis="y")
        ax.set_title(f"{sp.id} · {sp.n} paradas · {sp.nq} qubits", fontsize=9.5, loc="left")
        if k // ncol == nfil - 1 or k + ncol >= len(subs):
            ax.set_xlabel("evaluaciones de la función objetivo")
        if k % ncol == 0:
            ax.set_ylabel("brecha al óptimo (%)")
    for k in range(len(subs), nfil * ncol):
        axs[k // ncol, k % ncol].axis("off")
    hs, ls = axs[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper left", ncol=4, bbox_to_anchor=(0.01, 0.985), fontsize=8.5)
    fig.text(0.01, 0.935, "Mismo QUBO para todos. Una evaluación del QAOA = ejecutar el circuito completo "
             "(miles de disparos en un QPU); una de SA = proponer un volteo de bit; una de GA = evaluar una ruta.\n"
             "El QAOA reporta la CVaR de toda su distribución (no la mejor muestra), por eso no siempre llega a 0. "
             "Brecha 0 se dibuja en 0,001 %.", ha="left", fontsize=8.3, color=TINTA2)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def distribucion_qaoa(R, ruta, sp_id=None):
    estilo()
    cands = [s for s in R.subproblemas if s.nq >= 9]
    sp = next((s for s in cands if s.id == sp_id), None) or max(cands, key=lambda s: (s.nq, s.id))
    d = R.sub[sp.id]
    esc = R.inst.esc
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, var, et in ((axs[0], "qaoa_estandar", "QAOA estándar"), (axs[1], "qaoa_xy", "QAOA-XY")):
        top = d[var]["probs_top"][:16]
        cols = [OK if t["optimo"] else (SERIES[0] if t["factible"] else BASE) for t in top]
        ax.bar(range(len(top)), [t["prob"] for t in top], width=0.62, color=cols)
        etq = []
        for t in top:
            pm = qb.x_a_perm(qb.indice_a_x(t["indice"], sp.nq), sp.n)
            etq.append("infactible" if pm is None else "→".join(esc.nodos[sp.nodos[i]].id for i in pm))
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(etq, rotation=70, fontsize=6.6, ha="right")
        ax.set_title(f"{et} (p = {d[var]['p']}) · P(óptimo) = {d[var]['p_optimo']:.4f} · "
                     f"P(ruta válida) = {d[var]['p_factible']:.3f}", fontsize=9.5, loc="left")
        ax.grid(True, axis="y")
        ax.set_ylabel("probabilidad de medir el estado")
    h = [Line2D([], [], color=OK, marker="s", ls="", ms=8, label="ruta óptima"),
         Line2D([], [], color=SERIES[0], marker="s", ls="", ms=8, label="ruta válida no óptima"),
         Line2D([], [], color=BASE, marker="s", ls="", ms=8, label="bitstring infactible")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    nombres = ", ".join(esc.nodos[i].id + " " + _nombre_corto(esc.nodos[i]) for i in sp.nodos)
    fig.suptitle(f"Los 16 estados más probables al final del QAOA · {sp.id} ({sp.nq} qubits): {nombres}",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
    return sp.id


def paisaje_p1(R, ruta, sp_id, malla=36):
    """P(ruta óptima) con una capa de QAOA en función de (gamma, beta)."""
    estilo()
    sp = next(s for s in R.subproblemas if s.id == sp_id)
    prep = cuantico.preparar(sp)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, var, et in ((axs[0], "estandar", "QAOA estándar"), (axs[1], "xy", "QAOA-XY")):
        info = R.sub[sp.id][f"qaoa_{var}"]
        if var == "estandar":
            sim = cuantico.SimuladorQAOA(prep.E_total / sp.A, sp.nq, "x")
        else:
            esc_ = info["escala"]
            E = qb.energias_todas(sp.Qc / esc_, sp.const_costo / esc_)
            sim = cuantico.SimuladorQAOA(E, sp.nq, "swap", n=sp.n, orden_inicial=info["orden_inicial"])
        gs = np.linspace(0, np.pi, malla)
        bs = np.linspace(0, np.pi / 2, malla)
        Z = np.zeros((malla, malla))
        for a, g in enumerate(gs):
            for b, bb in enumerate(bs):
                Z[b, a] = sim.probabilidades([g], [bb])[prep.optimos].sum()
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=(0, np.pi, 0, np.pi / 2), cmap="Blues", vmin=0)
        ax.set_xlabel("γ (ángulo de la capa de costo)")
        ax.set_ylabel("β (ángulo del mezclador)")
        ax.set_title(f"{et}: P(ruta óptima) con p = 1 (máx. {Z.max():.3f})", fontsize=9.5, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    fig.text(0.01, 0.015, "En QAOA-XY el estado inicial es una ruta clásica: la primera capa de costo solo le "
             "agrega una fase global, por eso con p = 1 el resultado no depende de γ y hacen falta más capas.",
             fontsize=8, color=TINTA2)
    fig.suptitle(f"Paisaje de una capa · subproblema {sp.id} ({sp.nq} qubits)", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def metricas(R, ruta):
    estilo()
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    nombres = sorted(R.soluciones, key=lambda n: R.soluciones[n].J)
    comp = ["termino_tiempo", "termino_combustible", "termino_riesgo", "termino_atraso"]
    etq = ["tiempo de entrega", "combustible", "riesgo", "atraso de medicinas"]
    izq = np.zeros(len(nombres))
    for k, (c, e) in enumerate(zip(comp, etq)):
        vals = np.array([R.soluciones[n].componentes[c] for n in nombres])
        ax.barh(range(len(nombres)), vals, left=izq, height=0.56, color=SERIES[k], label=e,
                edgecolor=SUPERFICIE, linewidth=1.5)
        izq += vals
    for i, n in enumerate(nombres):
        ax.text(izq[i] + 0.25, i, f"J = {R.soluciones[n].J:.2f}", va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels(nombres, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, izq.max() * 1.1)
    ax.set_xlabel("función objetivo J (menor es mejor)")
    ax.grid(True, axis="x")
    ax.legend(loc="lower left", ncol=4, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Calidad de la solución completa (misma función exacta para todos)", loc="left", pad=26)

    subs = [s for s in R.subproblemas if s.n > 1]
    x = np.arange(len(subs))
    w = 0.26
    ax = fig.add_subplot(gs[1, 0])
    series = [("QAOA", lambda d: d["qaoa_estandar"]["p_optimo"]),
              ("QAOA-XY", lambda d: d["qaoa_xy"]["p_optimo"]),
              ("SA", lambda d: d["sa"]["p_optimo_medio"])]
    for k, (et, f) in enumerate(series):
        ax.bar(x + (k - 1) * w, [max(f(R.sub[s.id]), 1e-5) for s in subs], width=w * 0.9,
               color=COLOR_METODO[et], label=et)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.id}\n{s.nq} q" for s in subs], fontsize=8)
    ax.set_ylabel("P(óptimo) en un disparo / lectura")
    ax.grid(True, axis="y")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Probabilidad de acertar el óptimo en un intento", loc="left", pad=22)

    ax = fig.add_subplot(gs[1, 1])
    series = [("SA (CPU)", lambda d: d["sa"]["tiempo_s"], COLOR_METODO["SA"]),
              ("QAOA simulado (CPU)", lambda d: d["qaoa_estandar"]["tiempo_s"], COLOR_METODO["QAOA"]),
              ("QAOA en QPU (estimado)", lambda d: d["qaoa_estandar"]["qpu"]["tiempo_entrenamiento_qpu_s"],
               SERIES[6])]
    for k, (et, f, col) in enumerate(series):
        ax.bar(x + (k - 1) * w, [f(R.sub[s.id]) for s in subs], width=w * 0.9, color=col, label=et)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.id}\n{s.nq} q" for s in subs], fontsize=8)
    ax.set_ylabel("segundos (escala log)")
    ax.grid(True, axis="y")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Tiempo hasta converger por subproblema", loc="left", pad=22)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generar_figuras(R, dir_salida) -> dict:
    os.makedirs(dir_salida, exist_ok=True)
    rutas = {}

    def destino(nombre):
        return os.path.join(dir_salida, nombre)

    mapa_escenario(R.inst, destino("1_mapa_escenario.png"))
    rutas["escenario"] = destino("1_mapa_escenario.png")
    mapa_clusters(R.inst, R.clusters, destino("2_mapa_clusters.png"))
    rutas["clusters"] = destino("2_mapa_clusters.png")
    puro = min((n for n in R.soluciones if n.startswith("Clásico puro")), key=lambda n: R.soluciones[n].J)
    hib, hib_xy = "Híbrido: clustering + QAOA", "Híbrido: clustering + QAOA-XY"
    mismas = [tuple(r.plan.nodos) for r in R.soluciones[hib].rutas] == \
             [tuple(r.plan.nodos) for r in R.soluciones[hib_xy].rutas]
    sel = {puro: R.soluciones[puro], hib + (" (QAOA-XY da las mismas rutas)" if mismas else ""): R.soluciones[hib]}
    if not mismas:
        sel[hib_xy] = R.soluciones[hib_xy]
    mapa_rutas(R.inst, sel, destino("3_mapa_rutas_clasico_vs_hibrido.png"))
    rutas["rutas"] = destino("3_mapa_rutas_clasico_vs_hibrido.png")
    convergencia(R, destino("4_convergencia.png"))
    rutas["convergencia"] = destino("4_convergencia.png")
    sid = distribucion_qaoa(R, destino("5_qaoa_distribucion.png"))
    rutas["distribucion"] = destino("5_qaoa_distribucion.png")
    paisaje_p1(R, destino("6_qaoa_paisaje_p1.png"), sid, malla=26 if R.cfg.rapido else 36)
    rutas["paisaje"] = destino("6_qaoa_paisaje_p1.png")
    metricas(R, destino("7_metricas_comparacion.png"))
    rutas["metricas"] = destino("7_metricas_comparacion.png")
    return rutas

