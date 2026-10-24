import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from solvers import NODOS, construir_grafo, dijkstra, qaoa_seguro as qaoa

st.set_page_config(page_title="Rutas Piura - El Niño", layout="wide")
st.title("Rutas de emergencia en Piura ante El Niño")
st.caption("Comparación entre un algoritmo clásico (Dijkstra) y un enfoque híbrido cuántico-clásico (QAOA en Qiskit). Datos ilustrativos.")

with st.sidebar:
    nivel = st.slider("Intensidad de El Niño", 0.0, 1.0, 0.6, 0.05)
    origen = st.selectbox("Origen", list(NODOS), index=1)
    destino = st.selectbox("Destino", list(NODOS), index=5)
    reps = st.slider("Capas QAOA (p)", 1, 3, 2)
    iters = st.slider("Iteraciones del optimizador", 40, 200, 120, 10)
    ejecutar = st.button("Calcular rutas", type="primary")

G, bloqueados = construir_grafo(nivel)


def dibujar(ax, titulo, ruta):
    for a, b in bloqueados:
        ax.plot([NODOS[a][1], NODOS[b][1]], [NODOS[a][0], NODOS[b][0]], "r--", lw=1)
    for a, b in G.edges:
        ax.plot([NODOS[a][1], NODOS[b][1]], [NODOS[a][0], NODOS[b][0]], color="lightgray", lw=1.5)
    if ruta:
        xs = [NODOS[n][1] for n in ruta]; ys = [NODOS[n][0] for n in ruta]
        ax.plot(xs, ys, color="green", lw=3)
    for n, (lat, lon) in NODOS.items():
        ax.scatter(lon, lat, c="navy", s=30, zorder=3)
        ax.annotate(n, (lon, lat), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(titulo, fontsize=10); ax.axis("off")


if origen == destino:
    st.warning("Elige un origen y un destino distintos.")
elif ejecutar:
    if not (origen in G and destino in G) or not __import__("networkx").has_path(G, origen, destino):
        st.error("Con esta intensidad, no hay ruta transitable entre esos puntos.")
        fig, ax = plt.subplots(figsize=(5, 4)); dibujar(ax, "Red vial (rojo = bloqueado)", None); st.pyplot(fig)
    else:
        r_c, c_c, t_c = dijkstra(G, origen, destino)
        with st.spinner("Ejecutando QAOA en simulador..."):
            r_q = qaoa(G, origen, destino, reps=reps, maxiter=iters)
        col1, col2 = st.columns(2)
        for col, titulo, ruta in ((col1, "Clásico: Dijkstra", r_c), (col2, "Híbrido: QAOA + COBYLA", r_q["ruta"])):
            fig, ax = plt.subplots(figsize=(5, 4)); dibujar(ax, titulo, ruta); col.pyplot(fig)
        df = pd.DataFrame([
            {"Método": "Dijkstra", "Ruta": " → ".join(r_c), "Costo (min)": round(c_c, 1),
             "Tiempo (s)": round(t_c, 4), "Qubits": 0, "Prob. estado válido": "-"},
            {"Método": f"QAOA (p={reps})", "Ruta": " → ".join(r_q["ruta"]) if r_q["ruta"] else "sin solución",
             "Costo (min)": round(r_q["costo"], 1) if r_q["costo"] else None, "Tiempo (s)": round(r_q["tiempo"], 2),
             "Qubits": r_q["qubits"], "Prob. estado válido": f"{float(r_q['prob_valida']):.1%}"},
        ])
        st.dataframe(df, hide_index=True, width="stretch")
        st.info("Tramos bloqueados: " + (", ".join(f"{a}–{b}" for a, b in bloqueados) or "ninguno"))
else:
    fig, ax = plt.subplots(figsize=(5, 4)); dibujar(ax, "Red vial (rojo = bloqueado)", None); st.pyplot(fig)
    st.write("Ajusta la intensidad de El Niño y pulsa **Calcular rutas**.")
