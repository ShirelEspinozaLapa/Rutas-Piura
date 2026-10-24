"""Genera un mapa web interactivo en HTML utilizando Leaflet.js y OpenStreetMap."""
from __future__ import annotations

import csv
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
DATOS = os.path.join(RAIZ, "datos")
RESULTADOS = os.path.join(RAIZ, "resultados")


def generar_mapa_html(ruta_salida: str = os.path.join(RESULTADOS, "visualizador_interactivo.html")):
    # 1. Cargar nodos
    nodos = []
    with open(os.path.join(DATOS, "nodos.csv"), "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nodos.append({
                "id": row["id"],
                "nombre": row["nombre"],
                "tipo": row["tipo"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "familias": int(row["familias"]) if row["familias"] else 0,
                "prioridad": int(row["prioridad_medica"]) if row["prioridad_medica"] else 0,
                "acceso": row["acceso_terrestre"]
            })

    # 2. Cargar puentes
    puentes = []
    with open(os.path.join(DATOS, "puentes.csv"), "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            puentes.append({
                "nombre": row["nombre"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "estado": row["estado"]
            })

    # 3. Cargar rio
    rio = []
    with open(os.path.join(DATOS, "rio_piura_aprox.csv"), "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rio.append([float(row["lat"]), float(row["lon"])])

    # 4. Generar HTML con Leaflet
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q-Rutas Piura - Visor Cartográfico Interactivo</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}
        .panel-info {{
            position: absolute; top: 15px; right: 15px; z-index: 1000;
            background: rgba(15, 23, 42, 0.92); color: white; padding: 16px;
            border-radius: 10px; max-width: 320px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.1);
        }}
        .leyenda-item {{ display: flex; align-items: center; margin-top: 6px; font-size: 13px; }}
        .badge {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="panel-info">
        <h3 style="margin: 0 0 8px 0; color: #38bdf8;">🚁 Q-Rutas Piura</h3>
        <p style="margin: 0 0 12px 0; font-size: 12px; color: #94a3b8;">Mapa táctico de ruteo humanitario ante crecida del río Piura.</p>
        <div class="leyenda-item"><span class="badge" style="background: #2563eb;"></span> Centros de Acopio (Bases)</div>
        <div class="leyenda-item"><span class="badge" style="background: #ef4444;"></span> Comunidad con corte terrestre</div>
        <div class="leyenda-item"><span class="badge" style="background: #10b981;"></span> Comunidad con acceso operativo</div>
        <div class="leyenda-item"><span class="badge" style="background: #f59e0b;"></span> Puente restringido / cerrado</div>
        <div class="leyenda-item"><span class="badge" style="background: #dc2626;"></span> Puente Independencia colapsado</div>
        <div class="leyenda-item"><span class="badge" style="background: #0284c7; width: 18px; height: 4px; border-radius: 2px;"></span> Río Piura (Cauce aproximado)</div>
    </div>

    <script>
        const map = L.map('map').setView([-5.30, -80.65], 11);

        const osmLayer = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors | Q-Rutas Piura'
        }});

        const esriDarkLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 16,
            attribution: 'Tiles © Esri | Q-Rutas Piura'
        }}).addTo(map);

        L.control.layers({{
            "Mapa Táctico Oscuro (Esri)": esriDarkLayer,
            "OpenStreetMap Estándar": osmLayer
        }}, null, {{ position: 'topright' }}).addTo(map);

        // Río Piura
        const rioCoords = {json.dumps(rio)};
        L.polyline(rioCoords, {{ color: '#0284c7', weight: 5, opacity: 0.8, dashArray: '5, 5' }}).addTo(map)
         .bindPopup('<b>Cauce del Río Piura</b><br>Zona de desborde y barrera natural.');

        // Puentes
        const puentes = {json.dumps(puentes)};
        puentes.forEach(p => {{
            let color = '#10b981';
            if (p.estado === 'restringido') color = '#f59e0b';
            if (p.estado === 'cerrado' || p.estado === 'colapsado') color = '#dc2626';

            L.circleMarker([p.lat, p.lon], {{
                radius: 7, fillColor: color, color: '#ffffff', weight: 2, fillOpacity: 0.9
            }}).addTo(map).bindPopup(`<b>${{p.nombre}}</b><br>Estado en el escenario: <b>${{p.estado.toUpperCase()}}</b>`);
        }});

        // Nodos (Comunidades y Depósitos)
        const nodos = {json.dumps(nodos)};
        nodos.forEach(n => {{
            let color = n.acceso === 'cortado' ? '#ef4444' : '#10b981';
            let radius = 6;
            if (n.tipo === 'deposito') {{
                color = '#2563eb';
                radius = 9;
            }}

            const marker = L.circleMarker([n.lat, n.lon], {{
                radius: radius, fillColor: color, color: '#ffffff', weight: 2, fillOpacity: 0.95
            }}).addTo(map);

            let pop = `<b>${{n.nombre}}</b> (${{n.id}})<br>`;
            if (n.tipo === 'deposito') {{
                pop += `Centro de Acopio y Base de Despliegue.`;
            }} else {{
                pop += `Familias afectadas: <b>${{n.familias}}</b> (${{n.familias * 10}} kg ayuda)<br>` +
                       `Acceso terrestre: <b>${{n.acceso.toUpperCase()}}</b><br>` +
                       `Prioridad médica: ${{n.prioridad ? '🚨 URGENTE (Medicinas)' : 'Normal'}}`;
            }}
            marker.bindPopup(pop);
        }});
    </script>
</body>
</html>"""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Mapa interactivo generado exitosamente en: {ruta_salida}")


if __name__ == "__main__":
    generar_mapa_html()
