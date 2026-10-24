# Datos del escenario

| Archivo | Contenido | Origen |
|---|---|---|
| `nodos.csv` | 2 centros de acopio y 23 comunidades: coordenadas, código INEI, distrito, margen del río, familias afectadas, prioridad médica, estado del acceso terrestre y motivo de inclusión | Coordenadas: INEI (comunidades) y Wikipedia (aeropuerto, UDEP). Familias, prioridad y acceso: escenario |
| `puentes.csv` | 5 puentes sobre el río Piura y su estado en el escenario | Existencia: noticias y documentos oficiales. Posición: aproximada sobre el cauce. Estado: escenario |
| `rio_piura_aprox.csv` | Polilínea del cauce, en sentido aguas abajo | Digitalizada a mano a partir de las márgenes reportadas (CENEPRED, RPP) y de los centros poblados INEI. Aproximada |
| `puntos_auxiliares.csv` | Cruces y puntos intermedios de la red vial | Aproximados; la trocha de desvío es un supuesto |
| `corredores.csv` | Tramos de la red vial (desde, hasta, tipo de vía) | Aproximados entre puntos reales |
| `limites_piura.geojson` | Límites distritales del departamento de Piura (solo para el mapa) | Derivado de [juaneladio/peru-geojson](https://github.com/juaneladio/peru-geojson), licencia MPL-2.0 |

## Columnas de `nodos.csv`

- `acceso_terrestre`: `operativo` o `cortado`. Un acceso cortado bloquea todos los tramos que llegan a esa comunidad (vía anegada o cortada por quebrada activa).
- `prioridad_medica`: 1 si la comunidad necesita medicinas dentro de la ventana (2,5 h si va por aire, 3 h por tierra; se cambia en `qrutas/config.py`).
- `familias`: valor de escenario. La demanda es 10 kg por familia (víveres, agua, higiene y botiquín para la primera respuesta).
- `margen`: MI (izquierda) o MD (derecha) del río Piura, según la orientación aguas abajo. NA para Tambo Grande.

## Para reproducir

`herramientas/preparar_datos.py` vuelve a generar `nodos.csv` y `limites_piura.geojson` desde los repositorios abiertos (las instrucciones están al inicio del archivo). Las coordenadas de los centros poblados provienen del padrón del INEI publicado en [jmcastagnetto/ubigeo-peru-aumentado](https://github.com/jmcastagnetto/ubigeo-peru-aumentado) (licencia MIT).

## Cómo cambiar el escenario

- Otro puente caído: cambia su `estado` a `colapsado` o `cerrado` en `puentes.csv`.
- Otra comunidad aislada: pon `cortado` en `acceso_terrestre`.
- Una comunidad nueva: agrégala a `herramientas/preparar_datos.py` con su nombre exacto en el padrón INEI y conéctala a la red en `corredores.csv`.
- Más o menos vehículos: `flota` en `qrutas/config.py`.
