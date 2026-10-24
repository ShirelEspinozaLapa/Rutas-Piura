# Resumen de resultados · Q-Rutas Piura

Generado el 2026-10-24 13:21 · semilla 2026 · modo completo · motor cuántico `auto` · Qiskit instalado: no

## 1. Escenario

- 23 comunidades afectadas (3,070 familias, 30.7 t de ayuda en la primera respuesta) y 2 centros de acopio: Aeropuerto de Piura / base aérea, Campus UDEP (centro de acopio).
- Flota: 2 helicópteros en Aeropuerto de Piura / base aérea (4 t c/u); 2 camiones en Aeropuerto de Piura / base aérea (8 t c/u); 2 camiones en Campus UDEP (centro de acopio) (8 t c/u).
- Puentes: Puente Cáceres → **restringido**; Puente Sánchez Cerro → **cerrado**; Puente Bolognesi → **cerrado**; Puente Grau (Panamericana) → **operativo**; Puente Independencia (Catacaos - La Arena) → **colapsado**.
- Triage modal: 11 comunidades solo por aire (1 de ellas tiene ruta terrestre, pero el camión no llega a tiempo con las medicinas: Sinchao (El Tallán)); 12 por camión.

## 2. Fase clásica: reducción de dimensionalidad

Formular el VRP completo como QUBO (tour gigante con copias del depósito) pide **784 qubits** (espacio de 2^784 estados). Tras el triage y el k-medoides capacitado quedan **7 subproblemas de 4 a 16 qubits**.
 El refinamiento por búsqueda local bajó el costo estimado de los clusters de 40.69 a 36.13 (6 movimientos).


| Cluster | Vehículo | Sale de | Paradas | Qubits | Carga (kg) | Salida prevista | Comunidades |
|---|---|---|---|---|---|---|---|
| Aéreo-1 | helicóptero | Aeropuerto de Piura | 3 | 9 | 2400 | 2.58 h | Cucungará, Chato Grande, Pedregal Chico |
| Aéreo-2 | helicóptero | Aeropuerto de Piura | 4 | 16 | 3800 | 0.00 h | Pedregal Grande, Santa Rosa (Cura Mori), Chato Chico, Sinchao (El Tallán) |
| Aéreo-3 | helicóptero | Aeropuerto de Piura | 4 | 16 | 3300 | 0.00 h | Viduque, Narihualá, Locuto (Tambo Grande), San Martín de Malingas |
| Terr-A1 | camión | Aeropuerto de Piura | 3 | 9 | 5800 | 0.00 h | Pueblo Nuevo (Cura Mori), San Pablo, Catacaos (ciudad) |
| Terr-A2 | camión | Aeropuerto de Piura | 2 | 4 | 4000 | 0.00 h | Simbilá, La Legua - San Jacinto |
| Terr-U1 | camión | Campus UDEP | 3 | 9 | 7800 | 0.00 h | La Unión, La Arena, Las Malvinas |
| Terr-U2 | camión | Campus UDEP | 4 | 16 | 3600 | 0.00 h | Paredones, Monte Castillo, Casagrande, Chatito |

## 3. Mismo modelo reducido: QAOA frente a métodos clásicos

Todos resuelven exactamente el mismo QUBO de cada cluster. *P(opt)* es la probabilidad de obtener la ruta óptima en un solo disparo (QAOA) o en una sola lectura (SA). *TTS* es cuántos disparos o lecturas hacen falta para verla al menos una vez con 99 % de confianza.

| Subproblema | Qubits | QAOA p | QAOA P(fact) | QAOA P(opt) | TTS QAOA | QAOA-XY P(opt) | TTS XY | SA P(opt) | TTS SA | Brecha final (QAOA / XY / SA / GA) |
|---|---|---|---|---|---|---|---|---|---|---|
| Aéreo-1 | 9 | 3 | 0.410 | 0.0658 | 68 | 0.382 | 10 | 0.75 | 4 | 0.0 % / 0.0 % / 0.0 % / 0.0 % |
| Aéreo-2 | 16 | 3 | 0.101 | 0.0035 | 1300 | 0.166 | 26 | 0.26 | 14 | 0.0 % / 0.0 % / 0.0 % / 0.0 % |
| Aéreo-3 | 16 | 3 | 0.070 | 0.0019 | 2445 | 0.192 | 22 | 0.32 | 10 | 0.0 % / 0.0 % / 0.0 % / 0.0 % |
| Terr-A1 | 9 | 3 | 0.141 | 0.0274 | 166 | 0.241 | 17 | 0.85 | 3 | 0.0 % / 0.0 % / 0.0 % / 0.0 % |
| Terr-A2 | 4 | 3 | 0.342 | 0.1714 | 25 | 0.623 | 5 | 0.99 | 2 | 0.0 % / 0.0 % / 0.0 % / 0.0 % |
| Terr-U1 | 9 | 2 | 0.215 | 0.0454 | 100 | 0.185 | 23 | 0.90 | 3 | 0.0 % / 0.0 % / 0.0 % / 0.0 % |
| Terr-U2 | 16 | 2 | 0.079 | 0.0022 | 2126 | 0.153 | 28 | 0.24 | 19 | 0.0 % / 0.0 % / 0.0 % / 0.0 % |

Costo computacional y estimación en hardware real:

| Subproblema | Evaluaciones QAOA | QAOA simulado (CPU) | QAOA en QPU (estimado) | CX lógicas QAOA / XY | Prob. disparo sin error QAOA / XY | SA (CPU) | GA (CPU) |
|---|---|---|---|---|---|---|---|
| Aéreo-1 | 540 | 241 ms | 28.4 min | 180 / 936 | 0.169 / 1.1e-04 | 40 ms | 17 ms |
| Aéreo-2 | 717 | 11.1 s | 38.9 min | 504 / 2808 | 0.008 / 1.5e-12 | 71 ms | 18 ms |
| Aéreo-3 | 620 | 9.0 s | 33.6 min | 504 / 2808 | 0.008 / 1.5e-12 | 70 ms | 18 ms |
| Terr-A1 | 709 | 380 ms | 37.3 min | 180 / 936 | 0.169 / 1.1e-04 | 40 ms | 27 ms |
| Terr-A2 | 283 | 19 ms | 14.5 min | 36 / 156 | 0.675 / 2.1e-01 | 18 ms | 17 ms |
| Terr-U1 | 662 | 341 ms | 34.4 min | 120 / 936 | 0.292 / 1.1e-04 | 39 ms | 17 ms |
| Terr-U2 | 779 | 11.5 s | 41.4 min | 336 / 2808 | 0.037 / 1.5e-12 | 76 ms | 19 ms |

La estimación de QPU usa tiempos y tasas de error de orden de magnitud (ver `qrutas/config.py` y `qrutas/hardware.py`); con Qiskit y `qiskit-ibm-runtime` instalados se reemplaza por la transpilación a un backend falso de IBM.

## 4. Solución completa

| Método | J | Entrega acumulada (h) | Combustible (L) | Riesgo | Atraso medicinas (h) | Ventanas incumplidas | Última entrega (h) | Rutas | Cómputo |
|---|---|---|---|---|---|---|---|---|---|
| Clásico puro: SA sin clustering | **35.05** | 50.3 | 932 | 0.303 | 0.00 | 0 | 4.79 | 7 | 777 ms |
| Clásico puro: GA sin clustering | **35.24** | 50.4 | 979 | 0.303 | 0.00 | 0 | 4.78 | 7 | 958 ms |
| Clustering + exacto (fuerza bruta) | **36.02** | 52.2 | 936 | 0.303 | 0.00 | 0 | 4.79 | 7 | 227 ms |
| Clustering + SA sobre QUBO | **36.02** | 52.2 | 936 | 0.303 | 0.00 | 0 | 4.79 | 7 | 581 ms |
| Clustering + algoritmo genético | **36.02** | 52.2 | 936 | 0.303 | 0.00 | 0 | 4.79 | 7 | 360 ms |
| Híbrido: clustering + QAOA | **36.02** | 52.2 | 936 | 0.303 | 0.00 | 0 | 4.79 | 7 | 32.9 s |
| Híbrido: clustering + QAOA-XY | **36.02** | 52.2 | 936 | 0.303 | 0.00 | 0 | 4.79 | 7 | 4.7 s |

**Híbrido: clustering + QAOA**

- H1 sale 0.00 h de Aeropuerto de Piura / base aérea: Pedregal Grande (0.57 h) → Santa Rosa (Cura Mori) (0.91 h) → Chato Chico (1.27 h) → Sinchao (El Tallán) (1.63 h); vuelve 2.08 h · 3800 kg
- H1 sale 2.58 h de Aeropuerto de Piura / base aérea: Pedregal Chico (3.15 h) → Cucungará (3.50 h) → Chato Grande (3.85 h); vuelve 4.27 h · 2400 kg
- H2 sale 0.00 h de Aeropuerto de Piura / base aérea: Viduque (0.54 h) → Narihualá (0.90 h) → Locuto (Tambo Grande) (1.50 h) → San Martín de Malingas (1.87 h); vuelve 2.44 h · 3300 kg
- T1 sale 0.00 h de Aeropuerto de Piura / base aérea: San Pablo (1.09 h) → Catacaos (ciudad) (1.92 h) → Pueblo Nuevo (Cura Mori) (3.18 h); vuelve 4.45 h · 5800 kg
- T2 sale 0.00 h de Aeropuerto de Piura / base aérea: Simbilá (1.02 h) → La Legua - San Jacinto (1.64 h); vuelve 2.53 h · 4000 kg
- T3 sale 0.00 h de Campus UDEP: La Arena (2.71 h) → Las Malvinas (3.31 h) → La Unión (3.93 h); vuelve 6.61 h · 7800 kg
- T4 sale 0.00 h de Campus UDEP: Monte Castillo (2.30 h) → Paredones (2.89 h) → Casagrande (3.75 h) → Chatito (4.79 h); vuelve 7.49 h · 3600 kg

**Clásico puro: SA sin clustering**

- H1 sale 0.00 h de Aeropuerto de Piura / base aérea: Viduque (0.54 h) → Narihualá (0.90 h) → Pedregal Chico (1.23 h) → Locuto (Tambo Grande) (1.84 h) → San Martín de Malingas (2.22 h); vuelve 2.78 h · 3800 kg
- H2 sale 0.00 h de Aeropuerto de Piura / base aérea: Pedregal Grande (0.57 h) → Santa Rosa (Cura Mori) (0.91 h) → Chato Chico (1.27 h) → Sinchao (El Tallán) (1.63 h); vuelve 2.08 h · 3800 kg
- H2 sale 2.58 h de Aeropuerto de Piura / base aérea: Cucungará (3.15 h) → Chato Grande (3.50 h); vuelve 3.92 h · 1900 kg
- T1 sale 0.00 h de Aeropuerto de Piura / base aérea: San Pablo (1.09 h) → Catacaos (ciudad) (1.92 h) → Pueblo Nuevo (Cura Mori) (3.18 h); vuelve 4.45 h · 5800 kg
- T2 sale 0.00 h de Aeropuerto de Piura / base aérea: Simbilá (1.02 h) → La Legua - San Jacinto (1.64 h); vuelve 2.53 h · 4000 kg
- T3 sale 0.00 h de Campus UDEP: La Arena (2.71 h) → Las Malvinas (3.31 h) → La Unión (3.93 h); vuelve 6.61 h · 7800 kg
- T4 sale 0.00 h de Campus UDEP: Monte Castillo (2.30 h) → Paredones (2.89 h) → Casagrande (3.75 h) → Chatito (4.79 h); vuelve 7.49 h · 3600 kg

## 5. Lectura de los números

- El QAOA estándar llegó a la ruta óptima en 7 de 7 subproblemas, pero gracias a la post-selección de miles de disparos: su P(opt) por disparo va de 0.0019 a 0.1714. Con el mezclador que preserva permutaciones (QAOA-XY) sube a 0.15-0.62, porque nunca gasta amplitud en bitstrings inválidos.
- Simulated Annealing sobre el mismo QUBO acierta el óptimo en 0.24-0.99 de sus lecturas y tarda milisegundos. A este tamaño (≤ 16 qubits) no hay ventaja cuántica: el valor del prototipo es el flujo completo y que escala en la dirección correcta cuando haya más qubits.
- En hardware ruidoso actual, la probabilidad de que un disparo del QAOA estándar corra sin un solo error estaría entre 0.01 y 0.67. El QAOA-XY es mucho mejor en simulación pero pide hasta 2808 CNOT por circuito, así que sin corrección de errores su salida sería casi puro ruido: la profundidad del circuito es el cuello de botella real, más que el número de qubits.
- La solución híbrida tiene J = 36.02; el mejor clásico puro sin clustering, J = 35.05 (+2.8 % a favor del clásico). La diferencia no viene del QAOA (clustering + fuerza bruta da exactamente J = 36.02) sino de la descomposición: el clásico puro arma viajes de hasta 5 paradas, que en el híbrido no caben (5^2 = 25 qubits).
- Ninguna solución deja comunidades sin atender. Ventanas de medicinas incumplidas: híbrido 0, clásico puro 0.

## 6. Figuras

- `1_mapa_escenario.png`
- `2_mapa_clusters.png`
- `3_mapa_rutas_clasico_vs_hibrido.png`
- `4_convergencia.png`
- `5_qaoa_distribucion.png`
- `6_qaoa_paisaje_p1.png`
- `7_metricas_comparacion.png`
