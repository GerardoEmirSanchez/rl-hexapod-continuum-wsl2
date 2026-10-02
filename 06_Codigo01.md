### Código 1: Motor Estocástico y Deriva Anisotrópica (40 min)

#### 1. Contexto Físico y Mecatrónico del Problema
En este experimento modelaremos el comportamiento físico real de un robot caminante (hexápodo) operando sobre una mesa de trabajo plana discretizada como una cuadrícula de $5 \times 5$:
* **Coordenadas:** Las filas r van de 0 (Norte/arriba) a 4 (Sur/abajo); las columnas c van de 0 (Oeste/izquierda) a 4 (Este/derecha).
* **Posición Inicial:** El robot parte centrado en la posición (2, 0) (fila 2, columna 0).
* **Objetivo Mecánico (Gaveta):** El acople de herramientas se ubica en (2, 4) (fila 2, columna 4).
* **Controlador en Lazo Abierto (*Open-Loop*):** El controlador emite de forma ciega y repetitiva el comando nominal de avanzar siempre hacia la derecha (**Acción a = 3, desplazamiento (0, 1)**).

```text
       Col 0      Col 1      Col 2      Col 3      Col 4
Fila 0 [ PARED ]  [      ]   [      ]   [      ]  [  PELIGRO  ]
Fila 1 [       ]  [      ]   [      ]   [      ]  [           ]
Fila 2 [ INICIO] ──► ──► ──► ──► ──► ──► ──► ──►  [  GAVETA   ]  <-- Trayectoria nominal deseada
Fila 3 [       ]  [      ]   [      ]   [      ]  [           ]
Fila 4 [ PARED ]  [      ]   [      ]   [      ]  [  PELIGRO  ]
```

#### 2. La Perturbación Física (Deriva por Pérdida de Tracción)
En la realidad física, los elastómeros de las patas sufren deslizamiento sobre la mesa:
1. Con probabilidad $1 - p_{\text{deriva}}$, el robot tracciona bien y avanza en la dirección deseada (hacia el Este).
2. Con probabilidad $p_{\text{deriva}}$, el robot derrapa lateralmente y **no avanza al frente**: se desvía perpendicularmente con un $50\%$ de probabilidad hacia el Norte ((-1, 0) o un $50\%$ hacia el Sur ((1, 0)).
3. **Condición de Colisión:** Los rieles de la mesa están en la Fila 0 y Fila 4. Si la acumulación de resbalones desvía al robot a cualquiera de esas dos filas **antes de que cruce la Columna 4**, el chasis colisiona y el episodio se da por perdido.

---

#### 3. Requisitos y Reto para el Alumno
No deben utilizar librerías externas de simulación. Deberán completar de forma manual los bloques de lógica dentro de la plantilla:

1. **Ejercicio 1 (Bifurcación Estocástica):** Utilizar `random.random()` para bifurcar la cinemática: si el número generado es menor a $(1 - p_{\text{deriva}})$, aplicar el movimiento nominal; en caso contrario, elegir aleatoriamente con `random.choice()` entre el desvío vertical Norte o Sur.
2. **Ejercicio 2 (Confinamiento Cinemático):** Sumar los deltas $(dr, dc)$ a la posición actual y confinar las coordenadas al rango válido $[0, 4]$ para evitar salidas de la malla física.
3. **Ejercicio 3 (Detección de Eventos y Paro):** Evaluar en cada paso si `pos_actual` coincide con la meta $(2, 4)$ (éxito) o si ha tocado la Fila 0 o Fila 4 con columna menor a 4 (colisión), rompiendo el bucle del episodio con `break`.
4. **TODO 4 (Sustitución Manual y Registro Experimental):** Una vez implementado el código, **no automaticen un barrido con ciclos**. Deberán modificar manualmente la variable `p_evaluada` una por una para los valores $0.05$, $0.35$ y $0.70$, ejecutar la celda en cada corrida y transcribir los porcentajes calculados a la tabla de entrega.

---

#### 4. Tabla de Registro Experimental

| Condición de Fricción | Probabilidad (`p_evaluada`) | Tasa de Colisión (%) | Tasa de Éxitos (%) |
| :--- | :---: | :---: | :---: |
| **Piso Seco (Nominal)** | `0.05` | ________ % | ________ % |
| **Piso Húmedo (Taller)** | `0.35` | ________ % | ________ % |
| **Piso Aceitado (Falla)** | `0.70` | ________ % | ________ % |

---

```python
# ============================================================================
# CÓDIGO 1 — FÍSICA ESTOCÁSTICA DEL HEXÁPODO
# ============================================================================
import numpy as np
import random
# Fijar semilla para reproducibilidad de pruebas
random.seed(42)
def transicion_hexapodo(pos, accion_nominal, p_deriva):
    """
    Parámetros:
      - pos: tupla (fila, columna) indicando la posición actual del robot.
      - accion_nominal: int [0, 3] que indica la dirección pretendida por el controlador.
      - p_deriva: float en [0.0, 1.0], probabilidad de resbalón ortogonal.
    Retorna:
      - (r_nuevo, c_nuevo): tupla con la posición resultante tras aplicar la física.
    """
    # Mapeo cinemático: 0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # ------------------------------------------------------------------------
    # Ejercicio 1: BIFURCACIÓN ESTOCÁSTICA DE TRACCIÓN
    # ------------------------------------------------------------------------
    # 1. Genera un número pseudoaleatorio uniforme en el intervalo [0.0, 1.0).
    # 2. Si el valor es MENOR a (1.0 - p_deriva):
    #       dr, dc toma el desplazamiento de la acción nominal.
    #    En caso contrario (hubo resbalón lateral):
    #       dr, dc debe ser aleatoriamente (-1, 0) [Norte] o (1, 0) [Sur] con igual probabilidad.
    dr, dc = (0, 0)  # <<< BORRA ESTA LÍNEA E IMPLEMENTA Ejercicio 1 >>>

    # ------------------------------------------------------------------------
    # Ejercicio 2: CONFINAMIENTO FÍSICO Y SATURACIÓN EN BORDES (MALLA 5x5)
    # ------------------------------------------------------------------------
    # Calcula la nueva fila y columna sumando (dr, dc) a la posición actual.
    # Asegura que las coordenadas resultantes no salgan del rango válido [0, 4]
    # (utiliza funciones max/min para evitar índices fuera de rango).
    r_nuevo = 0  # <<< BORRA ESTA LÍNEA E IMPLEMENTA Ejercicio 2 >>>
    c_nuevo = 0  # <<< BORRA ESTA LÍNEA E IMPLEMENTA Ejercicio 2 >>>
    return (r_nuevo, c_nuevo)  

def evaluar_desempeno_lazo_abierto(p_deriva, n_episodios=100, max_pasos=10):
    """
    Simula n_episodios donde el robot ejecuta exclusivamente accion_nominal = 3 (Este).
    Retorna el porcentaje de episodios que terminaron en colisión lateral.
    """
    colisiones = 0
    exitos = 0
    for ep in range(n_episodios):
        pos_actual = (2, 0)  # Posición inicial fija en el centro-oeste
        for paso in range(max_pasos):
            pos_actual = transicion_hexapodo(pos_actual, accion_nominal=3, p_deriva=p_deriva)
            # ----------------------------------------------------------------
            # Ejercicio 3: CONDICIÓN DE PARO Y DETECCIÓN DE EVENTOS
            # ----------------------------------------------------------------
            # Revisa las coordenadas actuales de pos_actual:
            # A) Si el robot tocó la meta en (2, 4):
            #       Incrementa exitos en 1 y rompe el ciclo (break).
            # B) Si el robot tocó la Fila 0 o la Fila 4 ANTES de llegar a Columna 4:
            #       Incrementa colisiones en 1 y rompe el ciclo (break).
            # <<< IMPLEMENTA AQUÍ LA CONDICIÓN DE PARO (Ejercicio 3) >>>
            pass
    tasa_colision = (colisiones / n_episodios) * 100.0
    tasa_exito = (exitos / n_episodios) * 100.0
    return tasa_colision, tasa_exito
# ----------------------------------------------------------------------------
# TODO 4: EVALUACIÓN EXPERIMENTAL Y BARRIDO DE FRICCIÓN
# ----------------------------------------------------------------------------
# Modifica la variable p_evaluada para responder la tabla de entrega:
p_evaluada = 0.35
t_colision, t_exito = evaluar_desempeno_lazo_abierto(p_deriva=p_evaluada, n_episodios=100)
print(f"Resultados para p_deriva = {p_evaluada:.2f}:")
print(f" -> Tasa de Colisión Lateral: {t_colision:.1f}%")
print(f" -> Tasa de Éxitos a Meta:    {t_exito:.1f}%")
```














# CODIGO RESUELTO



```
# ============================================================================
# CÓDIGO 1 — FÍSICA ESTOCÁSTICA DEL HEXÁPODO 
# ============================================================================
import numpy as np
import random

# ----------------------------------------------------------------------------
# 1. PARÁMETRO EXPERIMENTAL (MODIFICAR AQUÍ EN CADA CORRIDA)
# ----------------------------------------------------------------------------
# Cambia esta variable manualmente para registrar cada fila de la TABLA 01:
#   Corrida 1 (Piso Seco):     p_evaluada = 0.05
#   Corrida 2 (Piso Húmedo):   p_evaluada = 0.35
#   Corrida 3 (Piso Aceitado): p_evaluada = 0.70


# ----------------------------------------------------------------------------
# 2. MOTOR DE FÍSICA Y TRANSICIÓN
# ----------------------------------------------------------------------------
def transicion_hexapodo(pos, accion_nominal, p_deriva):
    """
    pos: tupla (fila, columna) [0 a 4]
    accion_nominal: 3 (Avanzar al Este)
    p_deriva: Probabilidad de resbalón ortogonal
    """
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 0:Up, 1:Down, 2:Left, 3:Right
    
    # Bifurcación estocástica de tracción
    if random.random() < (1.0 - p_deriva):
        dr, dc = movimientos[accion_nominal]
    else:
        # Resbalón transversal: Norte (-1, 0) o Sur (1, 0) con 50% c/u
        dr, dc = random.choice([(-1, 0), (1, 0)])

    # Confinamiento físico en la cuadrícula [0, 4]
    r_nuevo = max(0, min(4, pos[0] + dr))
    c_nuevo = max(0, min(4, pos[1] + dc))

    return (r_nuevo, c_nuevo)


# ----------------------------------------------------------------------------
# 3. EVALUADOR EXPERIMENTAL (100 EPISODIOS EN LAZO ABIERTO)
# ----------------------------------------------------------------------------
def evaluar_desempeno_lazo_abierto(p_deriva, n_episodios=100, max_pasos=10, semilla=42):
    # La semilla se reinicia en cada llamada para garantizar resultados idénticos
    random.seed(semilla)
    
    colisiones = 0
    exitos = 0
    
    for ep in range(n_episodios):
        pos_actual = (2, 0)  # Inicia en el centro-oeste
        
        for paso in range(max_pasos):
            pos_actual = transicion_hexapodo(pos_actual, accion_nominal=3, p_deriva=p_deriva)
            
            # A) Acople exitoso en la gaveta
            if pos_actual == (2, 4):
                exitos += 1
                break
                
            # B) Colisión lateral destructiva antes de llegar a la columna 4
            elif (pos_actual[0] == 0 or pos_actual[0] == 4) and pos_actual[1] < 4:
                colisiones += 1
                break

    tasa_colision = (colisiones / n_episodios) * 100.0
    tasa_exito = (exitos / n_episodios) * 100.0
    return tasa_colision, tasa_exito


# ----------------------------------------------------------------------------
# TODO 4: EVALUACIÓN EXPERIMENTAL Y BARRIDO DE FRICCIÓN
# ----------------------------------------------------------------------------
# Modifica la variable p_evaluada para responder la tabla de entrega:
p_evaluada = 0.0

t_colision, t_exito = evaluar_desempeno_lazo_abierto(p_deriva=p_evaluada, n_episodios=100)
print(f"Resultados para p_deriva = {p_evaluada:.2f}:")
print(f" -> Tasa de Colisión Lateral: {t_colision:.1f}%")
print(f" -> Tasa de Éxitos a Meta:    {t_exito:.1f}%")

```




