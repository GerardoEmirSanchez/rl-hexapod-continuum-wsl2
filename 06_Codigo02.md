### Código 2: Motor Completo de Q-Learning Tabular y Extracción de Ruta (45 min)

#### 1. Contexto Físico y Mecatrónico del Problema
En este experimento abandonamos la simulación artesanal y conectamos el controlador a la plataforma industrial **Gymnasium**, abordando el entorno de referencia para navegación con riesgo crítico: `CliffWalking-v1`.

* **Topología del Entorno:** Cuadrícula de $4 \times 12$ casillas ($\vert{}\mathcal{S}\vert{} = 48\text{ estados discretizados}$, indexados linealmente de $0$ a $47$):
  * **Filas:** $0$ (superior), $1$, $2$ (cornisa de riesgo) y $3$ (planta baja).
  * **Columnas:** $0$ a $11$.
  * Conversión formal de coordenadas: $\text{Fila} = s // 12, \quad \text{Columna} = s \% 12$.
* **Punto de Partida:** Estado $s_0 = 36$ (Fila 3, Columna 0; esquina inferior izquierda).
* **Objetivo Mecánico (Gaveta de Acople):** Estado terminal meta $s_{\text{meta}} = 47$ (Fila 3, Columna 11; esquina inferior derecha).
* **Zona de Destrucción / Falla Crítica (El Acantilado):** Casillas $37$ a $46$ (Fila 3, Columnas 1 a 10). Caer aquí emite una señal de penalización severa $R = -100.0\text{ pts}$ y reinicia forzosamente la posición del robot a $s = 36$.
* **Costo Energético Nominal:** Cada paso regular sobre terreno firme penaliza con $R = -1.0\text{ pt}$ (consumo de batería en servomotores).
* **Espacio de Acciones:** $\mathcal{A} = \{0: \text{Arriba}, 1: \text{Derecha}, 2: \text{Abajo}, 3: \text{Izquierda}\}$.

```text
Col:   0    1    2    3    4    5    6    7    8    9   10   11
R0: [ 00 ][ 01 ][ 02 ][ 03 ][ 04 ][ 05 ][ 06 ][ 07 ][ 08 ][ 09 ][ 10 ][ 11 ]  <-- Ruta segura (lejos del abismo)
R1: [ 12 ][ 13 ][ 14 ][ 15 ][ 16 ][ 17 ][ 18 ][ 19 ][ 20 ][ 21 ][ 22 ][ 23 ]
R2: [ 24 ][ 25 ][ 26 ][ 27 ][ 28 ][ 29 ][ 30 ][ 31 ][ 32 ][ 33 ][ 34 ][ 35 ]  <-- Cornisa (riesgo de caída lateral)
R3: [ IN ][  X    X    X    X    X    X    X    X    X    X  ][META]
    s=36  [       ZONA DE ACANTILADO: s=37 a s=46 (-100 pts)       ]  s=47
```

---

#### 2. La Formulación Matemática de Diferencia Temporal (Q-Learning)
A diferencia de los métodos analíticos basados en modelo (*Model-Based*), el robot desconoce la función de transición física del entorno. Aprende directamente de las muestras de interacción sensorial $({s}_t, {a}_t, {r}_{t+1}, {s}_{t+1})$
mediante la regla de **Diferencia Temporal (TD)**:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \cdot \delta_t$$

Donde el término de error o discrepancia cibernética $\delta_t$ (**Error TD**) se define como:

$$\delta_t = \underbrace{\left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') \right]}_{\text{TD Target (Realidad inmediata + Promesa futura)}} - \underbrace{Q(s_t, a_t)}_{\text{Creencia previa en memoria RAM}}$$

* **$\alpha \in (0, 1]$ (Tasa de Aprendizaje):** Filtro digital de memoria. Modula cuánta importancia se otorga a la nueva experiencia frente al historial acumulado.
* **$\gamma \in [0, 1)$ (Factor de Descuento Temporal):** Define el horizonte de planificación. Si $\gamma \to 0$, el robot es miope; si $\gamma \to 1$, valora altamente las recompensas lejanas.
* **Naturaleza *Off-Policy*:** La política de recopilación de datos (comportamiento) utiliza una regla de compromiso estocástico **$\epsilon$-Greedy**, mientras que la política objetivo asume selección puramente voraz mediante el operador $\max_{a'}$.

---

#### 3. Requisitos y Reto para el Alumno
Deberán completar de forma manual los bloques de lógica dentro de la plantilla sin recurrir a bibliotecas de aprendizaje de alto nivel (como Stable-Baselines3):

1. **Ejercicio 1 (Mecanismo de Exploración $\epsilon$-Greedy):** Generar una condición de bifurcación con `random.random()`. Si el escalar pseudoaleatorio es estrictamente menor a $\epsilon$, seleccionar una acción uniforme al azar mediante `env.action_space.sample()`; en caso contrario, explotar la memoria seleccionando $\arg\max_a Q(s, :)$.
2. **Ejercicio 2 (Actualización de Bellman TD y Costo de Error):** Deducir y calcular en código el `td_target`, la señal de error `td_error` y la inyección proporcional a la tabla `Q[s, a]`. Registrar el valor absoluto $\vert{}\delta_t\vert{}$ para evaluar la convergencia numérica.
3. **Ejercicio 3 (Inferencia Determinista Voraz):** En la función `evaluar_politica`, ejecutar la política óptima aprendida fijando $\epsilon = 0.0$ (explotación pura mediante `np.argmax(Q[s, :])`), registrando la secuencia cinemática de casillas visitadas y el retorno total acumulado $G_0$.
4. **TODO 4 (Sustitución Manual de Hiperparámetros):** Una vez verificado el algoritmo, **no programen un bucle de barrido automático**. Deberán modificar manualmente las variables de configuración en la **Sección 1** del script para cada una de las 4 corridas de la tabla, ejecutar la celda completa y transcribir directamente los valores calculados de telemetría a la hoja de entrega.

---

#### 4. Tabla de Registro Experimental

Modifica manualmente en la **Sección 1** los valores de `alpha`, `gamma`, `epsilon` y `episodios` para cada corrida. Ejecuta el script y completa la telemetría correspondiente:

| Corrida Evaluada | Tasa ($\alpha$) | Descuento ($\gamma$) | Exploración ($\epsilon$) | Episodios ($N$) | Retorno Final ($G_0$) | ¿Llega a Meta? | Pasos Usados | Error TD Final ($\vert{}\delta_t\vert{}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1: Base Nominal** | `0.10` | `0.95` | `0.20` | `400` | ________ pts | ________ | ________ | ________ |
| **2: Agente Miope** | `0.10` | `0.15` | `0.20` | `400` | ________ pts | ________ | ________ | ________ |
| **3: Cero Azar** | `0.10` | `0.95` | `0.00` | `400` | ________ pts | ________ | ________ | ________ |
| **4: Infraentrenado** | `0.10` | `0.95` | `0.20` | `30` | ________ pts | ________ | ________ | ________ |

---

```python
# ============================================================================
# CÓDIGO 2 — MOTOR DE APRENDIZAJE Q-LEARNING TABULAR (PLANTILLA DEL ALUMNO)
# ============================================================================
import gymnasium as gym
import numpy as np
import random

# ----------------------------------------------------------------------------
# SECCIÓN 1: HIPERPARÁMETROS EXPERIMENTALES (MODIFICAR AQUÍ MANUALMENTE)
# ----------------------------------------------------------------------------
# Cambia estas 4 variables una a una para responder las filas de la tabla:
#   Corrida 1 (Base):           alpha = 0.10, gamma = 0.95, epsilon = 0.20, episodios = 400
#   Corrida 2 (Miope):          alpha = 0.10, gamma = 0.15, epsilon = 0.20, episodios = 400
#   Corrida 3 (Sin Explorar):   alpha = 0.10, gamma = 0.95, epsilon = 0.00, episodios = 400
#   Corrida 4 (Infraentrenado): alpha = 0.10, gamma = 0.95, epsilon = 0.20, episodios = 30

alpha_exp = 0.10
gamma_exp = 0.95
epsilon_exp = 0.20
episodios_exp = 400


# ----------------------------------------------------------------------------
# SECCIÓN 2: ALGORITMO DE ENTRENAMIENTO TEMPORAL DIFFERENCE (Q-LEARNING)
# ----------------------------------------------------------------------------
def entrenar_q_learning(env, alpha, gamma, epsilon, episodios, semilla=42):
    random.seed(semilla)
    np.random.seed(semilla)

    n_S = env.observation_space.n
    n_A = env.action_space.n
    Q = np.zeros((n_S, n_A))  # Memoria tabular inicializada en ceros
    historial_td = []

    for ep in range(episodios):
        s, _ = env.reset(seed=semilla + ep)
        errores_ep = []
        
        while True:
            # ----------------------------------------------------------------
            # Ejercicio 1: SELECCIÓN DE ACCIÓN EPSILON-GREEDY
            # ----------------------------------------------------------------
            # 1. Genera un número flotante pseudoaleatorio uniforme en [0.0, 1.0).
            # 2. Si es MENOR que epsilon:
            #       a = env.action_space.sample() (Exploración ciega)
            #    En caso contrario:
            #       a = np.argmax(Q[s, :])         (Explotación de memoria)
            
            a = 0  # <<< BORRA ESTA LÍNEA E IMPLEMENTA Ejercicio 1 >>>

            # Interacción física con la API Gymnasium
            s_sig, r, term, trunc, _ = env.step(a)

            # ----------------------------------------------------------------
            # Ejercicio 2: CÁLCULO DEL ERROR TD Y ACTUALIZACIÓN BELLMAN
            # ----------------------------------------------------------------
            # 1. Calcula td_target = r + gamma * max_a' Q(s_sig, a')
            # 2. Calcula td_error  = td_target - Q[s, a]
            # 3. Actualiza Q[s, a] inyectando alpha * td_error
            
            td_error = 0.0  # <<< BORRA ESTA LÍNEA E IMPLEMENTA Ejercicio 2 >>>

            errores_ep.append(abs(td_error))
            s = s_sig
            
            if term or trunc:
                break
                
        historial_td.append(np.mean(errores_ep))
        
    return Q, historial_td


# ----------------------------------------------------------------------------
# SECCIÓN 3: EXTRACTOR Y EVALUADOR DE POLÍTICA DETERMINISTA (INFERENCIA)
# ----------------------------------------------------------------------------
def evaluar_politica(env, Q, max_pasos=40, semilla=42):
    s, _ = env.reset(seed=semilla)
    ruta = [s]
    g0 = 0.0

    for _ in range(max_pasos):
        # --------------------------------------------------------------------
        # Ejercicio 3: ACCIÓN DETERMINISTA VORAZ PURA (EPSILON = 0)
        # --------------------------------------------------------------------
        # Selecciona estrictamente la mejor acción conocida desde la tabla Q:
        
        a = 0  # <<< BORRA ESTA LÍNEA E IMPLEMENTA Ejercicio 3 >>>

        s, r, term, trunc, _ = env.step(a)
        ruta.append(s)
        g0 += r
        
        if term or trunc:
            break

    return ruta, g0


# ----------------------------------------------------------------------------
# SECCIÓN 4: EJECUCIÓN EXPERIMENTAL Y TELEMETRÍA (TODO 4)
# ----------------------------------------------------------------------------
env_cliff = gym.make('CliffWalking-v1')

# Ejecutar el entrenamiento con los hiperparámetros fijados en la Sección 1
Q_opt, logs_td = entrenar_q_learning(
    env=env_cliff,
    alpha=alpha_exp,
    gamma=gamma_exp,
    epsilon=epsilon_exp,
    episodios=episodios_exp,
    semilla=42
)

# Evaluar la política aprendida en modo explotación pura
ruta_eval, retorno_eval = evaluar_politica(env_cliff, Q_opt, semilla=42)

pasos_usados = len(ruta_eval) - 1
alcanzo_meta = (ruta_eval[-1] == 47)
error_td_final = np.mean(logs_td[-10:]) if len(logs_td) >= 10 else np.mean(logs_td)

print("=" * 80)
print(f"{'TELEMETRÍA PARA COMPLETAR LA TABLA (CÓDIGO 2)':^80}")
print("=" * 80)
print(f" -> Tasa de Aprendizaje (alpha):       {alpha_exp:.2f}")
print(f" -> Factor de Descuento (gamma):        {gamma_exp:.2f}")
print(f" -> Tasa de Exploración (epsilon):     {epsilon_exp:.2f}")
print(f" -> Volumen de Episodios (N):          {episodios_exp}")
print(f" -> Retorno Final Acumulado (G0):      {retorno_eval:+06.1f} pts")
print(f" -> ¿Alcanzó la Meta Física (s=47)?:   {'SÍ' if alcanzo_meta else 'NO'}")
print(f" -> Pasos Empleados por el Robot:      {pasos_usados} pasos")
print(f" -> Error TD Final Promedio (|delta|): {error_td_final:.4f}")
print(f" -> Secuencia de Estados Visitados:    {ruta_eval}")
print("=" * 80)
```

---

### CODIGO RESUELTO

```python
# ============================================================================
# CÓDIGO 2 — MOTOR DE APRENDIZAJE Q-LEARNING TABULAR (SOLUCIÓN COMPLETA)
# ============================================================================
import gymnasium as gym
import numpy as np
import random

# ----------------------------------------------------------------------------
# SECCIÓN 1: HIPERPARÁMETROS EXPERIMENTALES (MODIFICAR AQUÍ MANUALMENTE)
# ----------------------------------------------------------------------------
# Cambia estas 4 variables una a una para registrar cada corrida en la tabla:
#   Corrida 1 (Base Nominal):   alpha = 0.10, gamma = 0.95, epsilon = 0.20, episodios = 400
#   Corrida 2 (Agente Miope):   alpha = 0.10, gamma = 0.15, epsilon = 0.20, episodios = 400
#   Corrida 3 (Cero Azar):      alpha = 0.10, gamma = 0.95, epsilon = 0.00, episodios = 400
#   Corrida 4 (Infraentrenado): alpha = 0.10, gamma = 0.95, epsilon = 0.20, episodios = 30

alpha_exp = 0.10
gamma_exp = 0.95
epsilon_exp = 0.20
episodios_exp = 400


# ----------------------------------------------------------------------------
# SECCIÓN 2: ALGORITMO DE ENTRENAMIENTO TEMPORAL DIFFERENCE (Q-LEARNING)
# ----------------------------------------------------------------------------
def entrenar_q_learning(env, alpha, gamma, epsilon, episodios, semilla=42):
    random.seed(semilla)
    np.random.seed(semilla)

    n_S = env.observation_space.n
    n_A = env.action_space.n
    Q = np.zeros((n_S, n_A))
    historial_td = []

    for ep in range(episodios):
        s, _ = env.reset(seed=semilla + ep)
        errores_ep = []

        while True:
            # ----------------------------------------------------------------
            # RESOLUCIÓN EJERCICIO 1: SELECCIÓN DE ACCIÓN EPSILON-GREEDY
            # ----------------------------------------------------------------
            if random.random() < epsilon:
                a = env.action_space.sample()  # Exploración uniforme
            else:
                a = np.argmax(Q[s, :])         # Explotación de la tabla Q

            # Interacción física con la API Gymnasium
            s_sig, r, term, trunc, _ = env.step(a)

            # ----------------------------------------------------------------
            # RESOLUCIÓN EJERCICIO 2: ERROR TD Y ACTUALIZACIÓN BELLMAN
            # ----------------------------------------------------------------
            td_target = r + gamma * np.max(Q[s_sig, :])
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error

            errores_ep.append(abs(td_error))
            s = s_sig

            if term or trunc:
                break

        historial_td.append(np.mean(errores_ep))

    return Q, historial_td


# ----------------------------------------------------------------------------
# SECCIÓN 3: EXTRACTOR Y EVALUADOR DE POLÍTICA DETERMINISTA (INFERENCIA)
# ----------------------------------------------------------------------------
def evaluar_politica(env, Q, max_pasos=40, semilla=42):
    s, _ = env.reset(seed=semilla)
    ruta = [s]
    g0 = 0.0

    for _ in range(max_pasos):
        # --------------------------------------------------------------------
        # RESOLUCIÓN EJERCICIO 3: ACCIÓN DETERMINISTA VORAZ PURA (EPSILON = 0)
        # --------------------------------------------------------------------
        a = np.argmax(Q[s, :])

        s, r, term, trunc, _ = env.step(a)
        ruta.append(s)
        g0 += r

        if term or trunc:
            break

    return ruta, g0


# ----------------------------------------------------------------------------
# SECCIÓN 4: EJECUCIÓN EXPERIMENTAL Y TELEMETRÍA (TODO 4)
# ----------------------------------------------------------------------------
env_cliff = gym.make('CliffWalking-v1')

# Entrenamiento con los parámetros fijados en la Sección 1
Q_opt, logs_td = entrenar_q_learning(
    env=env_cliff,
    alpha=alpha_exp,
    gamma=gamma_exp,
    epsilon=epsilon_exp,
    episodios=episodios_exp,
    semilla=42
)

# Evaluación determinista
ruta_eval, retorno_eval = evaluar_politica(env_cliff, Q_opt, semilla=42)

pasos_usados = len(ruta_eval) - 1
alcanzo_meta = (ruta_eval[-1] == 47)
error_td_final = np.mean(logs_td[-10:]) if len(logs_td) >= 10 else np.mean(logs_td)

print("=" * 80)
print(f"{'TELEMETRÍA PARA COMPLETAR LA TABLA (CÓDIGO 2)':^80}")
print("=" * 80)
print(f" -> Tasa de Aprendizaje (alpha):       {alpha_exp:.2f}")
print(f" -> Factor de Descuento (gamma):        {gamma_exp:.2f}")
print(f" -> Tasa de Exploración (epsilon):     {epsilon_exp:.2f}")
print(f" -> Volumen de Episodios (N):          {episodios_exp}")
print(f" -> Retorno Final Acumulado (G0):      {retorno_eval:+06.1f} pts")
print(f" -> ¿Alcanzó la Meta Física (s=47)?:   {'SÍ' if alcanzo_meta else 'NO'}")
print(f" -> Pasos Empleados por el Robot:      {pasos_usados} pasos")
print(f" -> Error TD Final Promedio (|delta|): {error_td_final:.4f}")
print(f" -> Secuencia de Estados Visitados:    {ruta_eval}")
print("=" * 80)

```


