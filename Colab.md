

**Módulo:** MR3005C — Sistemas Ciberfísicos (Módulo 9: Fundamentos de Aprendizaje por Refuerzo)
**Sesiones Integradas:** Sesión 1 (Introducción y Bucle Agente-Ambiente) y Sesión 2 (MDPs, Retorno $G_t$ y Ecuaciones de Bellman)
**Socio de Investigación:** Chalmers University of Technology (_Department of Electrical Engineering_)
**Entorno de Trabajo:** Google Colab (Python 3 en nube con visualización vectorial mediante Matplotlib y Seaborn)
## 1. Guía de Inicio Rápido en Google Colab

Google Colab permite ejecutar código Python interactivo en servidores remotos sin instalar librerías locales, compilar código en la terminal ni agotar la memoria de laptops escolares.
### Paso 1: Crear el Cuaderno de Trabajo
1. Abre tu navegador web (Google Chrome o Microsoft Edge) e ingresa a:
    ```Plaintext
    https://colab.research.google.com
    ```
    
2. Inicia sesión con tu cuenta.
3. En la ventana emergente, haz clic en el botón azul **"Nuevo cuaderno"** (_New notebook_).
4. En la esquina superior izquierda, haz clic sobre el nombre por defecto (`Untitled0.ipynb`) y renómbralo como:
    ```Plaintext
    MR3005C_M9_Fundamentos_RL.ipynb
    ```
    ### Paso 2: Dinámica de Celdas (Texto y Código)

- **Celdas de Código (`+ Código`):** Espacios donde se escribe y ejecuta Python. Para correr una celda, presiona **`Shift + Enter`** o haz clic en el botón circular de reproducción (_Play_).
- **Celdas de Texto (`+ Texto`):** Espacios en formato Markdown para documentar hipótesis, ecuaciones y conclusiones.
- **Reinicio de Entorno:** Si en algún momento una variable genera conflicto o el bucle se traba, ve al menú superior: **Entorno de ejecución $\rightarrow$ Reiniciar sesión** (_Runtime $\rightarrow$ Restart session_).

## 2. Encuadre Mecatrónico del Reto

En los módulos previos de Visión Artificial resolvimos la estimación métrica tridimensional de la escena (mediante ArUco y YOLOv8), entregando la distancia frontal $Z$ y el ángulo $\theta$ hacia la gaveta metálica de herramientas.

El reto  consiste en gobernar la cinemática acoplada de un robot móvil hexápodo que transporta un manipulador continuo flexible de TPU accionado por tendones:

- **El Problema del Manipulador Compliante:** A diferencia de un robot industrial rígido, el brazo elástico posee teóricamente infinitos grados de libertad. Al sujetar una pieza de hasta $50\text{ g}$, la gravedad y la flexión no lineal del material modifican su curvatura.
- **La Solución Basada en Datos (RL):** Antes de penetrar en el cajón, el robot hexápodo debe aprender de manera autónoma a alinearse milimétricamente frente a la boca de la gaveta sin colisionar contra la chapa exterior.

## 3. Construcción del Cuaderno Interactivo Celda por Celda

Copia y pega el contenido exacto de cada celda en tu cuaderno de Google Colab en el orden indicado.
### Celda 1 (Texto): Título Institucional y Objetivos

```Markdown
# MR3005C: Sistemas Ciberfísicos — Módulo 9: Aprendizaje por Refuerzo
## Práctica Integrada: Del Bucle Agente-Ambiente a la Ecuación de Bellman

* **Institución y Socio Formador:** Tecnológico de Monterrey y Chalmers University of Technology.
* **Objetivos de Aprendizaje:**
  1. Comprender la interacción cibernética Agente-Ambiente bajo incertidumbre mecánica (resbalones en patas).
  2. Evaluar empíricamente el fracaso de una política aleatoria mediante simulaciones de Monte Carlo.
  3. Formular matemáticamente un Proceso de Decisión de Markov (MDP) y calcular el retorno acumulado descontado $G_t$.
  4. Resolver numéricamente la Ecuación de Expectación de Bellman para obtener la política óptima $\pi^*(s)$.
```

### Celda 2 (Código): Configuración Gráfica y Librerías Base

Configura la estética visual de las gráficas de telemetría y fija la semilla pseudoaleatoria para reproducibilidad experimental.

```Python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Configuración del estilo gráfico profesional
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 120

# Fijar semilla estocástica
random.seed(42)
np.random.seed(42)

print("[OK] Entorno configurado: NumPy, Matplotlib y Seaborn listos.")
```

### Celda 3 (Texto): Fundamento del Bucle Agente-Ambiente

```Markdown
### 1. El Bucle Agente-Ambiente y la Dinámica de Aproximación

El robot hexápodo opera en un espacio unidimensional discreto hacia el cajón:
* **Espacio de Estados $\mathcal{S}$:** Coordenadas enteras $\{0, 1, 2, \dots, 12\}$.
  * $S_0 = 0$: Posición de partida (lejos del objetivo).
  * $S_{\text{meta}} = 10$: Boca de acople de la gaveta industrial (tolerancia de inserción).
  * $S > 10$: Zona de impacto mecánico contra el metal.
* **Espacio de Acciones $\mathcal{A}$:** $\{0: \text{Retroceder}, 1: \text{Detenerse}, 2: \text{Avanzar}\}$.
* **Incertidumbre Física:** Sobre pisos pulidos de laboratorio, las patas del hexápodo tienen un $15\%$ de probabilidad de patinar (resbalón), provocando un desplazamiento nulo ($0$).
```

### Celda 4 (Código): Clase del Entorno Físico 1D

Implementa la clase `EntornoAproximacionHexapodo` siguiendo el ciclo de vida de un entorno de control por refuerzo.
```Python
class EntornoAproximacionHexapodo:
    """
    Simulador físico 1D de la aproximación del hexápodo hacia la gaveta de herramientas.
    """
    def __init__(self, prob_resbalon=0.15, penalizacion_reversa=-1.0):
        self.POS_INICIAL = 0
        self.POS_META = 10         # Punto de acople exacto
        self.POS_MIN = 0           # Pared trasera del laboratorio
        self.POS_MAX = 12          # Zona de colisión física
        self.prob_resbalon = prob_resbalon
        self.penalizacion_reversa = penalizacion_reversa
        self.reset()
        
    def reset(self):
        """Reinicia el robot a su condición inicial."""
        self.pos = self.POS_INICIAL
        self.paso = 0
        self.terminado = False
        return self.pos
        
    def step(self, accion):
        """
        Ejecuta un paso de control discreto.
        Retorna: (nueva_posicion, recompensa, terminado, resbalon)
        """
        self.paso += 1
        desplazamiento = 0
        
        if accion == 0:
            desplazamiento = -1
        elif accion == 2:
            desplazamiento = 1
            
        # Simulación de resbalón mecánico
        resbalon = False
        if random.random() < self.prob_resbalon and accion != 1:
            desplazamiento = 0
            resbalon = True
            
        self.pos = max(self.POS_MIN, min(self.POS_MAX, self.pos + desplazamiento))
        
        # Evaluación de la señal de refuerzo escalar
        if self.pos == self.POS_META:
            recompensa = 100.0   # Acople exitoso frente al cajón
            self.terminado = True
        elif self.pos > self.POS_META:
            recompensa = -50.0   # Colisión mecánica
            self.terminado = True
        else:
            recompensa = self.penalizacion_reversa if accion == 0 else -1.0
            self.terminado = False
            
        return self.pos, recompensa, self.terminado, resbalon

print("[OK] Clase EntornoAproximacionHexapodo compilada exitosamente.")
```

### Celda 5 (Texto): Línea Base — La Caminata Aleatoria
```Markdown
### 2. Simulación de Trayectorias de una Política Aleatoria

Para justificar el uso de algoritmos avanzados de aprendizaje, primero evaluamos una **política uniforme aleatoria** ($\pi(a \mid s) = 1/3$).
En cada paso de tiempo, el agente elige al azar entre retroceder, detenerse o avanzar sin utilizar ninguna memoria del pasado.
```

### Celda 6 (Código): Visualización Temporal de Trayectorias

Ejecuta 8 episodios de la política aleatoria y grafica la evolución del estado $S_t$ en función del tiempo.

```Python
env = EntornoAproximacionHexapodo(prob_resbalon=0.15, penalizacion_reversa=-1.0)
n_episodios = 8
max_pasos = 35

plt.figure(figsize=(12, 5.5))
colores = plt.cm.tab10(np.linspace(0, 1, n_episodios))

for ep in range(n_episodios):
    pos = env.reset()
    historial = [pos]
    
    while not env.terminado and env.paso < max_pasos:
        accion = random.choice([0, 1, 2])
        pos, _, terminado, _ = env.step(accion)
        historial.append(pos)
        
    estado_final = "Éxito" if pos == 10 else ("Colisión" if pos > 10 else "Timeout")
    plt.plot(historial, marker='o', markersize=4, label=f'Ep {ep+1}: {estado_final}', color=colores[ep], alpha=0.85)

# Delimitación de regiones físicas
plt.axhspan(0, 9, color='gray', alpha=0.10, label='Zona de Tránsito')
plt.axhline(10, color='forestgreen', linestyle='--', linewidth=2.5, label='Meta: Gaveta (Pos 10)')
plt.axhspan(11, 12, color='crimson', alpha=0.20, label='Zona de Colisión Mecánica')

plt.title('Trayectorias Temporales del Hexápodo bajo Política Aleatoria (Línea Base)', fontsize=14, fontweight='bold')
plt.xlabel('Paso de Control Discreto (t)', fontsize=12)
plt.ylabel('Posición del Robot (S_t)', fontsize=12)
plt.ylim(-0.5, 12.5)
plt.xlim(0, max_pasos)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.show()
```

### Celda 7 (Código): Análisis Estadístico de Monte Carlo (500 Intentos)

Calcula cuantitativamente la tasa de fracaso de la política aleatoria mediante una ráfaga de 500 episodios.

```Python
def corrida_montecarlo(prob_resbalon=0.15, penalizacion_reversa=-1.0, episodios=500):
    env_mc = EntornoAproximacionHexapodo(prob_resbalon, penalizacion_reversa)
    resultados = {"Éxitos": 0, "Colisiones": 0, "Timeouts": 0}
    retornos = []
    
    for _ in range(episodios):
        env_mc.reset()
        retorno = 0
        while not env_mc.terminado and env_mc.paso < 35:
            accion = random.choice([0, 1, 2])
            _, r, term, _ = env_mc.step(accion)
            retorno += r
            
        retornos.append(retorno)
        if env_mc.pos == 10:
            resultados["Éxitos"] += 1
        elif env_mc.pos > 10:
            resultados["Colisiones"] += 1
        else:
            resultados["Timeouts"] += 1
            
    return resultados, retornos

res_base, ret_base = corrida_montecarlo(prob_resbalon=0.15, penalizacion_reversa=-1.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 1. Gráfica de Desenlaces
barras = ax1.bar(list(res_base.keys()), list(res_base.values()), color=['#22c55e', '#ef4444', '#f59e0b'], edgecolor='black', alpha=0.85)
ax1.set_title('Desenlace de 500 Intentos Aleatorios', fontweight='bold')
ax1.set_ylabel('Frecuencia Absoluta')
for bar in barras:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, h + 5, f'{h} ({(h/500)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

# 2. Distribución de Retornos
sns.histplot(ret_base, bins=25, kde=True, ax=ax2, color='#0284c7')
ax2.axvline(np.mean(ret_base), color='red', linestyle='--', label=f'Media: {np.mean(ret_base):.1f} pts')
ax2.set_title('Distribución del Retorno Total G_0', fontweight='bold')
ax2.set_xlabel('Retorno Acumulado')
ax2.set_ylabel('Densidad')
ax2.legend()

plt.tight_layout()
plt.show()
```

### Celda 8 (Texto): Formalización con Procesos de Decisión de Markov (MDP)

```Markdown
### 3. Procesos de Decisión de Markov y el Retorno Descontado $G_t$

El retorno acumulado $G_t$ pondera las consecuencias de las decisiones en el tiempo:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Aplicando recursividad temporal (Principio de Bellman):
$$G_t = R_{t+1} + \gamma G_{t+1}$$

* $\gamma \to 0$ (**Miopía**): El robot solo valora la recompensa inmediata.
* $\gamma \to 1$ (**Previsión**): El robot planifica metas lejanas y anticipa colisiones.
```

### Celda 9 (Código): Barrido Continuo de Sensibilidad de $\gamma$

Evalúa cómo varía el retorno esperado inicial $G_0$ frente al factor de descuento $\gamma$ para tres trayectorias alternativas.

```Python
def calcular_retorno_vectorial(recompensas, gamma):
    """Calcula recursivamente G_t de derecha a izquierda."""
    T = len(recompensas)
    G = np.zeros(T, dtype=np.float64)
    acumulado = 0.0
    for t in reversed(range(T)):
        acumulado = recompensas[t] + gamma * acumulado
        G[t] = acumulado
    return G

# Tres secuencias alternativas de desempeño
ruta_rapida = [-1.0, -1.0, -1.0, 100.0]           # 4 pasos directos
ruta_cautelosa = [-1.0] * 7 + [100.0]             # 8 pasos rodeando
ruta_choque = [-1.0, -1.0, -50.0]                 # Colisión en paso 3

gammas = np.linspace(0.01, 0.99, 200)
g0_rapida = [calcular_retorno_vectorial(ruta_rapida, g)[0] for g in gammas]
g0_cautelosa = [calcular_retorno_vectorial(ruta_cautelosa, g)[0] for g in gammas]
g0_choque = [calcular_retorno_vectorial(ruta_choque, g)[0] for g in gammas]

plt.figure(figsize=(11, 5.5))
plt.plot(gammas, g0_rapida, label='Ruta Rápida (4 pasos -> Meta +100)', color='#16a34a', linewidth=2.5)
plt.plot(gammas, g0_cautelosa, label='Ruta Cautelosa (8 pasos -> Meta +100)', color='#0284c7', linewidth=2.5)
plt.plot(gammas, g0_choque, label='Ruta Temeraria (3 pasos -> Choque -50)', color='#dc2626', linewidth=2.5, linestyle=':')

plt.axvline(0.38, color='purple', linestyle='--', alpha=0.7, label='Umbral Crítico de Miopía (γ ≈ 0.38)')
plt.title('Sensibilidad del Retorno G_0 ante el Factor de Descuento (γ)', fontsize=14, fontweight='bold')
plt.xlabel('Factor de Descuento Temporal (γ)', fontsize=12)
plt.ylabel('Retorno Esperado Inicial G_0 (Puntos)', fontsize=12)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Celda 10 (Texto): La Ecuación de Bellman para $Q(s, a)$
```Markdown
### 4. Selección Óptima de Acción mediante Bellman

Para decidir racionalmente en un estado $s$, el agente evalúa la función de valor de acción $Q(s, a)$:
$$Q(s, a) = \sum_{s'} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V(s') \right]$$

La política óptima $\pi^*(s)$ selecciona la acción que maximiza $Q$:
$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$
```

### Celda 11 (Código): Resolución Numérica de Bellman y Decisión Óptima

Modela una situación crítica a un paso de la gaveta considerando transiciones estocásticas y valores futuros $V(s')$ calculados previamente.

```Python
def resolver_bellman_un_paso(valores_v_siguiente, gamma=0.95):
    """
    Calcula los valores Q(s, a) para el estado actual frente a la gaveta.
    """
    q_dict = {}
    
    # Acción 0: Retroceder (100% determinista hacia S_atras, R = -5.0)
    q_dict["RETROCEDER"] = -5.0 + gamma * valores_v_siguiente["S_atras"]
    
    # Acción 1: Detenerse (100% determinista en S_actual, R = -1.0)
    q_dict["DETENERSE"] = -1.0 + gamma * valores_v_siguiente["S_actual"]
    
    # Acción 2: Avanzar (80% llega a S_meta con R=+100, 20% resbala con R=-1)
    q_meta = 100.0 + gamma * valores_v_siguiente["S_meta"]
    q_resbalon = -1.0 + gamma * valores_v_siguiente["S_actual"]
    q_dict["AVANZAR"] = (0.80 * q_meta) + (0.20 * q_resbalon)
    
    return q_dict

# Valores de estado asignados por el entorno
v_vecinos = {"S_atras": 5.0, "S_actual": 25.0, "S_meta": 0.0}
q_valores = resolver_bellman_un_paso(v_vecinos, gamma=0.95)

plt.figure(figsize=(9, 4.5))
colores_barras = ['#94a3b8', '#64748b', '#22c55e']
barras = plt.bar(list(q_valores.keys()), list(q_valores.values()), color=colores_barras, edgecolor='black', width=0.45)

plt.title('Valores de Acción Q(s, a) según la Ecuación de Expectación de Bellman', fontweight='bold')
plt.ylabel('Valor Q(s, a) (Retorno Futuro Esperado)')
plt.axhline(0, color='black', linewidth=1)

for bar in barras:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + (1.5 if y >= 0 else -4.0), f'{y:.2f} pts', ha='center', fontweight='bold', fontsize=11)

mejor_a = max(q_valores, key=q_valores.get)
plt.annotate(f'Política Óptima π*(s):\n¡Ejecutar {mejor_a}!', 
             xy=(2, q_valores[mejor_a]), xytext=(1.1, q_valores[mejor_a]-22),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.7))

plt.ylim(min(q_valores.values()) - 10, max(q_valores.values()) + 15)
plt.tight_layout()
plt.show()
```

### Celda 12 (Texto): Enunciado del Mini-Reto Hands-On

```Markdown
---
### 🛠️ MINI-RETO HANDS-ON EN EQUIPO (25 Minutos)

Un lote de lubricante derramado en la zona de pruebas modifica severamente la fricción del suelo.
Modifica la celda siguiente para resolver los siguientes dos escenarios de ingeniería mecatrónica:

1. **Escenario de Fricción Degradada (Patinaje Extremo):**
   * La probabilidad de resbalón al avanzar sube al **`0.65` (65%)**.
   * El costo de retroceso sube a **`-15.0` puntos** debido a sobreesfuerzo de los servomotores.
2. **Escenario de Agente Miope:**
   * Evalúa la decisión óptima si el factor de descuento colapsa a **$\gamma = 0.05$**.

**Preguntas a Responder en su Bitácora:**
* ¿Bajo qué probabilidad exacta de resbalón la acción de `DETENERSE` supera numéricamente a `AVANZAR`?
* Si $\gamma = 0.05$, ¿cuál es el nuevo valor de $Q(s, \text{AVANZAR})$ y qué patología exhibirá el hexápodo físico frente a la gaveta?
---
```

### Celda 13 (Código): Plantilla del Mini-Reto para Alumnos

```Python
# ============================================================================
# ESPACIO DE TRABAJO DEL ESTUDIANTE (MINI-RETO)
# ============================================================================

# Parámetros del Reto
resbalon_estudiante = 0.65       # 65% probabilidad de resbalón
castigo_reversa_reto = -15.0     # Sobreesfuerzo en retroceso
gamma_reto = 0.95                # Cambiar a 0.05 en el segundo análisis

v_estados = {"S_atras": 5.0, "S_actual": 25.0, "S_meta": 0.0}

q_reto = {}

# 1. Retroceder
q_reto["RETROCEDER"] = castigo_reversa_reto + gamma_reto * v_estados["S_atras"]

# 2. Detenerse
q_reto["DETENERSE"] = -1.0 + gamma_reto * v_estados["S_actual"]

# 3. Avanzar (TODO ALUMNOS: Implementar la ponderación probabilística)
p_exito = 1.0 - resbalon_estudiante
q_meta_calc = 100.0 + gamma_reto * v_estados["S_meta"]
q_resbalon_calc = -1.0 + gamma_reto * v_estados["S_actual"]

q_reto["AVANZAR"] = (p_exito * q_meta_calc) + (resbalon_estudiante * q_resbalon_calc)

# Visualización de Resultados
print("="*60)
print(f" RESULTADOS DEL MINI-RETO (Resbalón: {resbalon_estudiante*100:.0f}% | Gamma: {gamma_reto})")
print("="*60)
for act, val in q_reto.items():
    print(f" -> Q(S, {act:<10}): {val:.2f} pts")

ganadora = max(q_reto, key=q_reto.get)
print(f"\n Accion Optima Seleccionada: ¡{ganadora}!")
print("="*60)
```

### Celda 14 (Texto): Solución Oficial y Análisis Teórico (Uso del Profesor)


```Markdown
### 5. Análisis Técnico de la Solución del Mini-Reto

1. **Efecto del Patinaje Extremo ($\text{Resbalón} = 0.65$ con $\gamma = 0.95$):**
   * $Q(s, \text{AVANZAR}) = 0.35 \times 100.0 + 0.65 \times (-1.0 + 0.95 \times 25.0) = 35.0 + 0.65 \times 22.75 = \mathbf{49.79\text{ pts}}$.
   * $Q(s, \text{DETENERSE}) = -1.0 + 0.95 \times 25.0 = \mathbf{22.75\text{ pts}}$.
   * A pesar de que el robot patina dos de cada tres veces, avanzar sigue siendo la decisión óptima ($49.79 > 22.75$) debido a la magnitud de la recompensa de acople ($+100$).

2. **Efecto de la Miopía Extrema ($\gamma = 0.05$):**
   * $Q(s, \text{DETENERSE}) = -1.0 + 0.05 \times 25.0 = \mathbf{+0.25\text{ pts}}$.
   * $Q(s, \text{AVANZAR}) = 0.35 \times 100.0 + 0.65 \times (-1.0 + 0.05 \times 25.0) = 35.0 + 0.65 \times 0.25 = \mathbf{35.16\text{ pts}}$.
   * Si la meta estuviera a 2 pasos de distancia, el premio de acople se descontaría como $(0.05)^2 \times 100 = 0.25$, provocando que el robot prefiera detenerse indefinidamente para evitar el riesgo inmediato de resbalar.
```

## 4. Guía de Evaluación y Rúbrica de Salida

Para registrar la acreditación del estudiante al finalizar la sesión, verifica la correcta visualización de los tres gráficos clave:

  

|**Entregable Visual en Colab**|**Criterio de Aprobación Técnica**|**Error Común a Corregir**|
|---|---|---|
|**Gráfico 1: Trayectorias Aleatorias (Celda 6)**|Debe mostrar al menos 8 líneas de colores oscilando erráticamente en la zona gris, con la mayoría terminando en _Timeout_ en el paso 35.|El gráfico aparece vacío si no se iteró el bucle con `while not env.terminado`.|
|**Gráfico 2: Curva de Gamma (Celda 9)**|La curva verde de la ruta rápida debe cruzar por encima de la roja antes de $\gamma \approx 0.40$.|Graficar retornos sin invertir el orden temporal (calculando de izquierda a derecha en lugar de reversa).|
|**Gráfico 3: Barras de Bellman (Celda 11)**|La barra de `AVANZAR` debe ser verde y superar los $+80\text{ puntos}$ frente a las demás acciones.|Olvidar multiplicar el valor futuro $V(s')$ por $\gamma$ en el término estocástico de resbalón.|

Una vez completadas todas las celdas, los estudiantes deben exportar su trabajo mediante el menú superior: **Archivo $\rightarrow$ Descargar $\rightarrow$ Descargar .ipynb** y subir el archivo al repositorio oficial de su equipo (`cyberphysical-rl-continuum`).
