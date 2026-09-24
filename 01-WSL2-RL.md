# Control por Aprendizaje por Refuerzo en WSL2: Fundamentos de RL

Guía paso a paso para el despliegue, simulación estocástica y entrenamiento de agentes de aprendizaje por refuerzo tabular sobre arquitecturas robóticas continuas y móviles en WSL2 (Ubuntu), optimizado para estaciones de trabajo con recursos de cómputo limitados.

---

## Índice de Contenidos
1. [Contexto del Sistema Ciberfísico](#contexto-del-sistema-ciberfísico)
2. [Fase 0: Limpieza total de procesos residuales](#fase-0-limpieza-total-de-procesos-residuales)
3. [Fase 1: Configuración en Ubuntu (WSL2) y Entorno Virtual `rl_env`](#fase-1-configuración-en-ubuntu-wsl2-y-entorno-virtual-rl_env)
4. [Fase 2: Estructura del Repositorio](#fase-2-estructura-del-repositorio)
5. [Fase 3: Scripts de la Sesión 1](#fase-3-scripts-de-la-sesión-1)
6. [Fase 4: Ejecución, Inspección ANSI y Validación](#fase-4-ejecución-inspección-ansi-y-validación)
7. [Fase 5: Solución a Problemas Frecuentes (Troubleshooting)](#fase-5-solución-a-problemas-frecuentes-troubleshooting)

---

## Contexto del Sistema Ciberfísico
* **Reto:** Control de aproximación y posición de un robot continuum flexible accionado por tendones montado sobre un robot móvil hexápodo para interacción con una gaveta de herramientas industrial mediante métodos basados en datos.
* **Socio de Investigación:** Chalmers University of Technology (Department of Electrical Engineering).
* **Restricción de Renderizado:** Cero ventanas nativas bloqueantes (`cv2.imshow` o `pygame.display`) para evitar fugas de memoria o congelamientos de WSLg; ejecución matricial en memoria y visualización en consola ANSI de ultra-bajo consumo (<15 MB RAM, <1% CPU).

---

## Fase 1: Configuración en Ubuntu (WSL2) y Entorno Virtual `rl_env`

En la terminal de **Ubuntu (WSL2)**:

```bash
# 1. Crear y entrar al directorio del repositorio
mkdir -p ~/cyberphysical-rl-continuum && cd ~/cyberphysical-rl-continuum

# 2. Generar archivo de dependencias
cat << 'EOF' > requirements.txt
numpy>=1.24.0
EOF

# 3. Generar .gitignore
cat << 'EOF' > .gitignore
__pycache__/
*.pyc
*.pyo
.env
rl_env/
.vscode/
EOF
```

```bash
# 1. Dependencias base del sistema
sudo apt update && sudo apt install -y python3-pip python3-venv python3-dev

# 2. Creación del entorno virtual ligero
python3 -m venv ~/rl_env

# 3. Activación del entorno virtual
source ~/rl_env/bin/activate

# 4. Crear el archivo de dependencias (ESTO ES LO QUE FALTABA)
echo "numpy>=1.24.0" > requirements.txt

# 5. Instalación de librerías esenciales
pip install --upgrade pip
pip install -r requirements.txt

```

---

## Fase 2: Estructura del Repositorio

```text
~/cyberphysical-rl-continuum/
├── README.md
├── requirements.txt
├── .gitignore
├── M9_s1_caminata_aleatoria.py        # Línea base de exploración ciega
├── M9_s1_mini_reto_estudiantes.py     # Plantilla para práctica en clase
└── M9_s1_mini_reto_resuelto.py        # Solución de referencia con perturbaciones

```

---

## Fase 3: Scripts de la Sesión 1

* **`M9_s1_caminata_aleatoria.py`:** Simula el lazo cerrado discreto con incertidumbre física (15% de probabilidad de resbalón) y penalización temporal por paso.
* **`M9_s1_mini_reto_estudiantes.py`:** Plantilla de trabajo con retos de calibración de fricción (40% resbalón) y costo energético diferenciado en retroceso (-5.0).
* **`M9_s1_mini_reto_resuelto.py`:** Evaluación estadística (20 episodios) que comprueba cuantitativamente por qué una política sin memoria fracasa ante perturbaciones mecánicas.

---

## Fase 4: Ejecución, Inspección ANSI y Validación

### 1. Ejecutar Línea Base

```bash
python M9_s1_caminata_aleatoria.py

```

### 2. Matriz de Telemetría en Consola

| Elemento | Significado Mecatrónico | Comportamiento Esperado |
| --- | --- | --- |
| `H` | Posición del robot móvil hexápodo en el eje de aproximación. | Rango dinámico [00, 12]. |
| `G` | Boca de la gaveta de herramientas fijada en celda 10. | Posición meta estática. |
| `[Paso XX]` | Pasos de control discretos ejecutados (Horizonte temporal). | Incremento secuencial. |
| `Retorno Acumulado` | Retorno escalar total recibido ($G_t$). | Penalizaciones (-1) y premio (+100). |

---

## Fase 5: Solución a Problemas Frecuentes (Troubleshooting)

### Error: `bash: python: command not found`

* **Causa:** Subshell abierta sin entorno virtual activado.
* **Solución:** Ejecutar `source ~/rl_env/bin/activate`.

### Error: `ModuleNotFoundError: No module named 'numpy'`

* **Causa:** Ejecución fuera del entorno virtual `rl_env`.
* **Solución:** Confirmar el prefijo `(rl_env)` en el prompt de la terminal y correr `pip install -r requirements.txt`.

### Visualización deformada en terminal

* **Causa:** Emuladores de terminal antiguos que no reconocen el retorno de carro `\r`.
* **Solución:** Utilizar **Windows Terminal** estándar configurado con el perfil de Ubuntu WSL2.
EOF

# 5. Generar Script 1: Caminata Aleatoria (Línea Base)


```
cat << 'EOF' > M9_s1_caminata_aleatoria.py
#!/usr/bin/env python3
"""
==============================================================================
Sistemas Ciberfísicos — Módulo 9: Aprendizaje por Refuerzo
Sesión 1: Bucle Agente-Ambiente y Política Aleatoria (Línea Base)
==============================================================================
"""

import time
import random
import sys

POS_INICIAL = 0        # Posición de inicio (lejos del cajón)
POS_META = 10          # Punto de acople exacto frente al cajón
POS_MIN = 0            # Límite trasero del área de trabajo
POS_MAX = 12           # Zona de colisión (impacto mecánico contra la gaveta)

ACCIONES = {0: "RETROCEDER", 1: "DETENERSE", 2: "AVANZAR"}

def transicion_ambiente(posicion_actual, accion):
    desplazamiento = 0
    if accion == 0:
        desplazamiento = -1
    elif accion == 2:
        desplazamiento = 1
        
    if random.random() < 0.15 and accion != 1:
        desplazamiento = 0  # Pérdida de tracción en las patas
        
    nueva_pos = posicion_actual + desplazamiento
    nueva_pos = max(POS_MIN, min(POS_MAX, nueva_pos))
    
    if nueva_pos == POS_META:
        recompensa = 100.0   # Acople completado
        terminado = True
    elif nueva_pos > POS_META:
        recompensa = -50.0   # Colisión mecánica
        terminado = True
    else:
        recompensa = -1.0    # Costo temporal por paso
        terminado = False
        
    return nueva_pos, recompensa, terminado

def render_consola_ansi(pos, paso, recompensa_acumulada):
    pista = ["·"] * (POS_MAX + 1)
    pista[POS_META] = "G"  # Gaveta (Goal)
    if pos <= POS_MAX:
        pista[pos] = "H"   # Hexápodo
    
    linea_visual = "".join(pista)
    sys.stdout.write(f"\r[Paso {paso:02d}] Pista: [{linea_visual}] | Pos: {pos:02d} | Retorno Acumulado: {recompensa_acumulada:06.1f}")
    sys.stdout.flush()

def main():
    print("\n" + "="*75)
    print(" SIMULACIÓN DE AGENTE ALEATORIO EN LÍNEA BASE (SESIÓN 1)")
    print(" Meta: Llevar el Hexápodo (H) hacia la Gaveta (G) sin colisionar")
    print("="*75 + "\n")
    
    episodios_totales = 5
    for ep in range(1, episodios_totales + 1):
        pos = POS_INICIAL
        recompensa_acumulada = 0.0
        terminado = False
        paso = 0
        
        print(f"\n--- INICIO DEL EPISODIO {ep} ---")
        while not terminado and paso < 35:
            paso += 1
            accion = random.choice([0, 1, 2])
            
            pos, r, terminado = transicion_ambiente(pos, accion)
            recompensa_acumulada += r
            
            render_consola_ansi(pos, paso, recompensa_acumulada)
            time.sleep(0.08)
            
        print()
        if pos == POS_META:
            print(f" [RESULTADO EP {ep}]: ¡ÉXITO! Acople logrado en {paso} pasos. Retorno: {recompensa_acumulada:.1f}")
        elif pos > POS_META:
            print(f" [RESULTADO EP {ep}]: ¡COLISIÓN! Impacto contra el cajón. Retorno: {recompensa_acumulada:.1f}")
        else:
            print(f" [RESULTADO EP {ep}]: TIEMPO AGOTADO (Timeout). Retorno: {recompensa_acumulada:.1f}")
            
    print("\n[INFO] Simulación concluida con éxito.\n")

if __name__ == '__main__':
    main()
EOF

# 2. Plantilla de Estudiantes (M9_s1_mini_reto_estudiantes.py) corregida
cat << 'EOF' > M9_s1_mini_reto_estudiantes.py
#!/usr/bin/env python3
"""
==============================================================================
Sistemas Ciberfísicos — Módulo 9: Aprendizaje por Refuerzo
Sesión 1: Mini-Reto Hands-On — Calibración de Perturbaciones Físicas
==============================================================================

INSTRUCCIONES:
1. Reto 1: Modificar probabilidad_resbalon a 0.40 (40%).
2. Reto 2: Si accion == 0 (retroceder), asignar recompensa de -5.0.
"""

import time
import random
import sys

POS_INICIAL = 0
POS_META = 10
POS_MIN = 0
POS_MAX = 12

def transicion_ambiente_estudiantes(posicion_actual, accion):
    desplazamiento = 0
    if accion == 0:
        desplazamiento = -1
    elif accion == 2:
        desplazamiento = 1
        
    # [RETO 1]: Modificar a 0.40
    probabilidad_resbalon = 0.15
    if random.random() < probabilidad_resbalon and accion != 1:
        desplazamiento = 0
        
    nueva_pos = posicion_actual + desplazamiento
    nueva_pos = max(POS_MIN, min(POS_MAX, nueva_pos))
    
    # [RETO 2]: Modificar penalización de retroceso
    if nueva_pos == POS_META:
        recompensa = 100.0
        terminado = True
    elif nueva_pos > POS_META:
        recompensa = -50.0
        terminado = True
    else:
        # TODO: Asignar -5.0 si accion == 0, caso contrario -1.0
        recompensa = -1.0
        terminado = False
        
    return nueva_pos, recompensa, terminado

def main():
    total_episodios = 20
    exitos = 0
    colisiones = 0
    timeouts = 0
    
    print("\n" + "="*70)
    print(" EVALUACIÓN ESTADÍSTICA DE PERTURBACIONES (20 EPISODIOS)")
    print("="*70)
    
    for ep in range(1, total_episodios + 1):
        pos = POS_INICIAL
        terminado = False
        pasos = 0
        
        while not terminado and pasos < 40:
            pasos += 1
            accion = random.choice([0, 1, 2])
            pos, r, terminado = transicion_ambiente_estudiantes(pos, accion)
            
        if pos == POS_META:
            exitos += 1
        elif pos > POS_META:
            colisiones += 1
        else:
            timeouts += 1
            
    print(f"\nResultados del Agente sobre {total_episodios} Episodios:")
    print(f" * Éxitos (Acople):      {exitos} ({(exitos/total_episodios)*100:.1f}%)")
    print(f" * Colisiones (Impacto): {colisiones} ({(colisiones/total_episodios)*100:.1f}%)")
    print(f" * Timeouts:             {timeouts} ({(timeouts/total_episodios)*100:.1f}%)")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
EOF

# 3. Solución Docente (M9_s1_mini_reto_resuelto.py) corregida
cat << 'EOF' > M9_s1_mini_reto_resuelto.py
#!/usr/bin/env python3
"""
==============================================================================
Sistemas Ciberfísicos — Módulo 9: Aprendizaje por Refuerzo
Sesión 1: Solución Oficial del Mini-Reto Hands-On
==============================================================================
"""

import time
import random
import sys

POS_INICIAL = 0
POS_META = 10
POS_MIN = 0
POS_MAX = 12

def transicion_calibrada_reto(posicion_actual, accion):
    desplazamiento = 0
    if accion == 0:
        desplazamiento = -1
    elif accion == 2:
        desplazamiento = 1
        
    # RETO 1: Resbalón al 40%
    if random.random() < 0.40 and accion != 1:
        desplazamiento = 0
        
    nueva_pos = posicion_actual + desplazamiento
    nueva_pos = max(POS_MIN, min(POS_MAX, nueva_pos))
    
    # RETO 2: Penalización por retroceso
    if nueva_pos == POS_META:
        recompensa = 100.0
        terminado = True
    elif nueva_pos > POS_META:
        recompensa = -50.0
        terminado = True
    else:
        recompensa = -5.0 if accion == 0 else -1.0
        terminado = False
        
    return nueva_pos, recompensa, terminado

def main():
    exitos = 0
    colisiones = 0
    timeouts = 0
    total_episodios = 20
    
    print("\n" + "="*75)
    print(" EVALUACIÓN ESTADÍSTICA DEL MINI-RETO (20 EPISODIOS CON PERTURBACIONES)")
    print("="*75)
    
    for ep in range(1, total_episodios + 1):
        pos = POS_INICIAL
        terminado = False
        pasos = 0
        
        while not terminado and pasos < 40:
            pasos += 1
            accion = random.choice([0, 1, 2])
            pos, r, terminado = transicion_calibrada_reto(pos, accion)
            
        if pos == POS_META:
            exitos += 1
        elif pos > POS_META:
            colisiones += 1
        else:
            timeouts += 1
            
    print(f"\nResultados tras {total_episodios} pruebas del agente ciego:")
    print(f" -> Tasa de Éxitos:     {(exitos / total_episodios) * 100:.1f}% ({exitos}/{total_episodios})")
    print(f" -> Tasa de Colisiones: {(colisiones / total_episodios) * 100:.1f}% ({colisiones}/{total_episodios})")
    print(f" -> Tasa de Timeouts:   {(timeouts / total_episodios) * 100:.1f}% ({timeouts}/{total_episodios})")
    print("\nConclusión Técnica:")
    print("Bajo fricción degradada y penalización dinámica, la probabilidad de éxito")
    print("de una política aleatoria cae drásticamente. Se requiere aprendizaje formal.")
    print("="*75 + "\n")

if __name__ == '__main__':
    main()
EOF

# Actualizar el commit en Git
git add M9_s1_caminata_aleatoria.py M9_s1_mini_reto_estudiantes.py M9_s1_mini_reto_resuelto.py
git commit -m "fix: corregir indentacion estricta en scripts de sesion 1" 2>/dev/null || true
```
```
# Ejecución inmediata para validación
python M9_s1_caminata_aleatoria.py
```

# 6. Generar Script 2: Plantilla de Mini-Retos para Alumnos

```
cat << 'EOF' > M9_s1_mini_reto_estudiantes.py
#!/usr/bin/env python3
"""

# MR3005C: Sistemas Ciberfísicos — Módulo 9: Aprendizaje por Refuerzo
Sesión 1: Mini-Reto Hands-On — Calibración de Perturbaciones Físicas

INSTRUCCIONES:

1. Reto 1: Modificar probabilidad_resbalon a 0.40 (40%).
2. Reto 2: Si accion == 0 (retroceder), asignar recompensa de -5.0.
"""

import time
import random
import sys

POS_INICIAL = 0
POS_META = 10
POS_MIN = 0
POS_MAX = 12

def transicion_ambiente_estudiantes(posicion_actual, accion):
desplazamiento = 0
if accion == 0:
desplazamiento = -1
elif accion == 2:
desplazamiento = 1


# [RETO 1]: Modificar a 0.40
probabilidad_resbalon = 0.15
if random.random() < probabilidad_resbalon and accion != 1:
    desplazamiento = 0
    
nueva_pos = posicion_actual + desplazamiento
nueva_pos = max(POS_MIN, min(POS_MAX, nueva_pos))

# [RETO 2]: Modificar penalización de retroceso
if nueva_pos == POS_META:
    recompensa = 100.0
    terminado = True
elif nueva_pos > POS_META:
    recompensa = -50.0
    terminado = True
else:
    # TODO: Asignar -5.0 si accion == 0, caso contrario -1.0
    recompensa = -1.0
    terminado = False
    
return nueva_pos, recompensa, terminado



def main():
total_episodios = 20
exitos = 0
colisiones = 0
timeouts = 0

print("\n" + "="*70)
print(" EVALUACIÓN ESTADÍSTICA DE PERTURBACIONES (20 EPISODIOS)")
print("="*70)

for ep in range(1, total_episodios + 1):
    pos = POS_INICIAL
    terminado = False
    pasos = 0
    
    while not terminado and pasos < 40:
        pasos += 1
        accion = random.choice([0, 1, 2])
        pos, r, terminado = transicion_ambiente_estudiantes(pos, accion)
        
    if pos == POS_META:
        exitos += 1
    elif pos > POS_META:
        colisiones += 1
    else:
        timeouts += 1
        
print(f"\nResultados del Agente sobre {total_episodios} Episodios:")
print(f" * Éxitos (Acople):      {exitos} ({(exitos/total_episodios)*100:.1f}%)")
print(f" * Colisiones (Impacto): {colisiones} ({(colisiones/total_episodios)*100:.1f}%)")
print(f" * Timeouts:             {timeouts} ({(timeouts/total_episodios)*100:.1f}%)")
print("="*70 + "\n")


if **name** == '**main**':
main()
EOF
```

# 7. Generar Script 3: Mini-Reto Resuelto Oficial

```
cat << 'EOF' > M9_s1_mini_reto_resuelto.py
#!/usr/bin/env python3
"""

# MR3005C: Sistemas Ciberfísicos — Módulo 9: Aprendizaje por Refuerzo
Sesión 1: Solución Oficial del Mini-Reto Hands-On

"""

import time
import random
import sys

POS_INICIAL = 0
POS_META = 10
POS_MIN = 0
POS_MAX = 12

def transicion_calibrada_reto(posicion_actual, accion):
desplazamiento = 0
if accion == 0:
desplazamiento = -1
elif accion == 2:
desplazamiento = 1


# RETO 1: Resbalón al 40%
if random.random() < 0.40 and accion != 1:
    desplazamiento = 0
    
nueva_pos = posicion_actual + desplazamiento
nueva_pos = max(POS_MIN, min(POS_MAX, nueva_pos))

# RETO 2: Penalización por retroceso
if nueva_pos == POS_META:
    recompensa = 100.0
    terminado = True
elif nueva_pos > POS_META:
    recompensa = -50.0
    terminado = True
else:
    recompensa = -5.0 if accion == 0 else -1.0
    terminado = False
    
return nueva_pos, recompensa, terminado



def main():
exitos = 0
colisiones = 0
timeouts = 0
total_episodios = 20

print("\n" + "="*75)
print(" EVALUACIÓN ESTADÍSTICA DEL MINI-RETO (20 EPISODIOS CON PERTURBACIONES)")
print("="*75)

for ep in range(1, total_episodios + 1):
    pos = POS_INICIAL
    terminado = False
    pasos = 0
    
    while not terminado and pasos < 40:
        pasos += 1
        accion = random.choice([0, 1, 2])
        pos, r, terminado = transicion_calibrada_reto(pos, accion)
        
    if pos == POS_META:
        exitos += 1
    elif pos > POS_META:
        colisiones += 1
    else:
        timeouts += 1
        
print(f"\nResultados tras {total_episodios} pruebas del agente ciego:")
print(f" -> Tasa de Éxitos:     {(exitos / total_episodios) * 100:.1f}% ({exitos}/{total_episodios})")
print(f" -> Tasa de Colisiones: {(colisiones / total_episodios) * 100:.1f}% ({colisiones}/{total_episodios})")
print(f" -> Tasa de Timeouts:   {(timeouts / total_episodios) * 100:.1f}% ({timeouts}/{total_episodios})")
print("\nConclusión Técnica:")
print("Bajo fricción degradada y penalización dinámica, la probabilidad de éxito")
print("de una política aleatoria cae drásticamente. Se requiere aprendizaje formal.")
print("="*75 + "\n")



if **name** == '**main**':
main()
EOF
```

# 8. Configurar entorno virtual e instalar dependencias

```bash
python3 -m venv ~/rl_env
source ~/rl_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```
