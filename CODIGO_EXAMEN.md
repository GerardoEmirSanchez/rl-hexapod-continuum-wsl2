```python
# ============================================================================
# MR3005C: PLATAFORMA DE EVALUACIÓN EXPERIMENTAL — MÓDULO 9
# Entorno: CliffWalking-v1 Modificado con Capa de Instrumentación
# ============================================================================
import gymnasium as gym
import numpy as np
import random

# ----------------------------------------------------------------------------
# 1. PARÁMETROS GLOBALES DE CONTROL Y CONFIGURACIÓN EXPERIMENTAL
# ----------------------------------------------------------------------------
SEMILLA_GLOBAL = 42
random.seed(SEMILLA_GLOBAL)
np.random.seed(SEMILLA_GLOBAL)

# ----------------------------------------------------------------------------
# 2. CAPA DE ENTORNO: REWARD WRAPPER PERSONALIZABLE
# ----------------------------------------------------------------------------
class CustomCliffEnv(gym.RewardWrapper):
    """
    Permite modificar los costos mecánicos de paso, caídas y penalizaciones
    de zona sobre la plataforma industrial CliffWalking-v1.
    """
    def __init__(self, env, costo_paso=-1.0, castigo_caida=-100.0, 
                 penalizacion_fila2=0.0, shaping_manhattan=False):
        super().__init__(env)
        self.costo_paso = costo_paso
        self.castigo_caida = castigo_caida
        self.penalizacion_fila2 = penalizacion_fila2
        self.shaping_manhattan = shaping_manhattan
        self.meta_coord = (3, 11)
        self.pos_previa = (3, 0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.pos_previa = (obs // 12, obs % 12)
        return obs, info

    def reward(self, reward):
        s_actual = self.env.unwrapped.s
        fila = s_actual // 12
        columna = s_actual % 12

        # 1. Evento de caída al acantilado
        if reward == -100.0:
            self.pos_previa = (fila, columna)
            return self.castigo_caida

        # 2. Costo base de paso
        r_mod = self.costo_paso

        # 3. Penalización asimétrica de terreno (Fila 2: cornisa de riesgo)
        if fila == 2:
            r_mod += self.penalizacion_fila2

        # 4. Inyección de potencial Manhattan (+0.5 por acercamiento neto)
        if self.shaping_manhattan:
            d_prev = abs(self.meta_coord[0] - self.pos_previa[0]) + abs(self.meta_coord[1] - self.pos_previa[1])
            d_act = abs(self.meta_coord[0] - fila) + abs(self.meta_coord[1] - columna)
            delta_phi = float(d_prev - d_act)
            r_mod += 0.5 * delta_phi

        self.pos_previa = (fila, columna)
        return r_mod

# ----------------------------------------------------------------------------
# 3. MOTOR DE APRENDIZAJE: Q-LEARNING TABULAR INSTRUMENTADO
# ----------------------------------------------------------------------------
def entrenar_agente(env, alpha=0.10, gamma=0.95, epsilon=0.20, episodios=400, semilla=42):
    random.seed(semilla)
    np.random.seed(semilla)

    n_estados = env.observation_space.n
    n_acciones = env.action_space.n
    Q = np.zeros((n_estados, n_acciones))
    
    historial_retornos = []
    historial_td_errors = []

    for ep in range(episodios):
        s, _ = env.reset(seed=semilla + ep)
        retorno_ep = 0.0
        td_errores_ep = []

        while True:
            # Mecanismo Epsilon-Greedy
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s, :])

            s_sig, r, term, trunc, _ = env.step(a)

            # Ecuación de Diferencia Temporal (Q-Learning)
            td_target = r + gamma * np.max(Q[s_sig, :])
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error

            td_errores_ep.append(abs(td_error))
            retorno_ep += r
            s = s_sig

            if term or trunc:
                break

        historial_retornos.append(retorno_ep)
        historial_td_errors.append(np.mean(td_errores_ep))

    return Q, historial_retornos, historial_td_errors

# ----------------------------------------------------------------------------
# 4. MOTOR DE INFERENCIA: EVALUACIÓN DETERMINISTA DE LA POLÍTICA VORAZ
# ----------------------------------------------------------------------------
def evaluar_politica_voraz(env, Q, max_pasos=40, semilla=42):
    s, _ = env.reset(seed=semilla)
    ruta = [s]
    retorno_eval = 0.0

    for _ in range(max_pasos):
        a_opt = np.argmax(Q[s, :])
        s, r, term, trunc, _ = env.step(a_opt)
        ruta.append(s)
        retorno_eval += r
        if term or trunc:
            break

    return ruta, retorno_eval

# ----------------------------------------------------------------------------
# 5. BLOQUE DE EJECUCIÓN NOMINAL (LÍNEA BASE)
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # Creación del entorno base estándar
    env_base = CustomCliffEnv(gym.make('CliffWalking-v1'))
    
    # Entrenamiento nominal
    Q_tabla, hist_ret, hist_td = entrenar_agente(
        env=env_base, 
        alpha=0.10, 
        gamma=0.95, 
        epsilon=0.20, 
        episodios=400, 
        semilla=SEMILLA_GLOBAL
    )

    # Evaluación voraz de la política aprendida
    trayectoria, retorno_final = evaluar_politica_voraz(env_base, Q_tabla, semilla=SEMILLA_GLOBAL)

    print("=" * 80)
    print(" TELEMETRÍA BASE DE CONTROL NOMINAL CONCLUIDA")
    print("=" * 80)
    print(f" -> Pasos ejecutados:           {len(trayectoria) - 1}")
    print(f" -> Retorno acumulado (G0):     {retorno_final:.1f} pts")
    print(f" -> ¿Meta alcanzada (s=47)?:    {'SÍ' if trayectoria[-1] == 47 else 'NO'}")
    print(f" -> Secuencia de estados:       {trayectoria}")
    print("=" * 80)
```
