### 1.1 Implementación de Código en Clase (Google Colab / WSL2)

#### Celda 1: Código — Nivelación de PyTorch (Autograd, nn.Linear y el Bucle Canónico)
```python
# ============================================================================
# CELDA 1: FUNDAMENTOS OPERATIVOS DE PYTORCH PARA RL (TENSORES Y AUTOGRAD)
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Verificación de hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[HARDWARE] Entorno configurado en: {device}")

# 1. Tensores y Autograd: El mecanismo de propagación hacia atrás
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([1.5])
y_target = torch.tensor([5.0])

# Pase hacia adelante (Forward)
y_pred = w * x + b
loss = (y_pred - y_target) ** 2  # Error cuadrático puntual

# Pase hacia atrás (Backward)
loss.backward()
print(f"[AUTOGRAD] Gradiente d(Loss)/dw: {w.grad.item():.4f} | d(Loss)/db: {b.grad.item():.4f}")

# 2. Demostración del vaciado obligatorio de gradientes
# En PyTorch, los gradientes se ACUMULAN por defecto. En RL esto corrompería el lote.
w.grad.zero_()
b.grad.zero_()
print(f"[AUTOGRAD] Gradientes tras zero_grad(): w.grad = {w.grad.item()}")
```

#### Celda 2: Código — Representación One-Hot y Red DQN con Inicialización He
```python
# ============================================================================
# CELDA 2: ARQUITECTURA DE LA RED DQN CON INICIALIZACIÓN HE (KAIMING UNIFORM)
# Referencia: DQN_Agent.pdf (Pág 3, Sección 3.2 y 3.3)
# ============================================================================
class DQN_Network(nn.Module):
    """
    Aproximador funcional feedforward con 2 capas ocultas.
    Arquitectura: Input(N) -> FC(12) -> ReLU -> FC(8) -> ReLU -> FC(Num_Acciones)
    """
    def __init__(self, num_actions: int, input_dim: int):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )
        
        # Inicialización He (Kaiming Uniform) adaptada a activaciones ReLU
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.FC(x)

# Prueba unitaria de dimensiones: FrozenLake 8x8 -> 64 estados discretos, 4 acciones
test_net = DQN_Network(num_actions=4, input_dim=64).to(device)
dummy_state = torch.zeros((1, 64), dtype=torch.float32, device=device)
dummy_state[0, 15] = 1.0  # Estado 15 activo (One-Hot)
q_values_test = test_net(dummy_state)
print(f"[TEST RED] Salida de Q(s, a): {q_values_test.shape} -> Valores: {q_values_test.detach().cpu().numpy()}")
```

#### Celda 3: Código — Búfer de Repetición Circular con collections.deque
```python
# ============================================================================
# CELDA 3: REPLAY MEMORY CIRCULAR EFICIENTE (O(1) APPEND)
# Referencia: DQN_Agent.pdf (Pág 4, Sección 4.1)
# ============================================================================
from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        """Almacena una transición física completa."""
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size: int, device: torch.device):
        """Muestreo uniforme sin reemplazo y conversión directa a tensores."""
        indices = np.random.choice(len(self.dones), size=batch_size, replace=False)
        
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices])
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices])
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)
        
        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.dones)
```

#### Celda 4: Código — Clase Completa del Agente DQN (DQN_Agent)
```python
# ============================================================================
# CELDA 4: AGENTE DQN INTEGRAL (BELLMAN TARGET, CLIPPING Y HARD UPDATE)
# Referencia: DQN_Agent.pdf (Pág 5-8, Secciones 4.2 - 4.8)
# ============================================================================
class DQNAgent:
    def __init__(self, env, lr=6e-4, gamma=0.93, batch_size=32, mem_capacity=4000,
                 target_update_freq=10, eps_max=0.999, eps_min=0.01, eps_decay=0.999,
                 clip_norm=3.0, dev=device):
        self.env = env
        self.device = dev
        self.n_obs = env.observation_space.n
        self.n_act = env.action_space.n
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.clip_norm = clip_norm
        
        # Red Principal y Red Objetivo
        self.main_net = DQN_Network(self.n_act, self.n_obs).to(self.device)
        self.target_net = DQN_Network(self.n_act, self.n_obs).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        
        self.memory = ReplayMemory(mem_capacity)
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.total_steps = 0
        self.loss_history = []
        self.reward_history = []

    def one_hot_encode(self, state: int) -> np.ndarray:
        vec = np.zeros(self.n_obs, dtype=np.float32)
        vec[state] = 1.0
        return vec

    def select_action(self, state_vec: np.ndarray) -> int:
        # Exploración Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.main_net(s_t)
            return torch.argmax(q_vals).item()

    def update_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def learn(self):
        # Condición de Gating estricta: Memoria suficiente Y al menos un éxito observado
        if len(self.memory) < self.batch_size or sum(self.reward_history) == 0:
            return

        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size, self.device)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # 1. Q Predicho por la Red Principal: Q(s, a; \theta)
        pred_q = self.main_net(states).gather(dim=1, index=actions)

        # 2. Bellman Target mediante la Red Objetivo Congelada: r + \gamma \max Q(s', a'; \theta^-)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            next_q[dones] = 0.0  # Estado terminal tiene valor futuro cero
            target_q = rewards + self.gamma * next_q

        loss = self.criterion(pred_q, target_q)

        # 3. Optimización con Recorte de Gradiente (Gradient Clipping)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), self.clip_norm)
        self.optimizer.step()

        self.loss_history.append(loss.item())

        # 4. Hard Target Update periódico
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
```

#### Celda 5: Código — Bucle de Entrenamiento y Evaluación Comparativa (FrozenLake 4x4 y 8x8)
```python
# ============================================================================
# CELDA 5: EXPERIMENTO COMPLETO FROZENLAKE CON TELEMETRÍA DE TIEMPO
# Referencia: Day 3 DQN agent.txt
# ============================================================================
import gymnasium as gym
import time

def ejecutar_experimento_dqn(map_name="4x4", episodios=1500, max_pasos=100, usar_gpu=True):
    dev = torch.device("cuda" if (torch.cuda.is_available() and usar_gpu) else "cpu")
    # Entorno resbaladizo determinista o estocástico (usamos determinista para validación rápida en clase)
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
    
    # Hiperparámetros según Tabla 2 del PDF
    agent = DQNAgent(env, lr=6e-4, gamma=0.93, batch_size=32, mem_capacity=4000,
                     target_update_freq=10, eps_max=0.999, eps_min=0.01, eps_decay=0.999,
                     clip_norm=3.0, dev=dev)
    
    t_inicio = time.time()
    ep_solucion = None
    exitos_consecutivos = 0

    for ep in range(1, episodios + 1):
        s, _ = env.reset()
        s_vec = agent.one_hot_encode(s)
        total_r = 0.0

        for t in range(max_pasos):
            agent.total_steps += 1
            a = agent.select_action(s_vec)
            s_sig, r, term, trunc, _ = env.step(a)
            done = term or trunc
            s_sig_vec = agent.one_hot_encode(s_sig)

            agent.memory.store(s_vec, a, s_sig_vec, r, done)
            agent.learn()

            s_vec = s_sig_vec
            total_r += r
            if done:
                break

        agent.reward_history.append(total_r)
        agent.update_epsilon()

        # Condición de resolución: 10 episodios consecutivos con éxito
        if total_r > 0:
            exitos_consecutivos += 1
            if exitos_consecutivos >= 10 and ep_solucion is None:
                ep_solucion = ep - 10
        else:
            exitos_consecutivos = 0

    t_total = time.time() - t_inicio
    print(f"[{map_name.upper()} | {dev.type.upper()}] Tiempo Total: {t_total:.2f}s | Episodio de Solución: {ep_solucion} | Éxito Reciente: {sum(agent.reward_history[-50:]):.0f}/50")
    return t_total, ep_solucion

# Demostración en vivo en clase
print("--- Iniciando Benchmarking Oficial de Laboratorio ---")
t_4_cpu, ep_4_cpu = ejecutar_experimento_dqn(map_name="4x4", episodios=800, usar_gpu=False)
t_4_gpu, ep_4_gpu = ejecutar_experimento_dqn(map_name="4x4", episodios=800, usar_gpu=True)
```

---

### 1.2 Dinámica Hands-On y Llenado de Tablas de Laboratorio

#### TABLA 1: Rendimiento Comparativo Hardware en FrozenLake-v1 (Evidencia Día 3)
*Instrucción:* Los estudiantes deben correr el código en sus instancias de Colab alternando entre entorno de ejecución CPU y GPU (T4/A100) para registrar la latencia y la eficiencia de muestra.

| Entorno / Mapa | Procesador | Tiempo de Ejecución (s) | Episodios para Resolver (K) | Diagnóstico de Rendimiento / Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **FrozenLake 4x4** | CPU | *(Anotar en vivo)* | *(Anotar en vivo)* | Mini-red muy pequeña ($12 \times 8$). El CPU suele ganar por ausencia de latencia de transferencia PCIe. |
| **FrozenLake 4x4** | GPU | *(Anotar en vivo)* | *(Anotar en vivo)* | Overhead de lanzamiento de kernels CUDA domina sobre el cálculo tensor. |
| **FrozenLake 8x8** | CPU | *(Anotar en vivo)* | *(Anotar en vivo)* | Espacio de 64 dimensiones. Mayor tiempo de inferencia acumulado. |
| **FrozenLake 8x8** | GPU | *(Anotar en vivo)* | *(Anotar en vivo)* | La GPU empieza a amortizar el procesamiento vectorial por lotes. |

#### TABLA 2: Estudio de Ablación: ¿Qué Pasa si Quitamos los Pilares de DQN?
*Instrucción:* Modifiquen la clase `DQNAgent` para desactivar individualmente cada componente y observen si el agente logra converger en FrozenLake 4x4 en 800 episodios.

| Experimento de Ablación | Modificación en Código | ¿Converge a la Meta? (SÍ/NO) | Estabilidad de Pérdida ($L_{\text{MSE}}$) | Causa Mecatrónica / Matemática |
| :--- | :--- | :--- | :--- | :--- |
| **A: Control Sano (DQN Nominal)** | Parámetros base: $C=4000, T_{\text{up}}=10, c=3$. | SÍ | Convergencia suave a cero. | Estimador insesgado y estable. |
| **B: Sin Red Objetivo ($T_{\text{up}}=1$)** | Actualizar `target_net` cada paso ($T_{\text{up}}=1$). | NO | Oscilaciones violentas. | Persecución de objetivo no estacionario; divergencia de punto fijo de Bellman. |
| **C: Sin Replay Memory ($B=1$)** | Entrenar únicamente con la última transición observada. | NO | Divergencia / Ruido estocástico alto. | Pérdida de la propiedad i.i.d.; sobreajuste catastrófico al último paso. |
| **D: Sin Gradient Clipping ($c=\infty$)** | Comentar la línea `clip_grad_norm_`. | NO | Picos repentinos de NaN o $\infty$. | Pasos de descenso de gradiente demasiado grandes expulsan los pesos del mínimo local. |

--- 


### 1.3 Cuestionario de Preparación Pre-Examen (DQN)

1. **¿Por qué la codificación de estado en FrozenLake se ingresa a la red neuronal como un vector One-Hot de longitud $N$ en lugar de ingresar el número entero escalar $s \in [0, N-1]$?**  
2. **Demuestre analíticamente el TD Target si el estado siguiente $s_{j}'$ es terminal ($d_j = 1$).**
3. **¿Cuál es la función matemática del parámetro $\text{fan\_in}$ en la inicialización uniforme de He?**  
