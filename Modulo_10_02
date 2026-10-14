Implementación de Código en Clase (`pendulo_sac_clase.py`)

```python
# ============================================================================
# IMPLEMENTACIÓN OFICIAL EN CLASE: SOFT ACTOR-CRITIC (PENDULUM-V1)
# Control: Estabilización en theta^* = -15 grados
# ============================================================================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from collections import deque
import os
import random
import time
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Configuración de Hardware y Semillas Estocásticas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Hardware de cómputo activo: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 2. Wrapper Mecatrónico: Recompensa Cuadrática hacia Ángulo Arbitrario (-15°)
class PendulumTargetWrapper(gym.Wrapper):

    def __init__(self, env, target_deg=-15.0):
        super().__init__(env)
        self.target_rad = np.radians(target_deg)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cos_th, sin_th, th_dot = obs
        theta = np.arctan2(sin_th, cos_th)

        # Cálculo del error angular normalizado en [-pi, pi]
        error_theta = (theta - self.target_rad + np.pi) % (2 * np.pi) - np.pi
        torque = action[0] if isinstance(action, (np.ndarray, list)) else action

        # Función de costo cuadrático: Error angular + Velocidad + Esfuerzo de control
        custom_reward = -(error_theta**2 + 0.1 * (th_dot**2) + 0.001 * (torque**2))
        return obs, custom_reward, terminated, truncated, info


# 3. Buffer de Repetición (Replay Memory)
class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.FloatTensor(np.array(a)).to(device),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(s2)).to(device),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# 4. Red Actor con Truco de Reparametrización y Acotamiento Tanh
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action=2.0, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.log_std(h)

    def sample(self, state):
        h = self.net(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        # Truco de reparametrización: z = mu + std * epsilon
        z = normal.rsample()
        action = torch.tanh(z)

        # Corrección Jacobiana por saturación Tanh
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action * self.max_action, log_prob


# 5. Red Twin Critic (Clipped Double Q)
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        return self.q1(sa), self.q2(sa)


# 6. Agente Soft Actor-Critic (SAC)
class SAC:

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=2.0,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(
            state_dim, action_dim, max_action, hidden_dim
        ).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = ReplayBuffer(100000)

    def select_action(self, state, eval_mode=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        if eval_mode:
            # Control determinista limpio para despliegue
            with torch.no_grad():
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean) * self.actor.max_action
            return action.cpu().detach().numpy()[0]
        else:
            action, _ = self.actor.sample(state_t)
            return action.cpu().detach().numpy()[0]

    def train(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return

        s, a, r, s2, d = self.buffer.sample(batch_size)

        # Actualización de Críticos con Bellman Target de Máxima Entropía
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1.0 - d) * (
                torch.min(q1_t, q2_t) - self.alpha * logp2
            )

        q1, q2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actualización del Actor
        a_new, logp = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, a_new)
        actor_loss = (self.alpha * logp - torch.min(q1_new, q2_new)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Actualización suave de Polyak sobre los Críticos Objetivo
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


# 7. Bucle Principal de Entrenamiento y Evaluación
def ejecutar_laboratorio():
    raw_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = PendulumTargetWrapper(raw_env, target_deg=-15.0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(state_dim, action_dim, max_action, hidden_dim=256)
    os.makedirs("gifs_clase", exist_ok=True)

    print("\n--- INICIANDO ENTRENAMIENTO SAC (PENDULUM-V1 -> -15 GRADOS) ---")
    episodes = 120
    warmup_steps = 1000  # Pasos iniciales para llenar el búfer con exploración rica
    total_steps = 0

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(200):
            total_steps += 1
            # Exploración inicial pura durante la fase de warmup
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, eval_mode=False)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            agent.buffer.push(state, action, reward, next_state, float(done))

            # Entrenar solo cuando se haya superado el warmup
            if total_steps >= warmup_steps:
                agent.train(batch_size=256)

            state = next_state
            total_reward += reward
            if done:
                break

        if ep % 10 == 0 or ep == episodes:
            print(
                f"[SAC] Episodio {ep:03d}/{episodes} | Retorno Acumulado:"
                f" {total_reward:+.2f}"
            )

    # Protocolo de Inferencia y Generación de GIF en Modo Evaluación
    print("\n[VALIDACIÓN] Generando animación GIF en modo evaluación...")
    frames = []
    eval_state, _ = env.reset(seed=101)

    agent.actor.eval()
    with torch.no_grad():
        for _ in range(200):
            frames.append(env.render())
            eval_action = agent.select_action(eval_state, eval_mode=True)
            eval_state, _, term, trunc, _ = env.step(eval_action)
            if term or trunc:
                break

    gif_path = "gifs_clase/pendulo_target_-15deg.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"[OK] Video GIF generado exitosamente en: {gif_path}")


if __name__ == "__main__":
    ejecutar_laboratorio()
```
