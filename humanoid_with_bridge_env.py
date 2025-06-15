from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.envs.registration import register
from gymnasium import spaces
import os
import numpy as np

class HumanoidOnBridge(HumanoidEnv):
    def __init__(self, xml_file_name: str = "humanoid_with_bridge.xml",lateral_penalty=0.05, k_prog=200.0, k_fwd=5.0, frame_skip=5, **kwargs):
        xml_file = os.path.join(os.path.dirname(__file__), xml_file_name)
        super().__init__(xml_file, frame_skip=frame_skip, **kwargs)
        self.lateral_penalty = lateral_penalty
        self.k_prog = k_prog
        self.k_fwd = k_fwd

        self._jump_threshold = 1.5
        # Tramo del puente
        self.bridge_start  = 0.0
        self.bridge_end    = 5.0
        self.bridge_height = 0.43  # coincide con el XML
        self.slope_angle_up   = +0.174  # 10° en rad
        self.slope_angle_down = -0.174

        # Inicializamos variables de episodio
        self._prev_prog  = 0.0
        self._step_count = 0

        self.push_mag = 0.5
        # Expande observation_space (igual que antes)...
        base = self.observation_space
        bridge_len = self.bridge_end - self.bridge_start
        low_extra = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=base.dtype)
        high_extra = np.array([1.0, bridge_len, 1.0, 1.0, 1.0, 1.0], dtype=base.dtype)
        low = np.concatenate([base.low, low_extra], axis=0)
        high = np.concatenate([base.high, high_extra], axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=base.dtype)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Inicializa _prev_prog y _step_count correctamente
        x0 = float(self.data.qpos[0])
        self._prev_prog = np.clip((x0 - self.bridge_start) / (self.bridge_end - self.bridge_start), 0.0, 1.0)
        self._step_count = 0
        self.data.qvel[0] += self.push_mag
        return obs, info

    def _get_obs(self):
        proprio = super()._get_obs()
        x = float(self.data.qpos[0])
        z = float(self.data.qpos[2])
        prog = np.clip((x - self.bridge_start) / (self.bridge_end - self.bridge_start), 0.0, 1.0)
        dist_goal = abs(self.bridge_end - x)
        dx = self.bridge_end - x
        dz = self.bridge_height - z
        norm = np.hypot(dx, dz) + 1e-8
        ux, uz = dx / norm, dz / norm
        if prog < 0.33:
            slope = self.slope_angle_up
        elif prog < 0.66:
            slope = 0.0
        else:
            slope = self.slope_angle_down
        sin_s, cos_s = np.sin(slope), np.cos(slope)
        extra = np.array([prog, dist_goal, ux, uz, sin_s, cos_s], dtype=proprio.dtype)
        return np.concatenate([proprio, extra], axis=0)

    def step(self, action):
        self._step_count += 1
        obs, reward, terminated, truncated, info = super().step(action)

        # 1) Bonus por Δprog
        prog = obs[-6]
        delta_prog = prog - self._prev_prog
        reward += self.k_prog * delta_prog
        self._prev_prog = prog

        # 2) Reward forward en pendiente
       # ux, uz = obs[-5], obs[-4]
       # vx, vy, vz = self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]
       # reward += self.k_fwd * (ux * vx + uz * vz)

        # 3) Penaliza salto vertical al inicio
        #if self._step_count < 20:
        #    if vz > self._jump_threshold:  # umbral de salto en m/s
        #        reward -= 0.5

        # 4) Región y penalización lateral (igual que antes)
        if prog < 0.33:
            info['bridge_region'] = 'ramp_up'
        elif prog < 0.66:
            info['bridge_region'] = 'bridge_top'
           # reward += 0.5
        else:
            info['bridge_region'] = 'ramp_down'
       # reward -= self.lateral_penalty * abs(vy)

        return obs, reward, terminated, truncated, info


    
# Registro del entorno
register(
    id="HumanoidOnBridge-v0",
    entry_point=__name__ + ":HumanoidOnBridge",
    kwargs={"xml_file_name": "humanoid_with_bridge.xml"},
    max_episode_steps=1000,
    reward_threshold=5000.0,
)
register(
    id="HumanoidOnBridgeSoft-v0",
    entry_point=__name__ + ":HumanoidOnBridge",
    kwargs={"xml_file_name": "humanoid_with_bridge_soft.xml"},
    max_episode_steps=1000,
    reward_threshold=5000.0,
)

