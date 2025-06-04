from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.envs.registration import register
from gymnasium import spaces
import os
import mujoco
import numpy as np

class HumanoidWithWalls(HumanoidEnv):
    def __init__(self,
                 lateral_penalty: float = 0.1,
                 frame_skip: int = 5,
                 **kwargs):
        # Carga tu XML con las paredes
        xml_file = os.path.join(os.path.dirname(__file__),
                                "humanoid_with_walls.xml")
        super().__init__(xml_file, frame_skip=frame_skip, **kwargs)
        self.lateral_penalty = lateral_penalty

        # IDs de las geoms de pared
        self.wall_left_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, b"wall_left")
        self.wall_right_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, b"wall_right")

        # Amplía el espacio de observación para incluir distancias a las paredes
        base_space = self.observation_space
        # Creamos límites adicionales (sin restricciones, infinito)
        low_extra = np.array([-np.inf, -np.inf], dtype=base_space.dtype)
        high_extra = np.array([ np.inf,  np.inf], dtype=base_space.dtype)
        # Concatena low/high originales con extras
        low  = np.concatenate([base_space.low,  low_extra], axis=0)
        high = np.concatenate([base_space.high, high_extra], axis=0)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=base_space.dtype
        )

    def _get_obs(self):
        # Observación proprioceptiva original
        proprio = super()._get_obs()
        # Posición Y del torso
        y = float(self.data.qpos[1])
        # Coordenadas Y de los centros de las geoms de pared
        left_y  = float(self.data.geom_xpos[self.wall_left_id][1])
        right_y = float(self.data.geom_xpos[self.wall_right_id][1])
        # Calcula distancias positivas a cada pared
        dist_left  = left_y  - y
        dist_right = y - right_y
        extra = np.array([dist_left, dist_right], dtype=proprio.dtype)
        # Devuelve proprio + distancias a paredes
        return np.concatenate([proprio, extra], axis=0)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Termina el episodio al colisionar con una pared
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in (self.wall_left_id, self.wall_right_id) or \
               c.geom2 in (self.wall_left_id, self.wall_right_id):
                terminated = True
                info["collision_with_wall"] = True
                break

        # Penalización por deriva lateral (eje Y)
        lateral_vel = info.get("y_velocity", obs[1])
        reward -= self.lateral_penalty * abs(lateral_vel)

        return obs, reward, terminated, truncated, info

# Registro del entorno personalizado
register(
    id="HumanoidWithWalls-v0",
    entry_point=__name__ + ":HumanoidWithWalls",
    max_episode_steps=1000,
    reward_threshold=5000.0,
)
