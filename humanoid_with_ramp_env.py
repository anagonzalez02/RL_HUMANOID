from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.envs.registration import register
import os
import mujoco

class HumanoidOnRamp(HumanoidEnv):
    def __init__(self,
                 lateral_penalty: float = 0.1,
                 forward_bonus: float = 0.5,
                 backward_penalty: float = 0.2,
                 frame_skip: int = 5,
                 **kwargs):
        # Ruta al XML con rampa inclinada
        xml_file = os.path.join(os.path.dirname(__file__),
                                "humanoid_with_ramp.xml")
        super().__init__(xml_file, frame_skip=frame_skip, **kwargs)
        # Coeficientes para shaping de recompensa
        self.lateral_penalty  = lateral_penalty
        self.forward_bonus    = forward_bonus
        self.backward_penalty = backward_penalty

    def step(self, action):
        # Ejecuta el step del entorno base
        obs, reward, terminated, truncated, info = super().step(action)

        # 1) Penalización de deriva lateral (eje Y)
        lateral_vel = info.get("y_velocity", obs[1])
        reward -= self.lateral_penalty * abs(lateral_vel)

        # 2) Incentivo por avanzar en X (subir pendiente)
        forward_vel = info.get("x_velocity", obs[0])
        if forward_vel > 0:
            reward += self.forward_bonus * forward_vel
        else:
            # Penalización si retrocede
            reward -= self.backward_penalty * (-forward_vel)

        return obs, reward, terminated, truncated, info

# Registro del entorno personalizado
register(
    id="HumanoidOnRamp-v0",
    entry_point=__name__ + ":HumanoidOnRamp",
    max_episode_steps=1000,
    reward_threshold=5000.0,
)
