import gymnasium as gym
import numpy as np
import torch
from lib.agent_ppo import PPOAgent
from lib.utils import log_video
import humanoid_with_bridge_env
from gymnasium.wrappers import RecordVideo


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("HumanoidOnBridge-v0", render_mode="rgb_array", camera_name="overhead")
    #env = gym.make("Humanoid-v5", render_mode="human")
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    agent = PPOAgent(obs_dim[0], action_dim[0]).to(device)
    agent.load_state_dict(torch.load("best.pt"))
    agent.eval()

    env = RecordVideo(
    env,
    video_folder="recordings2",      # carpeta destino
    name_prefix="overhead_walk",    # prefijo de fichero
    episode_trigger=lambda idx: True  # True = graba todos los episodios
    )
    obs, _ = env.reset()
    done = False
    while not done:
        # Render the frame
        env.render()
        # Sample an action
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        # Step the environment
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        #time.sleep(7)
        done = terminated or truncated
    env.close()
