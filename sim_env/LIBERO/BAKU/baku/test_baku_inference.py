import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from agent.baku import BCAgent  # Assuming BCAgent is defined in agent.baku
from pathlib import Path


obs_shape = {
    "pixels": (3, 128, 128),          # RGB image from an external camera
    "pixels_egocentric": (3,128, 128),  # RGB image from an egocentric (first-person) camera
    "proprioceptive": (9,),        # Proprioceptive state (e.g., joint angles, velocities)
    "features": (128,)              # 128-dimensional feature vector
}
action_shape = (7,)  # 4-dimensional continuous action space

def load_snapshot(agent, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in agent.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        agent.load_snapshot(agent_payload, eval=True)
        return agent

@hydra.main(config_path="./cfgs", config_name="config_eval.yaml")
def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    agent =  hydra.utils.instantiate(cfg.agent)
    snapshots = {}
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    agent = load_snapshot(agent, snapshots)
    return agent

# Define the evaluation function
@hydra.main(config_path="./cfgs", config_name="config_eval.yaml")
def eval_agent(cfg: DictConfig):
    print("Loaded Configuration:")
    print(OmegaConf.to_yaml(cfg))  # Print the full configuration

    # Define the obs_shape and action_shape (replace with actual values)
    # obs_shape = (3, 64, 64)  # Example: (channels, height, width) for pixel input
    # action_shape = (7,)  # Example: number of actions


    # Load model weights from the path specified in the config_eval.yaml
    weights_path = cfg.bc_weight  # e.g., '/path/to/weights/baku.pt'
    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path)
    agent.load_snapshot(checkpoint, eval=True)

    # Example observation (replace with actual observation)
    obs = {
        "pixels": torch.randn(1, *obs_shape["pixels"]),  # Simulating a camera input
        "pixels_egocentric": torch.randn(1, *obs_shape["pixels_egocentric"]),  # Simulating an egocentric camera input
        "proprioceptive": torch.randn(1, *obs_shape["proprioceptive"]),  # Simulating proprioception input
        "features": torch.randn(1, *obs_shape["features"])  # Simulating a feature vector input
    }

    # Perform inference
    print("Running inference...")

    prompt = {'task_emb': torch.randn(384)}
    action = agent.act(obs, prompt=prompt, norm_stats=None, step=0, global_step=0, eval_mode=True)
    print("Predicted Action:", action)

    # Here you could implement saving results, running multiple evaluation episodes, etc.

if __name__ == "__main__":
    agent = make_agent(obs_shape, action_shape)  # Create the agent

    # eval_agent()