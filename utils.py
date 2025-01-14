import random
import re
import time
import os
import gc
import torch
import logging
from transformers import AutoTokenizer, AutoModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributions.utils import _standard_normal
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs
from torch import distributions as pyd
# Initialize BERT model and tokenizer at module level
_tokenizer = None
_bert_model = None

def get_bert_model_and_tokenizer():
    """Lazy initialization of BERT model and tokenizer"""
    global _tokenizer, _bert_model
    if _tokenizer is None:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        _bert_model = AutoModel.from_pretrained("bert-base-cased")
    return _tokenizer, _bert_model

def encode_task(task_name: str, max_word_len: int = 77, device: str = 'cpu') -> torch.Tensor:
    """Encode task name using BERT
    
    Args:
        task_name (str): Name of the task to encode
        max_word_len (int): Maximum length for tokenization
        device (str): Device to put the model on
        
    Returns:
        torch.Tensor: Task embedding of shape (1, 768)
    """
    tokenizer, bert_model = get_bert_model_and_tokenizer()
    
    tokens = tokenizer(
        text=task_name,
        add_special_tokens=True,
        max_length=max_word_len,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    # Move to specified device
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    bert_model = bert_model.to(device)
    
    # Get BERT embedding
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask)
        task_emb = outputs["pooler_output"]  # Use pooled output for [CLS] token
        
    return task_emb  # Shape: (1, 768)

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# def to_torch(xs, device):
#     return tuple(torch.as_tensor(x, device=device) for x in xs)
def to_torch(xs, device):
    for key, value in xs.items():
        xs[key] = torch.as_tensor(value, device=device)
    return xs


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()
        # Keep track of evaluation time so that total time only includes train time
        self._eval_start_time = 0
        self._eval_time = 0
        self._eval_flag = False

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time - self._eval_time
        return elapsed_time, total_time

    def eval(self):
        if not self._eval_flag:
            self._eval_flag = True
            self._eval_start_time = time.time()
        else:
            self._eval_time += time.time() - self._eval_start_time
            self._eval_flag = False
            self._eval_start_time = 0

    def total_time(self):
        return time.time() - self._start_time - self._eval_time

    def get_elapsed_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """

    # num_channels: num_groups
    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(
                        GROUP_NORM_LOOKUP[num_channels], num_channels
                    )
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def evaluate_multitask_training_success(cfg,benchmark, task_ids, model ):
    """
    Evaluate the success rate for all task in task_ids.
    """
    
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = encode_task(task_i.language)
        success_rate = evaluate_one_task_success(cfg, model, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)

def evaluate_one_task_success(
    cfg, model,task, task_emb, task_id, sim_states=None, task_str=""
):
    """
    Evaluate a single task's success rate
    sim_states: if not None, will keep track of all simulated states during
                evaluation, mainly for visualization and debugging purpose
    task_str:   the key to access sim_states dictionary
    """
    with Timer() as t:

        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # Try to handle the frame buffer issue
        env_creation = False

        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in range(eval_loop_num):
            env.reset()
            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            model.reset()
            obs = env.set_init_state(init_states_)

            # dummy actions [env_num, 7] all zeros for initial physics simulation
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)
            # print(f"Obs shape: {obs}")
            if task_str != "":
                sim_state = env.get_sim_state()
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                        sim_states[i * env_num + k].append(sim_state[k])

            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                if cfg.model_type == "transformer":
                    actions = model.get_action(data)
                else:
                    actions = model.get_action(data,steps)

                obs, reward, done, info = env.step(actions)

                # record the sim states for replay purpose
                if task_str != "":
                    sim_state = env.get_sim_state()
                    for k in range(env_num):
                        if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                            sim_states[i * env_num + k].append(sim_state[k])

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]

                if all(dones):
                    break

            # a new form of success record
            for k in range(env_num):
                if i * env_num + k < cfg.eval.n_eval:
                    num_success += int(dones[k])

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate

def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
    Prepare the tensor observations as input for the algorithm.
    Convert raw environment observations to tensor format matching the dataset structure.
    For transformer model: Uses LIBERO's raw_obs_to_tensor_obs from metric.py
    For baku model: Uses the original implementation
    """
    if cfg.model_type == "transformer":
        env_num = len(obs)
        device = cfg.device
        
        # Initialize the data structure
        data = {
            "obs": {},
            "task_emb": task_emb.repeat(env_num, 1).to(device),
        }
        
        # Define key mapping from environment to model
        key_mapping = {
            "agentview_image": "agentview_rgb",
            "robot0_eye_in_hand_image": "eye_in_hand_rgb",
            "robot0_gripper_qpos": "gripper_states",
            "robot0_joint_pos": "joint_states"
        }
        
        # Process each observation key with mapping
        for k in range(env_num):
            for env_key, model_key in key_mapping.items():
                if model_key not in data["obs"]:
                    data["obs"][model_key] = []
                data["obs"][model_key].append(torch.from_numpy(obs[k][env_key]).float())
        
        # Stack all observations
        for model_key in data["obs"].keys():
            if "rgb" in model_key:  # RGB images need to be normalized and permuted
                # Stack and move to device
                stacked = torch.stack(data["obs"][model_key]).to(device)
                # Normalize to [0, 1]
                stacked = stacked / 255.0
                # Permute from [B, H, W, C] to [B, C, H, W]
                data["obs"][model_key] = stacked.permute(0, 3, 1, 2)
            else:  # Other observations (proprioceptive) don't need normalization
                data["obs"][model_key] = torch.stack(data["obs"][model_key]).to(device)
        
        return data
    
    # Original implementation for baku model
    env_num = len(obs)
    device = cfg.device

    # Initialize lists to store batched observations
    agentview_images = []
    hand_images = []
    proprio_states = []

    # Process each environment's observations
    for k in range(env_num):
        # Get RGB images
        agentview_img = torch.from_numpy(obs[k]['agentview_image']).float() / 255.0  # Normalize to [0,1]
        hand_img = torch.from_numpy(obs[k]['robot0_eye_in_hand_image']).float() / 255.0
        
        # Get proprioceptive states (gripper + joint states)
        gripper_state = torch.from_numpy(obs[k]['robot0_gripper_qpos']).float()
        joint_state = torch.from_numpy(obs[k]['robot0_joint_pos']).float()
        proprio = torch.cat([gripper_state, joint_state], dim=-1)

        # Append to lists
        agentview_images.append(agentview_img)
        hand_images.append(hand_img)
        proprio_states.append(proprio)

    # Stack all observations into batches
    data = {
        "pixels": torch.stack(agentview_images).to(device),  # [B, H, W, C]
        "pixels_egocentric": torch.stack(hand_images).to(device),  # [B, H, W, C]
        "proprioceptive": torch.stack(proprio_states).to(device),  # [B, D]
        "task_emb": task_emb.repeat(env_num, 1).to(device)  # [B, E]
    }

    # Permute image dimensions from [B, H, W, C] to [B, C, H, W] for PyTorch
    data["pixels"] = data["pixels"].permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, C, H, W]
    data["pixels_egocentric"] = data["pixels_egocentric"].permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, C, H, W]
    data["proprioceptive"] = data["proprioceptive"].unsqueeze(1)  # [B, 1, D]

    return data