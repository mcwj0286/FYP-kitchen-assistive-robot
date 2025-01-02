import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite,
        scenes,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        intermediate_goal_step=50,
        store_actions=False,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.img_size = img_size
        self.intermediate_goal_step = intermediate_goal_step

        # Temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries
        self.num_demos_per_task = num_demos_per_task

        # Convert task_names, which is a list, to a dictionary
        tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}

        # Get relevant task names
        task_name_list = []
        for scene in scenes:
            task_name_list.extend([t_name for t_name in tasks[scene]])

        # Get data paths
        self._paths = list((Path(path) / suite).glob("*"))

        # Filter paths based on task names
        if task_name_list:
            filtered_paths = {}
            idx2name = {}
            for path in self._paths:
                task = str(path).split(".")[0].split("/")[-1]
                if task in task_name_list:
                    idx = task_name_list.index(task)
                    filtered_paths[idx] = path
                    idx2name[idx] = task
            self._paths = filtered_paths
        else:
            self._paths = {idx: path for idx, path in enumerate(self._paths)}

        # Store actions if required
        self.store_actions = store_actions
        if store_actions:
            self.actions = []

        # Read data to collect metadata
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Processing {str(self._paths[_path_idx])}")
            # Read data
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = data["observations"] if self._obs_type == "pixels" else data["states"]
            actions = data["actions"]
            task_emb = data["task_emb"]
            # Store episode indices and metadata
            self._episodes[_path_idx] = []
         
            for i in range(min(num_demos_per_task, len(observations))):
                # Get episode length and state dimension
                obs_i = observations[i]
                episode_info = dict(
                    observation=None,  # We no longer store full observations here
                    action=None,       # We no longer store full actions here
                    episode_idx=i,
                    file_path=str(self._paths[_path_idx]),
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode_info)

                # Update max_episode_len, max_state_dim, num_samples
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i]["pixels"])
                    ),
                )
                self._max_state_dim = max(
                    self._max_state_dim, data["states"][i].shape[-1]
                )
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i]["pixels"])
                )

                # Store actions if required
                if store_actions:
                    # Cannot store actions[i] here without loading data
                    pass

        # Create a list of environment keys for indexing
        self._env_keys = list(self._episodes.keys())
        self.envs_till_idx = len(self._env_keys)

        # Augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        # Preprocessing statistics
        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
            "proprioceptive": {
                "min": 0,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (
                self.stats["proprioceptive"]["max"]
                - self.stats["proprioceptive"]["min"]
                + 1e-5
            ),
        }

    def _sample_episode(self, env_idx=None):
        if env_idx is None:
            # Randomly select an environment
            env_key = random.choice(self._env_keys)
        else:
            # Map env_idx to env_key
            env_key = self._env_keys[env_idx]

        episodes_in_env = self._episodes[env_key]
        episode_info = random.choice(episodes_in_env)
        return (episode_info, env_key)

    def _sample(self):
        episode_info, env_idx = self._sample_episode()
        episode_idx = episode_info['episode_idx']
        file_path = episode_info['file_path']
        task_emb = episode_info['task_emb']

        # Load data from pkl file
        data = pkl.load(open(file_path, "rb"))
        observations = data["observations"] if self._obs_type == "pixels" else data["states"]
        actions = data["actions"]

        observations = observations[episode_idx]
        actions = actions[episode_idx]


        if self._obs_type == "pixels":
            # Sample obs, action
            max_sample_idx = len(observations["pixels"]) - self._history_len
            if max_sample_idx <= 0:
                # Skip this episode if too short
                return self._sample()
            sample_idx = np.random.randint(0, max_sample_idx)
            sampled_pixel = observations["pixels"][sample_idx : sample_idx + self._history_len]
            sampled_pixel_egocentric = observations["pixels_egocentric"][sample_idx : sample_idx + self._history_len]
            sampled_pixel = torch.stack(
                [self.aug(sampled_pixel[i]) for i in range(len(sampled_pixel))]
            )
            sampled_pixel_egocentric = torch.stack(
                [self.aug(sampled_pixel_egocentric[i]) for i in range(len(sampled_pixel_egocentric))]
            )
            sampled_proprioceptive_state = np.concatenate(
                [
                    observations["joint_states"][sample_idx : sample_idx + self._history_len],
                    observations["gripper_states"][sample_idx : sample_idx + self._history_len],
                ],
                axis=-1,
            )
            if self._temporal_agg:
                # Arrange sampled action to be of shape (history_len, num_queries, action_dim)
                num_actions = self._history_len + self._num_queries - 1
                if sample_idx + num_actions > len(actions):
                    # Not enough actions, skip and resample
                    return self._sample()
                act = actions[sample_idx : sample_idx + num_actions]
                sampled_action = np.lib.stride_tricks.sliding_window_view(
                    act, (self._num_queries, actions.shape[-1])
                )
                sampled_action = sampled_action[:, 0]
            else:
                sampled_action = actions[sample_idx : sample_idx + self._history_len]
# Load data
            # Handle prompts
            if self._prompt == "text":
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](sampled_proprioceptive_state),
                    "actions": self.preprocess["actions"](sampled_action),
                    "task_emb": task_emb,
                }
            elif self._prompt == "goal":
                # Sample a prompt episode from the same task
                prompt_episode_info, _ = self._sample_episode(env_idx)
                prompt_episode_idx = prompt_episode_info['episode_idx']
                prompt_file_path = prompt_episode_info['file_path']  # Correct file path for prompt episode
                # Load data
                data = pkl.load(open(prompt_file_path, "rb"))
                prompt_observations = data["observations"][prompt_episode_idx]
                prompt_actions = data["actions"][prompt_episode_idx]

                prompt_pixel = self.aug(prompt_observations["pixels"][-1])[None]
                prompt_pixel_egocentric = self.aug(prompt_observations["pixels_egocentric"][-1])[None]
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["joint_states"][-1:],
                        prompt_observations["gripper_states"][-1:],
                    ],
                    axis=-1,
                )
                prompt_action = prompt_actions[-1:]
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](sampled_proprioceptive_state),
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_pixels": prompt_pixel,
                    "prompt_pixels_egocentric": prompt_pixel_egocentric,
                    "prompt_proprioceptive": self.preprocess["proprioceptive"](prompt_proprioceptive_state),
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                }
            elif self._prompt == "intermediate_goal":
                # Use the same episode
                prompt_observations = observations
                prompt_actions = actions
                intermediate_goal_step = self.intermediate_goal_step + np.random.randint(-30, 30)
                goal_idx = min(sample_idx + intermediate_goal_step, len(prompt_observations["pixels"]) - 1)
                prompt_pixel = self.aug(prompt_observations["pixels"][goal_idx])[None]
                prompt_pixel_egocentric = self.aug(prompt_observations["pixels_egocentric"][goal_idx])[None]
                prompt_proprioceptive_state = np.concatenate(
                    [
                        prompt_observations["joint_states"][goal_idx : goal_idx + 1],
                        prompt_observations["gripper_states"][goal_idx : goal_idx + 1],
                    ],
                    axis=-1,
                )
                prompt_action = prompt_actions[goal_idx : goal_idx + 1]
                return {
                    "pixels": sampled_pixel,
                    "pixels_egocentric": sampled_pixel_egocentric,
                    "proprioceptive": self.preprocess["proprioceptive"](sampled_proprioceptive_state),
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_pixels": prompt_pixel,
                    "prompt_pixels_egocentric": prompt_pixel_egocentric,
                    "prompt_proprioceptive": self.preprocess["proprioceptive"](prompt_proprioceptive_state),
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                }
        elif self._obs_type == "features":
            # Sample obs, action
            max_sample_idx = len(observations) - self._history_len
            if max_sample_idx <= 0:
                # Skip this episode if it's too short
                return self._sample()
            sample_idx = np.random.randint(0, max_sample_idx)
            sampled_obs = np.array(observations[sample_idx : sample_idx + self._history_len])
            sampled_action = actions[sample_idx : sample_idx + self._history_len]
            # Pad obs to match self._max_state_dim
            obs = np.zeros((self._history_len, self._max_state_dim))
            state_dim = sampled_obs.shape[-1]
            obs[:, :state_dim] = sampled_obs
            sampled_obs = obs

            # Handle prompts
            if self._prompt == "text":
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "task_emb": task_emb,
                }
            elif self._prompt == "goal":
                # Sample a prompt episode from the same task
                prompt_episode_info, _ = self._sample_episode(env_idx)
                prompt_episode_idx = prompt_episode_info['episode_idx']
                prompt_file_path = prompt_episode_info['file_path']  # Correct file path for prompt episode
                # Load data
                data = pkl.load(open(prompt_file_path, "rb"))  # Use the correct file path
                prompt_observations = data["observations"][prompt_episode_idx]
                prompt_actions = data["actions"][prompt_episode_idx]

                prompt_obs = np.array(prompt_observations[-1:])
                prompt_action = prompt_actions[-1:]
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_obs": prompt_obs,
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                }
            elif self._prompt == "intermediate_goal":
                prompt_observations = observations
                prompt_actions = actions
                goal_idx = min(sample_idx + self.intermediate_goal_step, len(prompt_observations) - 1)
                prompt_obs = np.array(prompt_observations[goal_idx : goal_idx + 1])
                prompt_action = prompt_actions[goal_idx : goal_idx + 1]
                return {
                    "features": sampled_obs,
                    "actions": self.preprocess["actions"](sampled_action),
                    "prompt_obs": prompt_obs,
                    "prompt_actions": self.preprocess["actions"](prompt_action),
                    "task_emb": task_emb,
                }

        # Store actions if required
        if self.store_actions:
            self.actions.append(actions)

    def sample_test(self, env_idx, step=None):
        # Map env_idx to env_key
        if env_idx >= len(self._env_keys):
            raise ValueError(f"env_idx {env_idx} out of range.")
        env_key = self._env_keys[env_idx]
        episode_list = self._episodes[env_key]
        if not episode_list:
            raise ValueError(f"No episodes found for env_idx {env_idx}")
        episode_info = random.choice(episode_list)
        episode_idx = episode_info['episode_idx']
        file_path = episode_info['file_path']
        task_emb = episode_info['task_emb']

        data = pkl.load(open(file_path, "rb"))
        observations = data["observations"][episode_idx]
        actions = data["actions"][episode_idx]

        if self._obs_type == "pixels":
            # Observation
            if self._prompt == None or self._prompt == "text":
                prompt_pixel = None
                prompt_pixel_egocentric = None
                prompt_proprioceptive_state = None
                prompt_action = None
            elif self._prompt == "goal":
                prompt_pixel = np.transpose(observations["pixels"][-1:], (0, 3, 1, 2))
                prompt_pixel_egocentric = np.transpose(
                    observations["pixels_egocentric"][-1:], (0, 3, 1, 2)
                )
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["joint_states"][-1:],
                        observations["gripper_states"][-1:],
                    ],
                    axis=-1,
                )
                prompt_action = None
            elif self._prompt == "intermediate_goal":
                if step is None:
                    step = 0
                goal_idx = min(
                    step + self.intermediate_goal_step, len(observations["pixels"]) - 1
                )
                prompt_pixel = np.transpose(
                    observations["pixels"][goal_idx : goal_idx + 1], (0, 3, 1, 2)
                )
                prompt_pixel_egocentric = np.transpose(
                    observations["pixels_egocentric"][goal_idx : goal_idx + 1],
                    (0, 3, 1, 2),
                )
                prompt_proprioceptive_state = np.concatenate(
                    [
                        observations["joint_states"][goal_idx : goal_idx + 1],
                        observations["gripper_states"][goal_idx : goal_idx + 1],
                    ],
                    axis=-1,
                )
                prompt_action = None

            return {
                "prompt_pixels": prompt_pixel,
                "prompt_pixels_egocentric": prompt_pixel_egocentric,
                "prompt_proprioceptive": (
                    self.preprocess["proprioceptive"](prompt_proprioceptive_state)
                    if prompt_proprioceptive_state is not None
                    else None
                ),
                "prompt_actions": (
                    self.preprocess["actions"](prompt_action)
                    if prompt_action is not None
                    else None
                ),
                "task_emb": task_emb,
            }

        elif self._obs_type == "features":
            # Observation
            if self._prompt == None or self._prompt == "text":
                prompt_obs, prompt_action = None, None
            elif self._prompt == "goal":
                prompt_obs = np.array(observations[-1:])
                prompt_action = None

            return {
                "prompt_features": prompt_obs,
                "prompt_actions": self.preprocess["actions"](prompt_action),
                "task_emb": task_emb,
            }

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples