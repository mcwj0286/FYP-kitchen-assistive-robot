from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
import os
from typing import List, Dict, Optional, Tuple
from utils import encode_task
import logging
import cv2

class LIBERODataset(Dataset):
    """Dataset for loading LIBERO demonstrations"""
    
    BENCHMARKS = {
        'libero_90': 'training set with 90% of the data',
        'libero_10': 'validation set with 10% of the data',
        'libero_spatial': 'test set for spatial generalization',
        'libero_goal': 'test set for goal generalization',
        'libero_object': 'test set for object generalization'
    }
    
    def __init__(
        self,
        data_path: str,
        benchmark: str = 'libero_90',
        device: str = 'cpu',
        load_task_emb: bool = True,
        num_queries: int = 10,  # Number of future actions to concatenate
        max_word_len: int = 77,  # Maximum length for tokenization
        seq_length: int = 10,  # Length of each sequence segment
        frame_stack: int = 1,  # Number of frames to stack
        overlap: int = 0,  # Number of overlapping timestamps between sequences
        pad_frame_stack: bool = False,  # Whether to pad for frame stacking
        pad_seq_length: bool = False,  # Whether to pad for sequence length
        get_pad_mask: bool = False,  # Whether to return padding masks
        get_action_padding: bool = False,  # Whether to return action padding masks
        train_ratio: float = 0.8,  # Ratio of demonstrations to use for training
        is_train: bool = True  # Whether this is a training or validation dataset
    ):
        """
        Args:
            data_path (str): Base path to LIBERO dataset
            benchmark (str): Which benchmark to load
            device (str): Device to load tensors to
            load_task_emb (bool): Whether to load task embeddings
            num_queries (int): Number of future actions to concatenate
            max_word_len (int): Maximum length for tokenization
            seq_length (int): Length of each sequence segment
            frame_stack (int): Number of frames to stack
            overlap (int): Number of overlapping timestamps between sequences
            pad_frame_stack (bool): Whether to pad for frame stacking
            pad_seq_length (bool): Whether to pad for sequence length
            get_pad_mask (bool): Whether to return padding masks
            get_action_padding (bool): Whether to return action padding masks
            train_ratio (float): Ratio of demonstrations to use for training (0.0 to 1.0)
            is_train (bool): Whether this dataset instance is for training (True) or validation (False)
        """
        super().__init__()
        
        if benchmark not in self.BENCHMARKS:
            raise ValueError(f"Benchmark must be one of {list(self.BENCHMARKS.keys())}")
        
        if not (0.0 <= train_ratio <= 1.0):
            raise ValueError(f"train_ratio must be between 0.0 and 1.0, got {train_ratio}")
            
        self.data_path = os.path.join(data_path, benchmark)
        self.device = 'cpu'  # Force CPU for dataset
        self.load_task_emb = load_task_emb
        self.num_queries = num_queries
        self.max_word_len = max_word_len
        self.seq_length = seq_length
        self.frame_stack = frame_stack
        self.overlap = min(overlap, seq_length - 1)  # Ensure overlap is less than sequence length
        self.pad_frame_stack = pad_frame_stack
        self.pad_seq_length = pad_seq_length
        self.get_pad_mask = get_pad_mask
        self.get_action_padding = get_action_padding
        self.train_ratio = train_ratio
        self.is_train = is_train
            
        # Load all HDF5 files and organize by tasks
        self.task_files = {}  # Maps task_name to file path
        self.task_embeddings = {}  # Cache for task embeddings
        self.segment_map = []  # Maps index to (file_path, demo_key, task_name, start_idx, is_padded, actual_length)
        
        for file in os.listdir(self.data_path):
            if file.endswith('.hdf5'):
                # Extract task name from filename
                task_name = self._get_task_name_from_filename(file)
                file_path = os.path.join(self.data_path, file)
                
                # Store file path for this task
                if task_name not in self.task_files:
                    self.task_files[task_name] = []
                self.task_files[task_name].append(file_path)
                
                # Map demonstrations in this file and create segments
                with h5py.File(file_path, 'r') as f:
                    # Collect all valid demo keys
                    valid_demos = []
                    for demo_key in f['data'].keys():
                        if 'actions' not in f['data'][demo_key]:
                            continue
                        valid_demos.append(demo_key)
                    
                    # Split demos into train and validation sets
                    n_demos = len(valid_demos)
                    n_train = int(n_demos * train_ratio)
                    
                    # Sort to ensure consistent splits across runs
                    valid_demos.sort()
                    
                    # Select the appropriate demo keys based on is_train flag
                    if is_train:
                        selected_demos = valid_demos[:n_train]
                    else:
                        selected_demos = valid_demos[n_train:]
                    
                    print(f"Task {task_name}: {len(selected_demos)}/{n_demos} demos selected for {'training' if is_train else 'validation'}")
                    
                    # Process only the selected demonstrations
                    for demo_key in selected_demos:
                        # Get number of timesteps in this demo
                        n_timesteps = len(f['data'][demo_key]['actions'])
                        
                        # Determine start index offset based on frame stacking
                        start_offset = 0 if self.pad_frame_stack else (self.frame_stack - 1)
                        
                        # Determine effective sequence length considering overlap
                        effective_length = self.seq_length - self.overlap
                        
                        # Calculate number of sequences with overlap
                        remaining_length = n_timesteps - start_offset
                        if not self.pad_seq_length:
                            remaining_length -= (self.seq_length - 1)
                        
                        if remaining_length <= 0:
                            # Skip if demo is too short
                            continue
                            
                        # Create sequences with overlap
                        current_start = start_offset
                        processed_segments = []
                        
                        while current_start + self.seq_length <= n_timesteps:
                            # This is a full-length segment
                            segment = (
                                file_path, 
                                demo_key, 
                                task_name, 
                                current_start, 
                                False,  # is_padded
                                self.seq_length
                            )
                            processed_segments.append(segment)
                            self.segment_map.append(segment)
                            
                            # Move to next sequence start, considering overlap
                            current_start += (self.seq_length - self.overlap)
                        
                        # Handle remaining frames when pad_seq_length is False
                        remaining_frames = n_timesteps - current_start
                        if not self.pad_seq_length and remaining_frames > 0:
                            # Create a segment that starts earlier to maintain seq_length
                            # This will overlap with previously processed frames
                            overlap_start = n_timesteps - self.seq_length
                            
                            # Add the overlapping segment
                            segment = (
                                file_path,
                                demo_key,
                                task_name,
                                overlap_start,
                                False,  # is_padded
                                self.seq_length
                            )
                            self.segment_map.append(segment)
                        elif self.pad_seq_length and remaining_frames > 0:
                            # Original padding logic
                            is_padded = True
                            self.segment_map.append((
                                file_path,
                                demo_key,
                                task_name,
                                current_start,
                                is_padded,
                                remaining_frames
                            ))
                
                # Pre-compute task embedding if needed
                if self.load_task_emb and task_name not in self.task_embeddings:
                    self.task_embeddings[task_name] = encode_task(task_name, self.max_word_len, self.device)
                        
        split_type = "training" if is_train else "validation"
        print(f"Loaded {len(self.segment_map)} {split_type} segments from {len(self.task_files)} tasks")
        for task_name, files in self.task_files.items():
            n_demos = sum(1 for f in files for _ in h5py.File(f, 'r')['data'].keys())
            print(f"Task: {task_name} - {n_demos} demonstrations available")
        
        # Cache for open HDF5 files
        self.file_cache = {}
        
    def _get_task_name_from_filename(self, filename: str) -> str:
        """Extract task name from filename by removing underscores and 'demo.hdf5'"""
        # Remove 'demo.hdf5' and split by underscores
        task_name = filename.replace('_demo.hdf5', '')
        # Replace underscores with spaces
        task_name = task_name.replace('_', ' ')
        return task_name
    
    def _get_file(self, file_path: str) -> h5py.File:
        """Get HDF5 file handle, using cache to avoid reopening"""
        if file_path not in self.file_cache:
            # Clear cache if too many files open
            if len(self.file_cache) > 5:
                for f in self.file_cache.values():
                    f.close()
                self.file_cache.clear()
            
            self.file_cache[file_path] = h5py.File(file_path, 'r')
        return self.file_cache[file_path]
    
    def _process_actions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process actions to include future actions and create padding mask
        
        Args:
            actions: Original actions array of shape (T, 7)
            
        Returns:
            tuple: (processed_actions, action_padding_mask) where:
                - processed_actions: shape (T, 7 * num_queries) where each timestep
                  includes the current and future actions, padded with zeros if needed
                - action_padding_mask: shape (T, num_queries) where True indicates valid
                  actions and False indicates padded actions
        """
        T, action_dim = actions.shape
        total_dim = action_dim * self.num_queries
        
        # Initialize output arrays with zeros
        processed_actions = np.zeros((T, total_dim), dtype=actions.dtype)
        action_padding_mask = np.zeros((T, self.num_queries), dtype=bool)
        
        # For each timestep, concatenate the next num_queries actions
        for t in range(T):
            future_actions = actions[t:t + self.num_queries]  # Get next actions (might be less than num_queries)
            actual_futures = len(future_actions)
            
            # Place the available future actions in the correct position
            processed_actions[t, :actual_futures * action_dim] = future_actions.flatten()
            
            # Mark which future actions are valid (not padded)
            action_padding_mask[t, :actual_futures] = True
            
        return processed_actions, action_padding_mask
    
    def __len__(self) -> int:
        return len(self.segment_map)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns:
            dict: Dictionary containing:
                - pixels: Agent view RGB images (T, 3, 128, 128) float32 tensor normalized to [0,1]
                - pixels_egocentric: Eye-in-hand RGB images (T, 3, 128, 128) float32 tensor normalized to [0,1] 
                - proprioceptive: Robot state features (T, 9) float32 tensor
                - actions: Ground truth actions (T, 7*num_queries) float32 tensor
                - padding_mask: Binary mask indicating padded timesteps (T,) bool tensor (if get_pad_mask=True)
                - task_emb: Task embedding (768,) float32 tensor (if load_task_emb=True)
                - action_padding_mask: Binary mask indicating padded actions (T, num_queries) bool tensor (if get_action_padding=True)
                
            where T is the sequence length. Padding mask is True for padded timesteps
            and False for valid timesteps.
        """
        """Get a sequence segment from a demonstration trajectory"""
        file_path, demo_key, task_name, start_idx, is_padded, actual_length = self.segment_map[idx]
        f = self._get_file(file_path)
        
        demo = f['data'][demo_key]
        
        # Initialize data with zeros (handles padding automatically)
        data = {
            "pixels": np.zeros((self.seq_length, 128, 128, 3), dtype=np.float32),
            "pixels_egocentric": np.zeros((self.seq_length, 128, 128, 3), dtype=np.float32),
            "proprioceptive": np.zeros((self.seq_length, 9), dtype=np.float32)
        }
        
        # Calculate frame stacking padding
        frame_stack_pad = 0
        if self.pad_frame_stack and start_idx < (self.frame_stack - 1):
            frame_stack_pad = self.frame_stack - 1 - start_idx
        
        # Load actual data
        data["pixels"][:actual_length] = demo['obs']['agentview_rgb'][start_idx:start_idx + actual_length]
        data["pixels_egocentric"][:actual_length] = demo['obs']['eye_in_hand_rgb'][start_idx:start_idx + actual_length]
        data["proprioceptive"][:actual_length] = np.concatenate([
            demo['obs']['gripper_states'][start_idx:start_idx + actual_length],
            demo['obs']['joint_states'][start_idx:start_idx + actual_length]
        ], axis=-1)
        
        # Handle frame stacking padding by repeating first frame
        if frame_stack_pad > 0:
            for k in data:
                data[k][:frame_stack_pad] = data[k][frame_stack_pad:frame_stack_pad+1].repeat(frame_stack_pad, axis=0)
        
        # Process actions
        actions = np.zeros((self.seq_length, 7), dtype=np.float32)
        actions[:actual_length] = demo['actions'][start_idx:start_idx + actual_length]
        if self.get_action_padding:
            gt_actions, action_padding_mask = self._process_actions(actions)
        else:
            gt_actions = self._process_actions(actions)[0]  # Only get the actions, ignore the mask
        
        # Convert images to float and normalize
        data["pixels"] = data["pixels"].astype(np.float32) / 255.0
        data["pixels_egocentric"] = data["pixels_egocentric"].astype(np.float32) / 255.0
        
        # Convert to tensors (keep on CPU)
        for k, v in data.items():
            data[k] = torch.from_numpy(v)
            if k in ["pixels", "pixels_egocentric"]:
                data[k] = data[k].permute(0, 3, 1, 2)
            
        # Add task embedding if requested
        if self.load_task_emb:
            data["task_emb"] = self.task_embeddings[task_name]
            
        # Add padding mask if requested
        if self.get_pad_mask:
            pad_mask = torch.zeros(self.seq_length, 1, dtype=torch.bool)
            pad_mask[frame_stack_pad:actual_length] = 1
            data["pad_mask"] = pad_mask
        
        data["actions"] = torch.from_numpy(gt_actions)
        if self.get_action_padding:
            data["action_padding_mask"] = torch.from_numpy(action_padding_mask)
        
        return data

    def close(self):
        """Close all open HDF5 files"""
        for f in self.file_cache.values():
            f.close()
        self.file_cache.clear()
        
    def __del__(self):
        self.close()


class Kinova_Dataset(Dataset):
    """
    Dataset for loading real-world demonstrations collected using the robotic arm.
    
    Expected HDF5 file structure for each task (one file per task):
      └── <task_name>.hdf5
           ├── demo_00
           │      ├── images
           │      │      ├── cam_0
           │      │      └── cam_1
           │      ├── joint_angles    (dataset of shape (T, 9)) - When in 'joint' control mode
           │      ├── cartesian_pose  (dataset of shape (T, 7)) - When in 'cartesian' control mode
           │      └── actions         (dataset of shape (T, 7))
           ├── demo_01
           │      ├── images
           │      │      ├── cam_0
           │      │      └── cam_1
           │      ├── joint_angles    or cartesian_pose
           │      └── actions
           └── ...
    
    The dataset segments each demonstration into sequences of length `seq_length`,
    optionally with frame stacking padding and overlap (if not full-length).
    
    The following modalities are returned (as torch.Tensors):
      - For each camera defined in camera_mapping, images are loaded, normalized to [0,1],
        resized to `image_shape`, and permuted to (T, C, H, W).
      - "proprioceptive": 
         * In 'joint' mode: loaded from "joint_angles" (T, 9) and normalized using linear normalization:
             (angle - 180)/180, mapping the original range [0, 360] to [-1, 1].
         * In 'cartesian' mode: loaded from "cartesian_pose" (T, 7):
             - position XYZ: normalized to [-1, 1] 
             - orientation XYZW: quaternion values (already normalized)
      - "actions": processed actions with future concatenation if requested (T, 7*num_queries),
             where the first 6 elements (joint velocities, originally in [-30, 30]) are normalized by dividing by 30,
             and the 7th element (gripper velocity, originally in [-3000, 3000]) is normalized by dividing by 3000.
      - "pad_mask": (optional) binary mask of valid timesteps (T, 1)
      - "action_padding_mask": (optional) binary mask for padded future actions.
    """
    def __init__(
        self,
        data_path: str,
        seq_length: int = 10,
        frame_stack: int = 1,
        overlap: int = 0,
        pad_frame_stack: bool = True,
        pad_seq_length: bool = True,
        get_pad_mask: bool = False,
        get_action_padding: bool = False,
        num_queries: int = 10,
        camera_mapping: Optional[Dict[str, str]] = None,  # mapping output key -> camera dataset key in HDF5 group
        image_shape: Tuple[int, int] = (128, 128),  # desired output image size (height, width)
        action_velocity_scale: float = 30.0,       # scale for joint velocities normalization
        gripper_scale: float = 3000.0,             # scale for gripper velocity normalization
        linear_velocity_scale: float = 0.2,        # scale for cartesian linear velocity normalization
        angular_velocity_scale: float = 40.0,      # scale for cartesian angular velocity normalization
        load_task_emb: bool = True,                # whether to encode task embedding from HDF5 filename
        max_word_len: int = 25,                    # maximum word length for task encoding
        train_ratio: float = 0.8,                  # ratio of demonstrations to use for training
        is_train: bool = True,                     # whether this is a training or validation dataset
        control_mode: str = 'joint'                # control mode: 'joint' or 'cartesian'
    ):
        """
        Args:
            data_path (str): Directory containing the task HDF5 files.
            seq_length (int): Desired sequence length.
            frame_stack (int): Frames to stack (affects starting index).
            overlap (int): Overlap between consecutive sequences.
            pad_frame_stack (bool): Whether to pad for frame stacking (repeating the first frame).
            pad_seq_length (bool): Whether to pad sequences to full length.
            get_pad_mask (bool): Whether to return a padding mask.
            get_action_padding (bool): Whether to return a mask for padded future actions.
            num_queries (int): Number of future actions to concatenate.
            camera_mapping (dict): Mapping of output image key to dataset key within the "images" group.
                                   Default is {"pixels": "cam_0", "pixels_egocentric": "cam_1"}.
            image_shape (tuple): The target image size (height, width). Default is (128, 128).
            action_velocity_scale (float): Scale for joint velocities normalization (for joint control mode).
            gripper_scale (float): Scale for gripper velocity normalization.
            linear_velocity_scale (float): Scale for cartesian linear velocity (for cartesian control mode).
            angular_velocity_scale (float): Scale for cartesian angular velocity (for cartesian control mode).
            load_task_emb (bool): Whether to encode task embedding from HDF5 filename.
            max_word_len (int): Maximum word length for task encoding.
            train_ratio (float): Ratio of demonstrations to use for training (0.0 to 1.0).
            is_train (bool): Whether this dataset instance is for training (True) or validation (False).
            control_mode (str): Control mode used for data recording: 'joint' or 'cartesian'.
        """
        self.data_path = data_path
        self.seq_length = seq_length
        self.frame_stack = frame_stack
        self.overlap = min(overlap, seq_length - 1)
        self.pad_frame_stack = pad_frame_stack
        self.pad_seq_length = pad_seq_length
        self.get_pad_mask = get_pad_mask
        self.get_action_padding = get_action_padding
        self.num_queries = num_queries
        self.device = "cpu"  # Force CPU
        self.image_shape = image_shape
        self.action_velocity_scale = action_velocity_scale
        self.gripper_scale = gripper_scale
        self.linear_velocity_scale = linear_velocity_scale
        self.angular_velocity_scale = angular_velocity_scale
        self.load_task_emb = load_task_emb
        self.max_word_len = max_word_len
        self.train_ratio = train_ratio
        self.is_train = is_train
        self.control_mode = control_mode
        
        # Validate control mode
        if self.control_mode not in ['joint', 'cartesian']:
            raise ValueError(f"control_mode must be 'joint' or 'cartesian', got {self.control_mode}")
        
        if not (0.0 <= train_ratio <= 1.0):
            raise ValueError(f"train_ratio must be between 0.0 and 1.0, got {train_ratio}")
        
        if self.load_task_emb:
            self.task_embeddings = {}
        
        # Define which cameras to load.
        if camera_mapping is None:
            # By default, return two camera feeds if available.
            self.camera_mapping = {"pixels": "cam_0", "pixels_egocentric": "cam_2"}
        else:
            self.camera_mapping = camera_mapping
        
        # Dictionary mapping task_name to file path.
        self.task_files = {}
        # List of segments: (file_path, demo_key, task_name, start_idx, is_padded, actual_length)
        self.segment_map = []
        
        # Loop through all HDF5 files in data_path.
        for file in os.listdir(self.data_path):
            if file.endswith('.hdf5'):
                file_path = os.path.join(self.data_path, file)
                # Use the filename (without extension) as the task name.
                task_name = os.path.splitext(file)[0]
                task_name = task_name.replace('_', ' ')
                self.task_files[task_name] = file_path
                if self.load_task_emb and task_name not in self.task_embeddings:
                    self.task_embeddings[task_name] = encode_task(task_name, self.max_word_len, self.device)
                
                # Open file to gather demos and split into train/val sets
                with h5py.File(file_path, 'r') as f:
                    # Collect all valid demo keys
                    valid_demos = []
                    for demo_key in f.keys():
                        if not demo_key.startswith('demo_'):
                            continue
                        demo = f[demo_key]
                        
                        # Check if data has the required format based on control mode
                        if self.control_mode == 'joint' and "joint_angles" not in demo:
                            continue
                        elif self.control_mode == 'cartesian' and "cartesian_pose" not in demo:
                            continue
                        if "actions" not in demo:
                            continue
                            
                        valid_demos.append(demo_key)
                    
                    # Split demos into train and validation sets
                    n_demos = len(valid_demos)
                    n_train = int(n_demos * train_ratio)
                    
                    # Sort to ensure consistent splits across runs
                    valid_demos.sort()
                    
                    # Select the appropriate demo keys based on is_train flag
                    if is_train:
                        selected_demos = valid_demos[:n_train]
                    else:
                        selected_demos = valid_demos[n_train:]
                    
                    print(f"Task {task_name}: {len(selected_demos)}/{n_demos} demos selected for {'training' if is_train else 'validation'}")
                    
                    # Process the selected demonstrations
                    for demo_key in selected_demos:
                        demo = f[demo_key]
                        
                        # Get appropriate dataset based on control mode
                        if self.control_mode == 'joint':
                            state_dataset = demo["joint_angles"]
                        else:  # cartesian mode
                            state_dataset = demo["cartesian_pose"]
                            
                        n_timesteps = state_dataset.shape[0]
                        
                        # Determine offset based on frame stacking
                        start_offset = 0 if self.pad_frame_stack else (self.frame_stack - 1)
                        remaining_length = n_timesteps - start_offset
                        if not self.pad_seq_length:
                            remaining_length -= (self.seq_length - 1)
                        if remaining_length <= 0:
                            continue
                        current_start = start_offset
                        processed_segments = []
                        
                        while current_start + self.seq_length <= n_timesteps:
                            # This is a full-length segment
                            segment = (
                                file_path,
                                demo_key,
                                task_name,
                                current_start,
                                False,  # is_padded
                                self.seq_length
                            )
                            processed_segments.append(segment)
                            self.segment_map.append(segment)
                            
                            # Move to next sequence start, considering overlap
                            current_start += (self.seq_length - self.overlap)
                        
                        # Handle remaining frames
                        remaining_frames = n_timesteps - current_start
                        if not self.pad_seq_length and remaining_frames > 0:
                            # Create a segment that starts earlier to maintain seq_length
                            # This will overlap with previously processed frames
                            overlap_start = n_timesteps - self.seq_length
                            
                            # Add the overlapping segment
                            segment = (
                                file_path,
                                demo_key,
                                task_name,
                                overlap_start,
                                False,  # is_padded
                                self.seq_length
                            )
                            self.segment_map.append(segment)
                        elif self.pad_seq_length and remaining_frames > 0:
                            # Original padding logic
                            is_padded = True
                            self.segment_map.append((
                                file_path,
                                demo_key,
                                task_name,
                                current_start,
                                is_padded,
                                remaining_frames
                            ))
        
        split_type = "training" if is_train else "validation"
        print(f"Loaded {len(self.segment_map)} {split_type} segments from {len(self.task_files)} tasks in {control_mode} mode")
        self.file_cache = {}
    
    def _get_file(self, file_path: str) -> h5py.File:
        """Get HDF5 file handle (using cache to avoid reopening files)"""
        if file_path not in self.file_cache:
            if len(self.file_cache) > 5:
                for f in self.file_cache.values():
                    f.close()
                self.file_cache.clear()
            self.file_cache[file_path] = h5py.File(file_path, 'r')
        return self.file_cache[file_path]
    
    def _process_actions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process actions by concatenating the next num_queries actions at every timestep.
        Returns tuple of (processed_actions, action_padding_mask)
        """
        T, action_dim = actions.shape
        total_dim = action_dim * self.num_queries
        processed_actions = np.zeros((T, total_dim), dtype=actions.dtype)
        action_padding_mask = np.zeros((T, self.num_queries), dtype=bool)
        for t in range(T):
            future_actions = actions[t:t + self.num_queries]
            actual_futures = len(future_actions)
            processed_actions[t, :actual_futures * action_dim] = future_actions.flatten()
            action_padding_mask[t, :actual_futures] = True
        return processed_actions, action_padding_mask

    def __len__(self) -> int:
        return len(self.segment_map)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path, demo_key, task_name, start_idx, is_padded, actual_length = self.segment_map[idx]
        f = self._get_file(file_path)
        demo = f[demo_key]
        data = {}
        
        # Process images from each camera specified in camera_mapping.
        data['images'] = {}
        if 'images' not in demo:
            raise ValueError(f"Demo {demo_key} in file {file_path} does not contain an 'images' group")
        images_group = demo['images']
        for out_key, cam_key in self.camera_mapping.items():
            if cam_key not in images_group:
                raise ValueError(f"Camera {cam_key} not found in demo {demo_key} in file {file_path}")
            cam_ds = images_group[cam_key]
            T_img, H, W, C = cam_ds.shape
            img_array = np.zeros((self.seq_length, H, W, C), dtype=cam_ds.dtype)
            end_idx = start_idx + actual_length
            img_array[:actual_length] = cam_ds[start_idx:end_idx]
            # Apply frame stacking padding if needed.
            frame_stack_pad = 0
            if self.pad_frame_stack and start_idx < (self.frame_stack - 1):
                frame_stack_pad = self.frame_stack - 1 - start_idx
                for i in range(frame_stack_pad):
                    img_array[i] = img_array[frame_stack_pad]
            
            # Resize each frame to the desired resize_shape
            if self.image_shape is not None:
                H_target, W_target = self.image_shape
                # Create an empty container for resized images.
                resized_array = np.empty((self.seq_length, H_target, W_target, C), dtype=img_array.dtype)
                for i in range(self.seq_length):
                    #change to rgb
                    image_rgb = cv2.cvtColor(img_array[i], cv2.COLOR_BGR2RGB)
                    # Note: cv2.resize expects (width, height)
                    resized_array[i] = cv2.resize(image_rgb, (W_target, H_target))
                img_array = resized_array
                # Optionally update H and W if needed (not strictly required here)
                H, W = H_target, W_target

            # Normalize to [0,1], convert to float32, and permute to (T, C, H, W).
            img_array = img_array.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(0, 3, 1, 2)
            data['images'][out_key] = img_tensor
        
        # Process proprioceptive data based on control mode
        if self.control_mode == 'joint':
            # Process joint angles
            if "joint_angles" not in demo:
                raise ValueError(f"Demo {demo_key} in file {file_path} does not contain 'joint_angles'")
            ja_ds = demo["joint_angles"]
            T_ja, feat_dim = ja_ds.shape
            ja_array = np.zeros((self.seq_length, feat_dim), dtype=ja_ds.dtype)
            ja_array[:actual_length] = ja_ds[start_idx:start_idx + actual_length]
            
            # Apply frame stack padding if needed
            frame_stack_pad = 0
            if self.pad_frame_stack and start_idx < (self.frame_stack - 1):
                frame_stack_pad = self.frame_stack - 1 - start_idx
                for i in range(frame_stack_pad):
                    ja_array[i] = ja_array[frame_stack_pad]
            
            # Normalize joint angles from [0, 360] to [-1, 1]
            ja_array = ja_array.astype(np.float32)
            ja_array = (ja_array - 180.0) / 180.0
            data["proprioceptive"] = torch.from_numpy(ja_array)
        else:
            # Process cartesian pose
            if "cartesian_pose" not in demo:
                raise ValueError(f"Demo {demo_key} in file {file_path} does not contain 'cartesian_pose'")
            cp_ds = demo["cartesian_pose"]
            T_cp, feat_dim = cp_ds.shape
            cp_array = np.zeros((self.seq_length, feat_dim), dtype=cp_ds.dtype)
            cp_array[:actual_length] = cp_ds[start_idx:start_idx + actual_length]
            
            # Apply frame stack padding if needed
            frame_stack_pad = 0
            if self.pad_frame_stack and start_idx < (self.frame_stack - 1):
                frame_stack_pad = self.frame_stack - 1 - start_idx
                for i in range(frame_stack_pad):
                    cp_array[i] = cp_array[frame_stack_pad]
            
            # Normalize cartesian pose
            # - Position (XYZ): first 3 values
            # - Orientation (XYZ): next 3 values (Euler angles in radians)
            # - Gripper: last value
            cp_array = cp_array.astype(np.float32)
            
            # Apply normalization using our static method
            cp_array = Kinova_Dataset.normalize_cartesian_pose(
                cp_array, 
                position_scale=1.0, 
                gripper_min=-6.0, 
                gripper_max=7020.0
            )
            
            data["proprioceptive"] = torch.from_numpy(cp_array)
        
        # Process actions.
        if "actions" not in demo:
            raise ValueError(f"Demo {demo_key} in file {file_path} does not contain 'actions'")
        act_ds = demo["actions"]
        T_act, act_dim = act_ds.shape
        act_array = np.zeros((self.seq_length, act_dim), dtype=act_ds.dtype)
        act_array[:actual_length] = act_ds[start_idx:start_idx + actual_length]
        
        # Normalize actions based on control mode
        act_array = act_array.astype(np.float32)
        if self.control_mode == 'joint':
            # For joint control, all joint velocities use the same scale (30.0)
            norm_factors = np.array([self.action_velocity_scale]*6 + [self.gripper_scale], dtype=np.float32)
        else:
            # For cartesian control, action is (linear vel x,y,z, angular vel x,y,z, gripper)
            norm_factors = np.array(
                [self.linear_velocity_scale] * 3 +  # Linear velocity (x, y, z)
                [self.angular_velocity_scale] * 3 +  # Angular velocity (x, y, z) 
                [self.gripper_scale],           # Gripper velocity
                dtype=np.float32
            )
        
        # Apply normalization
        act_array = act_array / norm_factors
        
        if self.get_action_padding:
            proc_actions, action_padding_mask = self._process_actions(act_array)
        else:
            proc_actions = self._process_actions(act_array)[0]
        data["actions"] = torch.from_numpy(proc_actions)
        if self.get_action_padding:
            data["action_padding_mask"] = torch.from_numpy(action_padding_mask)
        
        # Add padding mask if requested.
        if self.get_pad_mask:
            pad_mask = torch.zeros(self.seq_length, 1, dtype=torch.bool)
            valid_start = 0 if self.pad_frame_stack else (self.frame_stack - 1)
            pad_mask[valid_start:actual_length] = True
            data["pad_mask"] = pad_mask
        
        # Add task embedding to the returned data if enabled.
        if self.load_task_emb:
            data["task_emb"] = self.task_embeddings[task_name]
        
        # Flatten observations to match LIBERODataset format
        data["pixels"] = data["images"].get("pixels")
        data["pixels_egocentric"] = data["images"].get("pixels_egocentric")
        del data["images"]
        
        return data

    def close(self):
        """Close all open HDF5 files in the cache."""
        for f in self.file_cache.values():
            f.close()
        self.file_cache.clear()
    
    def __del__(self):
        self.close()

    @staticmethod
    def analyze_data_ranges(data_path: str, verbose: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Analyzes the min/max values for proprioceptive and action data across all demonstrations.
        
        Args:
            data_path (str): Path to the directory containing HDF5 files
            verbose (bool): Whether to print analysis results
            
        Returns:
            Dict containing statistics for each data type:
            {
                'joint_angles': {
                    'min': array of min values per dimension,
                    'max': array of max values per dimension,
                    'mean': array of mean values per dimension,
                    'std': array of standard deviation values per dimension
                },
                'cartesian_pose': { ... same structure ... },
                'actions': { ... same structure ... }
            }
        """
        # Initialize storage for min/max values
        stats = {
            'joint_angles': {
                'min': None, 'max': None, 
                'mean': None, 'std': None,
                'count': 0
            },
            'cartesian_pose': {
                'min': None, 'max': None, 
                'mean': None, 'std': None,
                'count': 0
            },
            'actions': {
                'min': None, 'max': None, 
                'mean': None, 'std': None,
                'count': 0
            }
        }
        
        # Running sum for mean calculation
        sums = {
            'joint_angles': None,
            'cartesian_pose': None,
            'actions': None
        }
        
        # Running sum of squares for std calculation
        sum_squares = {
            'joint_angles': None,
            'cartesian_pose': None, 
            'actions': None
        }
        
        # Collect all files
        num_files = 0
        for file in os.listdir(data_path):
            if not file.endswith('.hdf5'):
                continue
                
            num_files += 1
            file_path = os.path.join(data_path, file)
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # Process each demo in the file
                    for demo_key in f.keys():
                        if not demo_key.startswith('demo_'):
                            continue
                            
                        demo = f[demo_key]
                        
                        # Process joint angles if available
                        if 'joint_angles' in demo:
                            data = demo['joint_angles'][:]
                            stats['joint_angles']['count'] += len(data)
                            
                            if stats['joint_angles']['min'] is None:
                                stats['joint_angles']['min'] = np.min(data, axis=0)
                                stats['joint_angles']['max'] = np.max(data, axis=0)
                                sums['joint_angles'] = np.sum(data, axis=0)
                                sum_squares['joint_angles'] = np.sum(data**2, axis=0)
                            else:
                                stats['joint_angles']['min'] = np.minimum(stats['joint_angles']['min'], np.min(data, axis=0))
                                stats['joint_angles']['max'] = np.maximum(stats['joint_angles']['max'], np.max(data, axis=0))
                                sums['joint_angles'] += np.sum(data, axis=0)
                                sum_squares['joint_angles'] += np.sum(data**2, axis=0)
                        
                        # Process cartesian pose if available
                        if 'cartesian_pose' in demo:
                            data = demo['cartesian_pose'][:]
                            stats['cartesian_pose']['count'] += len(data)
                            
                            if stats['cartesian_pose']['min'] is None:
                                stats['cartesian_pose']['min'] = np.min(data, axis=0)
                                stats['cartesian_pose']['max'] = np.max(data, axis=0)
                                sums['cartesian_pose'] = np.sum(data, axis=0)
                                sum_squares['cartesian_pose'] = np.sum(data**2, axis=0)
                            else:
                                stats['cartesian_pose']['min'] = np.minimum(stats['cartesian_pose']['min'], np.min(data, axis=0))
                                stats['cartesian_pose']['max'] = np.maximum(stats['cartesian_pose']['max'], np.max(data, axis=0))
                                sums['cartesian_pose'] += np.sum(data, axis=0)
                                sum_squares['cartesian_pose'] += np.sum(data**2, axis=0)
                        
                        # Process actions if available
                        if 'actions' in demo:
                            data = demo['actions'][:]
                            stats['actions']['count'] += len(data)
                            
                            if stats['actions']['min'] is None:
                                stats['actions']['min'] = np.min(data, axis=0)
                                stats['actions']['max'] = np.max(data, axis=0)
                                sums['actions'] = np.sum(data, axis=0)
                                sum_squares['actions'] = np.sum(data**2, axis=0)
                            else:
                                stats['actions']['min'] = np.minimum(stats['actions']['min'], np.min(data, axis=0))
                                stats['actions']['max'] = np.maximum(stats['actions']['max'], np.max(data, axis=0))
                                sums['actions'] += np.sum(data, axis=0)
                                sum_squares['actions'] += np.sum(data**2, axis=0)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        # Calculate mean and std for each data type
        for data_type in stats:
            if stats[data_type]['count'] > 0:
                stats[data_type]['mean'] = sums[data_type] / stats[data_type]['count']
                # Calculate standard deviation: sqrt(E[X^2] - E[X]^2)
                mean_squares = sum_squares[data_type] / stats[data_type]['count']
                stats[data_type]['std'] = np.sqrt(mean_squares - stats[data_type]['mean']**2)
        
        # Print results if verbose
        if verbose:
            print(f"Analyzed {num_files} files")
            
            # Print joint angles stats
            if stats['joint_angles']['count'] > 0:
                print("\n=== Joint Angles Statistics (9 dimensions) ===")
                print(f"Total samples: {stats['joint_angles']['count']}")
                print("Min values:")
                for i, val in enumerate(stats['joint_angles']['min']):
                    print(f"  Dim {i}: {val:.4f}")
                print("Max values:")
                for i, val in enumerate(stats['joint_angles']['max']):
                    print(f"  Dim {i}: {val:.4f}")
                print("Mean values:")
                for i, val in enumerate(stats['joint_angles']['mean']):
                    print(f"  Dim {i}: {val:.4f}")
                print("Std values:")
                for i, val in enumerate(stats['joint_angles']['std']):
                    print(f"  Dim {i}: {val:.4f}")
                    
                # Normalized range for joint angles
                print("\nJoint Angles Normalized Range (min-180)/180 to (max-180)/180:")
                for i in range(len(stats['joint_angles']['min'])):
                    min_norm = (stats['joint_angles']['min'][i] - 180.0) / 180.0
                    max_norm = (stats['joint_angles']['max'][i] - 180.0) / 180.0
                    print(f"  Dim {i}: [{min_norm:.4f}, {max_norm:.4f}]")
            
            # Print cartesian pose stats
            if stats['cartesian_pose']['count'] > 0:
                print("\n=== Cartesian Pose Statistics (7 dimensions) ===")
                print(f"Total samples: {stats['cartesian_pose']['count']}")
                print("Min values:")
                for i, val in enumerate(stats['cartesian_pose']['min']):
                    print(f"  Dim {i}: {val:.4f}" + (" (position)" if i < 3 else " (gripper)" if i == 6 else " (orientation)"))
                print("Max values:")
                for i, val in enumerate(stats['cartesian_pose']['max']):
                    print(f"  Dim {i}: {val:.4f}" + (" (position)" if i < 3 else " (gripper)" if i == 6 else " (orientation)"))
                print("Mean values:")
                for i, val in enumerate(stats['cartesian_pose']['mean']):
                    print(f"  Dim {i}: {val:.4f}" + (" (position)" if i < 3 else " (gripper)" if i == 6 else " (orientation)"))
                print("Std values:")
                for i, val in enumerate(stats['cartesian_pose']['std']):
                    print(f"  Dim {i}: {val:.4f}" + (" (position)" if i < 3 else " (gripper)" if i == 6 else " (orientation)"))
                
                # Range for position values after normalization
                print("\nCartesian Pose Normalized Range:")
                # Position values normalized by 1.0
                for i in range(3):
                    min_norm = stats['cartesian_pose']['min'][i] / 1.0
                    max_norm = stats['cartesian_pose']['max'][i] / 1.0
                    print(f"  Dim {i}: [{min_norm:.4f}, {max_norm:.4f}] (position/1.0)")
                
                # Orientation values normalized by dividing by pi
                for i in range(3, 6):
                    min_norm = stats['cartesian_pose']['min'][i] / np.pi
                    max_norm = stats['cartesian_pose']['max'][i] / np.pi
                    print(f"  Dim {i}: [{min_norm:.4f}, {max_norm:.4f}] (orientation/π)")
                
                # Gripper value normalized by min-max scaling
                i = 6
                gripper_min, gripper_max = -6.0, 7020.0
                min_norm = 2.0 * (stats['cartesian_pose']['min'][i] - gripper_min) / (gripper_max - gripper_min) - 1.0
                max_norm = 2.0 * (stats['cartesian_pose']['max'][i] - gripper_min) / (gripper_max - gripper_min) - 1.0
                print(f"  Dim {i}: [{min_norm:.4f}, {max_norm:.4f}] (gripper, min-max scaled from {gripper_min} to {gripper_max})")
            
            # Print actions stats
            if stats['actions']['count'] > 0:
                print("\n=== Actions Statistics (7 dimensions) ===")
                print(f"Total samples: {stats['actions']['count']}")
                print("Min values:")
                for i, val in enumerate(stats['actions']['min']):
                    print(f"  Dim {i}: {val:.4f}" + (" (joint velocity)" if i < 6 else " (gripper)"))
                print("Max values:")
                for i, val in enumerate(stats['actions']['max']):
                    print(f"  Dim {i}: {val:.4f}" + (" (joint velocity)" if i < 6 else " (gripper)"))
                print("Mean values:")
                for i, val in enumerate(stats['actions']['mean']):
                    print(f"  Dim {i}: {val:.4f}" + (" (joint velocity)" if i < 6 else " (gripper)"))
                print("Std values:")
                for i, val in enumerate(stats['actions']['std']):
                    print(f"  Dim {i}: {val:.4f}" + (" (joint velocity)" if i < 6 else " (gripper)"))
                
                # Range for normalized values
                print("\nActions Normalized Range:")
                
                # Check if actions appear to be cartesian or joint velocities
                # If action dimensions 0-2 have similar ranges around ±0.2 and dims 3-5 around ±40, assume cartesian
                is_cartesian = (
                    abs(stats['actions']['max'][0]) <= 0.3 and
                    abs(stats['actions']['max'][1]) <= 0.3 and
                    abs(stats['actions']['max'][2]) <= 0.3 and
                    abs(stats['actions']['max'][3]) >= 10.0 and 
                    abs(stats['actions']['max'][4]) >= 10.0 and
                    abs(stats['actions']['max'][5]) >= 10.0
                )
                
                if is_cartesian:
                    print("Detected cartesian velocity actions")
                    # For cartesian control mode
                    for i in range(7):
                        if i < 3:  # Linear velocities (x,y,z)
                            scale = 0.2  # Linear velocity scale
                            component = "linear velocity"
                        elif i < 6:  # Angular velocities (x,y,z)
                            scale = 40.0  # Angular velocity scale
                            component = "angular velocity"
                        else:  # Gripper
                            scale = 3000.0  # Gripper scale
                            component = "gripper"
                        
                        min_norm = stats['actions']['min'][i] / scale
                        max_norm = stats['actions']['max'][i] / scale
                        print(f"  Dim {i}: [{min_norm:.4f}, {max_norm:.4f}] ({component}, scale={scale})")
                else:
                    print("Detected joint velocity actions")
                    # For joint control mode
                    for i in range(7):
                        if i < 6:  # Joint velocities
                            scale = 30.0  # Joint velocity scale
                            component = "joint velocity"
                        else:  # Gripper
                            scale = 3000.0  # Gripper scale
                            component = "gripper"
                        
                        min_norm = stats['actions']['min'][i] / scale
                        max_norm = stats['actions']['max'][i] / scale
                        print(f"  Dim {i}: [{min_norm:.4f}, {max_norm:.4f}] ({component}, scale={scale})")
        
        # Remove count from returned stats
        for data_type in stats:
            if 'count' in stats[data_type]:
                del stats[data_type]['count']
                
        return stats

    @staticmethod
    def normalize_cartesian_pose(cp_data, position_scale=1.0, gripper_min=-6.0, gripper_max=7020.0):
        """
        Normalize cartesian pose data using the recommended approach:
        - Position (X,Y,Z): Divide by position_scale (typically 1.0 meter)
        - Orientation (Euler angles): Divide by π to normalize to [-1, 1] range
        - Gripper: Min-max scaling to [-1, 1] range using provided min/max values
        
        Args:
            cp_data: numpy array of cartesian pose data with shape (..., 7)
            position_scale: scaling factor for position dimensions
            gripper_min: minimum value for gripper dimension
            gripper_max: maximum value for gripper dimension
        
        Returns:
            Normalized cartesian pose data with same shape as input
        """
        # Create a copy to avoid modifying the original data
        normalized = cp_data.copy().astype(np.float32)
        
        # 1. Position normalization (first 3 dimensions)
        normalized[..., :3] = normalized[..., :3] / position_scale
        
        # 2. Orientation normalization (next 3 dimensions)
        normalized[..., 3:6] = normalized[..., 3:6] / np.pi
        
        # 3. Gripper normalization (last dimension)
        normalized[..., 6] = 2.0 * (normalized[..., 6] - gripper_min) / (gripper_max - gripper_min) - 1.0
        
        return normalized


if __name__ == "__main__":
    # Example usage for analyzing data ranges in cartesian mode
    print("\n=== Analyzing Cartesian Mode Dataset ===")
    cartesian_dataset = Kinova_Dataset(
        data_path="/Volumes/Untitled/fyp/dataset/cartesain_position_dataset",
        seq_length=10,
        frame_stack=1,
        pad_frame_stack=False,
        get_pad_mask=False,
        get_action_padding=False,
        linear_velocity_scale=0.2,      # Cartesian specific parameter
        angular_velocity_scale=40.0,    # Cartesian specific parameter
        control_mode='cartesian'        # Specify cartesian mode
    )
    cartesian_dataset.analyze_data_ranges(
        data_path="/Volumes/Untitled/fyp/dataset/cartesain_position_dataset", 
        verbose=True
    )
    cartesian_dataset.close()
    
    # Example usage for analyzing data ranges in joint mode (if available)
    print("\n=== Analyzing Joint Mode Dataset ===")
    try:
        joint_dataset = Kinova_Dataset(
            data_path="/Users/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/sim_env/Kinova_gen2/data",
            seq_length=10,
            frame_stack=1,
            pad_frame_stack=False,
            get_pad_mask=False,
            get_action_padding=False,
            action_velocity_scale=30.0,  # Joint specific parameter
            control_mode='joint'         # Specify joint mode
        )
        joint_dataset.analyze_data_ranges(
            data_path="/Users/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/sim_env/Kinova_gen2/data", 
            verbose=True
        )
        joint_dataset.close()
    except FileNotFoundError:
        print("Joint mode dataset directory not found. Skipping joint analysis.")
        
    print("\n=== Dataset Analysis Complete ===")
    
    # Example of loading dataset for training with correct parameters
    print("\nExample of initializing dataset for training:")
    print("- For Cartesian mode:")
    print("cartesian_train_dataset = Kinova_Dataset(")
    print("    data_path='path/to/data',")
    print("    control_mode='cartesian',")
    print("    linear_velocity_scale=0.2,")
    print("    angular_velocity_scale=40.0,")
    print("    # other parameters...")
    print(")")
    
    print("\n- For Joint mode:")
    print("joint_train_dataset = Kinova_Dataset(")
    print("    data_path='path/to/data',")
    print("    control_mode='joint',")
    print("    action_velocity_scale=30.0,")
    print("    # other parameters...")
    print(")")