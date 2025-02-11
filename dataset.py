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
        pad_frame_stack: bool = True,  # Whether to pad for frame stacking
        pad_seq_length: bool = True,  # Whether to pad for sequence length
        get_pad_mask: bool = False,  # Whether to return padding masks
        get_action_padding: bool = False  # Whether to return action padding masks
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
        """
        super().__init__()
        
        if benchmark not in self.BENCHMARKS:
            raise ValueError(f"Benchmark must be one of {list(self.BENCHMARKS.keys())}")
            
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
                    for demo_key in f['data'].keys():
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
                        while current_start < n_timesteps:
                            actual_length = min(self.seq_length, n_timesteps - current_start)
                            is_padded = actual_length < self.seq_length
                            
                            if not self.pad_seq_length and is_padded:
                                break
                                
                            self.segment_map.append((
                                file_path, 
                                demo_key, 
                                task_name, 
                                current_start, 
                                is_padded, 
                                actual_length
                            ))
                            
                            # Move to next sequence start, considering overlap
                            current_start += (self.seq_length - self.overlap)
                        
                # Pre-compute task embedding if needed
                if self.load_task_emb and task_name not in self.task_embeddings:
                    self.task_embeddings[task_name] = encode_task(task_name, self.max_word_len, self.device)
                        
        print(f"Loaded {len(self.segment_map)} segments from {len(self.task_files)} tasks")
        for task_name, files in self.task_files.items():
            n_demos = sum(1 for f in files for _ in h5py.File(f, 'r')['data'].keys())
            print(f"Task: {task_name} - {n_demos} demonstrations")
        
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
           │      ├── joint_angles    (dataset of shape (T, 9))
           │      └── actions         (dataset of shape (T, 7))
           ├── demo_01
           │      ├── images
           │      │      ├── cam_0
           │      │      └── cam_1
           │      ├── joint_angles
           │      └── actions
           └── ...
    
    The dataset segments each demonstration into sequences of length `seq_length`,
    optionally with frame stacking padding and overlap (if not full-length).
    
    The following modalities are returned (as torch.Tensors):
      - For each camera defined in camera_mapping, images are loaded, normalized to [0,1],
        resized to `image_shape`, and permuted to (T, C, H, W).
      - "proprioceptive": loaded from "joint_angles" (T, 9) and normalized using linear normalization:
             (angle - 180)/180, mapping the original range [0, 360] to [-1, 1].
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
        load_task_emb: bool = True,                # NEW: whether to encode task embedding from HDF5 filename
        max_word_len: int = 25                     # NEW: maximum word length for task encoding
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
            image_shape (tuple): NEW: The target image size (height, width). Default is (128, 128).
            action_velocity_scale (float): Scale for joint velocities normalization.
            gripper_scale (float): Scale for gripper velocity normalization.
            load_task_emb (bool): Whether to encode task embedding from HDF5 filename.
            max_word_len (int): Maximum word length for task encoding.
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
        self.load_task_emb = load_task_emb
        self.max_word_len = max_word_len
        if self.load_task_emb:
            self.task_embeddings = {}
        
        # Define which cameras to load.
        if camera_mapping is None:
            # By default, return two camera feeds if available.
            self.camera_mapping = {"pixels": "cam_0", "pixels_egocentric": "cam_1"}
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
                self.task_files[task_name] = file_path
                if self.load_task_emb and task_name not in self.task_embeddings:
                    self.task_embeddings[task_name] = encode_task(task_name, self.max_word_len, self.device)
                
                # Open file to gather segments from each demo.
                with h5py.File(file_path, 'r') as f:
                    for demo_key in f.keys():
                        if not demo_key.startswith('demo_'):
                            continue
                        demo = f[demo_key]
                        if "actions" not in demo:
                            continue
                        n_timesteps = demo["actions"].shape[0]
                        
                        # Determine offset based on frame stacking
                        start_offset = 0 if self.pad_frame_stack else (self.frame_stack - 1)
                        remaining_length = n_timesteps - start_offset
                        if not self.pad_seq_length:
                            remaining_length -= (self.seq_length - 1)
                        if remaining_length <= 0:
                            continue
                        current_start = start_offset
                        while current_start < n_timesteps:
                            actual_length = min(self.seq_length, n_timesteps - current_start)
                            is_padded = (actual_length < self.seq_length)
                            if not self.pad_seq_length and is_padded:
                                break
                            self.segment_map.append((
                                file_path,
                                demo_key,
                                task_name,
                                current_start,
                                is_padded,
                                actual_length
                            ))
                            current_start += (self.seq_length - self.overlap)
        
        print(f"Loaded {len(self.segment_map)} segments from {len(self.task_files)} tasks")
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
            
            # NEW: Resize each frame to the desired resize_shape (e.g., 128x128)
            if self.image_shape is not None:
                H_target, W_target = self.image_shape
                # Create an empty container for resized images.
                resized_array = np.empty((self.seq_length, H_target, W_target, C), dtype=img_array.dtype)
                for i in range(self.seq_length):
                    # Note: cv2.resize expects (width, height)
                    resized_array[i] = cv2.resize(img_array[i], (W_target, H_target))
                img_array = resized_array
                # Optionally update H and W if needed (not strictly required here)
                H, W = H_target, W_target

            # Normalize to [0,1], convert to float32, and permute to (T, C, H, W).
            img_array = img_array.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(0, 3, 1, 2)
            data['images'][out_key] = img_tensor
        
        # Process proprioceptive data from joint_angles.
        if "joint_angles" not in demo:
            raise ValueError(f"Demo {demo_key} in file {file_path} does not contain 'joint_angles'")
        ja_ds = demo["joint_angles"]
        T_ja, feat_dim = ja_ds.shape
        ja_array = np.zeros((self.seq_length, feat_dim), dtype=ja_ds.dtype)
        ja_array[:actual_length] = ja_ds[start_idx:start_idx + actual_length]
        frame_stack_pad = 0
        if self.pad_frame_stack and start_idx < (self.frame_stack - 1):
            frame_stack_pad = self.frame_stack - 1 - start_idx
            for i in range(frame_stack_pad):
                ja_array[i] = ja_array[frame_stack_pad]
        
        # NEW: Normalize proprioceptive joint angles
        # Apply linear normalization: (angle - 180)/180. This maps 0-360 to [-1, 1].
        ja_array = ja_array.astype(np.float32)
        ja_array = (ja_array - 180.0) / 180.0
        
        data["proprioceptive"] = torch.from_numpy(ja_array)
        
        # Process actions.
        if "actions" not in demo:
            raise ValueError(f"Demo {demo_key} in file {file_path} does not contain 'actions'")
        act_ds = demo["actions"]
        T_act, act_dim = act_ds.shape
        act_array = np.zeros((self.seq_length, act_dim), dtype=act_ds.dtype)
        act_array[:actual_length] = act_ds[start_idx:start_idx + actual_length]
        
        # NEW: Normalize actions
        # For the first 6 elements (joint velocities in [-30, 30]), divide by action_velocity_scale.
        # For the 7th element (gripper velocity in [-3000, 3000]), divide by gripper_scale.
        act_array = act_array.astype(np.float32)
        norm_factors = np.array([self.action_velocity_scale]*6 + [self.gripper_scale], dtype=np.float32)
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
        
        return data

    def close(self):
        """Close all open HDF5 files in the cache."""
        for f in self.file_cache.values():
            f.close()
        self.file_cache.clear()
    
    def __del__(self):
        self.close() 


if __name__ == "__main__":
    # Test RealDataset and visualize a resized image sequence as a video
    kinova_dataset = Kinova_Dataset(
        data_path="/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/sim_env/Kinova_gen2/data",
        seq_length=10,
        frame_stack=4,
        overlap=2,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        get_action_padding=True,
        num_queries=10,
        image_shape=(128, 128),  # Target image size: 128x128
        action_velocity_scale=30.0,
        gripper_scale=3000.0
    )
    print(f"Kinova_Dataset loaded {len(kinova_dataset)} segments from {len(kinova_dataset.task_files)} tasks.")
    sample = kinova_dataset[0]
    # print(kinova_dataset[0])
    # try:
    #     sample = real_dataset[0]
    # except Exception as e:
    #     print("Failed to retrieve a segment:", e)
    #     real_dataset.close()
    #     exit(1)
    
    # Print shapes in the sample for verification.
    # print("\n--- Sample Segment Details ---")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape:", value.shape)
        elif isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"  {sub_key} shape:", sub_value.shape)
    
    # # Visualize the first image using matplotlib
    # try:
    #     import matplotlib.pyplot as plt
    #     if "images" in sample and "pixels" in sample["images"]:
    #         img_tensor = sample["images"]["pixels"][0]
    #         img_np = img_tensor.permute(1, 2, 0).numpy()
    #         plt.figure(figsize=(4, 4))
    #         plt.imshow(img_np)
    #         plt.title("Resized Image from RealDataset (128x128)")
    #         plt.axis("off")
    #         plt.show()
    # except ImportError:
    #     print("matplotlib not installed; skipping static image visualization.")
    
    # # -------------------------------
    # # New: Visualize the entire sequence as a video using OpenCV.
    # try:
    #     video_window = "RealDataset Video Playback"
    #     cv2.namedWindow(video_window, cv2.WINDOW_NORMAL)
    #     num_frames = sample["images"]["pixels"].shape[0]
    #     print(f"\nDisplaying video: {num_frames} frames (press 'q' to quit)")
    #     for i in range(num_frames):
    #         # Each frame tensor has shape: (C, H, W)
    #         frame_tensor = sample["images"]["pixels_egocentric"][i]
    #         # Convert tensor to numpy array (H, W, C)
    #         frame_np = frame_tensor.permute(1, 2, 0).numpy()
    #         # Scale back to [0,255] for display
    #         frame_disp = (frame_np * 255).astype(np.uint8)
    #         cv2.imshow(video_window, frame_disp)
    #         # Wait 300ms between frames; press 'q' to exit early.
    #         if cv2.waitKey(300) & 0xFF == ord('q'):
    #             break
    #     cv2.destroyWindow(video_window)
    # except Exception as e:
    #     print("Error during video playback:", e)
    
    # # Clean up by closing open HDF5 files
    # real_dataset.close()
    # print("\nRealDataset closed successfully")
    
