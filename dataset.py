from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
import os
from typing import List, Dict, Optional
from utils import encode_task
import logging

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

if __name__ == "__main__":
    # Example usage and validation
    base_path = "/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO/libero/datasets"
    
    print("\n=== Testing Different Dataset Configurations ===")
    
    # Test Case 1: Basic configuration with no overlap
    print("\nTest Case 1: Basic Configuration (No Overlap)")
    dataset_basic = LIBERODataset(
        data_path=base_path,
        benchmark='libero_spatial',
        seq_length=10,
        frame_stack=1,
        overlap=0,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        get_action_padding=True
    )
    print(f"Number of segments (no overlap): {len(dataset_basic)}")
    
    # Test Case 2: With overlap
    print("\nTest Case 2: With Overlap (3 timesteps)")
    dataset_overlap = LIBERODataset(
        data_path=base_path,
        benchmark='libero_spatial',
        seq_length=10,
        frame_stack=1,
        overlap=3,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        get_action_padding=True
    )
    print(f"Number of segments (with overlap): {len(dataset_overlap)}")
    
    # Test Case 3: With frame stacking
    print("\nTest Case 3: With Frame Stacking")
    dataset_stack = LIBERODataset(
        data_path=base_path,
        benchmark='libero_spatial',
        seq_length=10,
        frame_stack=4,
        overlap=0,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=True,
        get_action_padding=True
    )
    
    # Validate overlapping sequences
    print("\n=== Validating Overlapping Sequences ===")
    # Get two consecutive sequences
    data1, actions1 = dataset_overlap[0]
    data2, actions2 = dataset_overlap[1]
    
    # Check overlap in proprioceptive data
    print("\nChecking overlap between consecutive sequences:")
    seq1_end = data1['proprioceptive'][-3:].numpy()  # Last 3 timesteps of first sequence
    seq2_start = data2['proprioceptive'][:3].numpy()  # First 3 timesteps of second sequence
    overlap_match = np.allclose(seq1_end, seq2_start)
    print(f"Overlap matches: {overlap_match}")
    print(f"Last 3 timesteps of sequence 1:\n{seq1_end}")
    print(f"First 3 timesteps of sequence 2:\n{seq2_start}")
    
    # Validate padding masks
    print("\n=== Validating Padding Masks ===")
    # Get a sequence near the end of a demonstration (likely padded)
    for i in range(len(dataset_overlap)):
        data, _ = dataset_overlap[i]
        if data['pad_mask'].sum() < dataset_overlap.seq_length:  # Found a padded sequence
            print(f"\nFound padded sequence at index {i}")
            print(f"Padding mask: {data['pad_mask'].squeeze().numpy()}")
            print(f"Number of valid timesteps: {data['pad_mask'].sum().item()}")
            break
    
    # Validate frame stacking
    print("\n=== Validating Frame Stacking ===")
    data, _ = dataset_stack[0]
    print(f"Sequence length: {dataset_stack.seq_length}")
    print(f"Frame stack size: {dataset_stack.frame_stack}")
    if dataset_stack.pad_frame_stack:
        print("Checking first few frames (should be repeated for padding):")
        for k in ['pixels', 'pixels_egocentric', 'proprioceptive']:
            first_frames = data[k][:dataset_stack.frame_stack]
            are_identical = torch.allclose(first_frames[0], first_frames[1])
            print(f"{k} first frames are identical: {are_identical}")
    
    # Test different combinations of padding settings
    print("\n=== Testing Padding Configurations ===")
    padding_configs = [
        {"pad_frame_stack": True, "pad_seq_length": True},
        {"pad_frame_stack": True, "pad_seq_length": False},
        {"pad_frame_stack": False, "pad_seq_length": True},
        {"pad_frame_stack": False, "pad_seq_length": False},
    ]
    
    for config in padding_configs:
        dataset = LIBERODataset(
            data_path=base_path,
            benchmark='libero_spatial',
            seq_length=10,
            frame_stack=4,
            overlap=2,
            get_pad_mask=True,
            get_action_padding=True,
            **config
        )
        print(f"\nConfiguration: {config}")
        print(f"Number of segments: {len(dataset)}")
        data, _ = dataset[0]
        print(f"First sequence padding mask sum: {data['pad_mask'].sum().item()}/{dataset.seq_length}")
    
    print("\n=== All Tests Completed ===")
    
    # Clean up
    dataset_basic.close()
    dataset_overlap.close()
    dataset_stack.close()
    print("\nDatasets closed successfully")