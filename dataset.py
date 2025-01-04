from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
import os
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
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
        num_queries: int = 1,  # Number of future actions to concatenate
        max_word_len: int = 77,  # Maximum length for tokenization
        seq_length: int = 10  # Length of each sequence segment
    ):
        """
        Args:
            data_path (str): Base path to LIBERO dataset
            benchmark (str): Which benchmark to load
            device (str): Device to load tensors to (not used anymore, keeping for compatibility)
            load_task_emb (bool): Whether to load task embeddings
            num_queries (int): Number of future actions to concatenate
            max_word_len (int): Maximum length for tokenization
            seq_length (int): Length of each sequence segment
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
        
        # Initialize BERT tokenizer and model if loading task embeddings
        if self.load_task_emb:
            logging.getLogger("transformers").setLevel(logging.ERROR)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.bert_model = AutoModel.from_pretrained("bert-base-cased")  # Keep on CPU
            
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
                        
                        # Create complete segments of seq_length timesteps
                        num_complete_segments = n_timesteps // seq_length
                        for i in range(num_complete_segments):
                            start_idx = i * seq_length
                            self.segment_map.append((file_path, demo_key, task_name, start_idx, False, seq_length))
                        
                        # Handle remaining timesteps if any
                        remaining_timesteps = n_timesteps % seq_length
                        if remaining_timesteps > 0:
                            start_idx = num_complete_segments * seq_length
                            self.segment_map.append((file_path, demo_key, task_name, start_idx, True, remaining_timesteps))
                        
                # Pre-compute task embedding if needed
                if self.load_task_emb and task_name not in self.task_embeddings:
                    self.task_embeddings[task_name] = self._encode_task(task_name)
                        
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
    
    def _process_actions(self, actions: np.ndarray) -> np.ndarray:
        """Process actions to include future actions
        
        Args:
            actions: Original actions array of shape (T, 7)
            
        Returns:
            Processed actions of shape (T, 7 * num_queries) where each timestep
            includes the current and future actions, padded with zeros if needed
        """
        T, action_dim = actions.shape
        total_dim = action_dim * self.num_queries
        
        # Initialize output array with zeros (this handles padding automatically)
        processed_actions = np.zeros((T, total_dim), dtype=actions.dtype)
        
        # For each timestep, concatenate the next num_queries actions
        for t in range(T):
            future_actions = actions[t:t + self.num_queries]  # Get next actions (might be less than num_queries)
            actual_futures = len(future_actions)
            # Place the available future actions in the correct position
            processed_actions[t, :actual_futures * action_dim] = future_actions.flatten()
            
        return processed_actions
    
    def _encode_task(self, task_name: str) -> torch.Tensor:
        """Encode task name using BERT"""
        tokens = self.tokenizer(
            text=task_name,
            add_special_tokens=True,
            max_length=self.max_word_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        # Keep on CPU
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        # Get BERT embedding
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask)
            task_emb = outputs["pooler_output"]  # Use pooled output for [CLS] token
            
        return task_emb  # Shape: (1, 768)
    
    def __len__(self) -> int:
        return len(self.segment_map)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a fixed-length segment from a demonstration trajectory"""
        file_path, demo_key, task_name, start_idx, is_padded, actual_length = self.segment_map[idx]
        f = self._get_file(file_path)
        
        demo = f['data'][demo_key]
        
        # Initialize data with zeros (handles padding automatically)
        data = {
            "pixels": np.zeros((self.seq_length, 128, 128, 3), dtype=np.float32),
            "pixels_egocentric": np.zeros((self.seq_length, 128, 128, 3), dtype=np.float32),
            "proprioceptive": np.zeros((self.seq_length, 9), dtype=np.float32)
        }
        
        # Load actual data
        data["pixels"][:actual_length] = demo['obs']['agentview_rgb'][start_idx:start_idx + actual_length]
        data["pixels_egocentric"][:actual_length] = demo['obs']['eye_in_hand_rgb'][start_idx:start_idx + actual_length]
        data["proprioceptive"][:actual_length] = np.concatenate([
            demo['obs']['gripper_states'][start_idx:start_idx + actual_length],
            demo['obs']['joint_states'][start_idx:start_idx + actual_length]
        ], axis=-1)
        
        # Process actions
        actions = np.zeros((self.seq_length, 7), dtype=np.float32)
        actions[:actual_length] = demo['actions'][start_idx:start_idx + actual_length]
        gt_actions = self._process_actions(actions)
        
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
            
        return data, torch.from_numpy(gt_actions)

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
    
    print("Creating dataset...")
    dataset = LIBERODataset(
        data_path=base_path,
        benchmark='libero_spatial',
        device='cuda',
        num_queries=10,  # Concatenate 10 future actions
        load_task_emb=True
    )
    
    # 1. Validate task organization
    print("\n=== Task Organization Validation ===")
    for task_name, files in dataset.task_files.items():
        n_demos = sum(1 for f in files for _ in h5py.File(f, 'r')['data'].keys())
        print(f"\nTask: {task_name}")
        print(f"Number of files: {len(files)}")
        print(f"Number of demos: {n_demos}")
        assert n_demos == 50, f"Expected 50 demos per task, got {n_demos} for {task_name}"
        
        # Check task embedding
        if dataset.load_task_emb:
            emb = dataset.task_embeddings[task_name]
            print(f"Task embedding shape: {emb.shape}")
            assert emb.shape == (1, 768), f"Expected embedding shape (1, 768), got {emb.shape}"
    
    # 2. Validate data loading and shapes
    print("\n=== Data Shape Validation ===")
    # Sample first demo from each task
    task_names = list(dataset.task_files.keys())
    for task_name in task_names:
        # Find first demo for this task
        demo_idx = next(i for i, (_, _, t) in enumerate(dataset.segment_map) if t == task_name)
        data = dataset[demo_idx]
        
        print(f"\nValidating shapes for task: {task_name}")
        print(f"Demo index: {demo_idx}")
        
        # Check data shapes
        assert data["pixels"].shape[-3:] == (128, 128, 3), f"Wrong image shape: {data['pixels'].shape}"
        assert data["pixels_egocentric"].shape[-3:] == (128, 128, 3), f"Wrong ego image shape: {data['pixels_egocentric'].shape}"
        assert data["proprioceptive"].shape[-1] == 9, f"Wrong proprio shape: {data['proprioceptive'].shape}"
        assert data["actions"].shape[-1] == 7 * dataset.num_queries, f"Wrong action shape: {data['actions'].shape}"
        
        print("Shapes:")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
                
        # Validate pixel range
        assert torch.all((data["pixels"] >= 0) & (data["pixels"] <= 1)), "Pixels not normalized to [0,1]"
        assert torch.all((data["pixels_egocentric"] >= 0) & (data["pixels_egocentric"] <= 1)), "Ego pixels not normalized to [0,1]"
    
    # 3. Validate action processing
    print("\n=== Action Processing Validation ===")
    data = dataset[0]
    actions = data["actions"]
    print(f"Action tensor shape: {actions.shape}")
    print(f"Number of queries: {dataset.num_queries}")
    print(f"Action dimension: {actions.shape[-1]}")
    assert actions.shape[-1] == 7 * dataset.num_queries, f"Expected action dim {7 * dataset.num_queries}, got {actions.shape[-1]}"
    
    # Check if later timesteps are properly zero-padded
    last_timestep_actions = actions[-1]  # Should be partially or fully padded
    n_nonzero = torch.count_nonzero(last_timestep_actions)
    print(f"Number of non-zero values in last timestep: {n_nonzero}")
    print(f"Total values in last timestep: {last_timestep_actions.shape[0]}")
    
    # 4. Validate task embedding consistency
    if dataset.load_task_emb:
        print("\n=== Task Embedding Consistency Validation ===")
        # Check if all demos of same task have same embedding
        for task_name in task_names:
            # Get indices for all demos of this task
            task_indices = [i for i, (_, _, t) in enumerate(dataset.segment_map) if t == task_name][:2]  # Check first two demos
            
            if len(task_indices) >= 2:
                emb1 = dataset[task_indices[0]]["task_emb"]
                emb2 = dataset[task_indices[1]]["task_emb"]
                assert torch.allclose(emb1, emb2), f"Inconsistent embeddings for task {task_name}"
                print(f"Task {task_name}: Embeddings consistent across demos")
    
    print("\n=== All Validations Passed ===")
    print(f"Total number of demonstrations: {len(dataset)}")
    print(f"Number of tasks: {len(dataset.task_files)}")
    
    # Clean up
    dataset.close()
    print("\nDataset closed successfully")