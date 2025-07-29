# Training Guide

This guide covers training procedures for the Kitchen Assistive Robot system, including data collection, model training, and evaluation protocols.

## Overview

The system uses two main types of models:
1. **Action Models**: Deep learning models for fine-grained manipulation
2. **Vision-Language Models**: Pre-trained models used for high-level planning (not trained in this system)

## Data Collection

### 1. Real Robot Data Collection

#### Setup Requirements

- Kinova Gen2 robotic arm properly calibrated
- Dual camera system (wrist + environment cameras)
- PS4 controller for teleoperation
- Stable lighting conditions
- Clear workspace with known objects

#### Teleoperation Interface

The system uses a PS4 controller mapped to Cartesian velocity commands:

```python
# Controller mapping (from methodology_section.md)
CONTROLLER_MAPPING = {
    'left_stick_x': 'velocity_x',        # X-axis translation
    'left_stick_y': 'velocity_y',        # Y-axis translation  
    'l2_r2_triggers': 'velocity_z',      # Z-axis translation (vertical)
    'right_stick_x': 'angular_y',        # Pitch rotation
    'right_stick_y': 'angular_x',        # Roll rotation
    'l1_r1_buttons': 'angular_z',        # Yaw rotation
    'square_circle': 'gripper_control'   # Gripper open/close
}
```

#### Data Collection Protocol

```bash
# Start data collection session
python kinova_trainer.py --mode collect \
    --task_name "grasp_cup_side" \
    --num_demonstrations 30 \
    --save_path ./demonstrations/

# Collection process for each demonstration:
# 1. Position robot at starting pose
# 2. Place target object in workspace
# 3. Start recording (captures all modalities)
# 4. Perform teleoperated demonstration
# 5. End recording and save trajectory
```

**Collected Data Format:**
```python
demonstration = {
    'images_wrist': np.array,      # Shape: (T, 128, 128, 3)
    'images_environment': np.array, # Shape: (T, 128, 128, 3) 
    'robot_states': np.array,      # Shape: (T, 15) - joints + pose
    'actions': np.array,           # Shape: (T, 7) - velocities + gripper
    'language_instruction': str,   # Task description
    'task_metadata': dict         # Additional task information
}
```

#### Task Categories for Data Collection

Based on the project's experimental setup, collect data for these tasks:

1. **Cup Grasping Tasks:**
   ```bash
   # Side grasping
   python collect_demonstrations.py --task "grasp cup from side" --object cup --approach side
   
   # Overhead grasping  
   python collect_demonstrations.py --task "grasp cup from above" --object cup --approach overhead
   ```

2. **Other Object Tasks:**
   ```bash
   # Orange grasping
   python collect_demonstrations.py --task "grasp orange" --object orange --approach optimal
   
   # Blender handle grasping
   python collect_demonstrations.py --task "grasp blender handle" --object blender --approach handle
   ```

### 2. Simulation Data (LIBERO)

The system supports training on LIBERO benchmark tasks:

```bash
# Generate LIBERO demonstration data
python libero_trainer.py --mode collect \
    --benchmark libero_spatial \
    --num_demos_per_task 50 \
    --save_path ./sim_demonstrations/
```

**LIBERO Task Examples:**
- Pick up the black bowl and put it in the top drawer
- Put the wine bottle on top of the cabinet
- Move the yellow block to the yellow zone
- Open the top drawer and put the banana inside

## Model Training

### 1. Baseline Action Model Training

#### Configuration

Create a training configuration file:

```yaml
# configs/training/baseline_config.yaml
model:
  architecture: "baku_transformer"
  vision_encoder: "resnet18"
  language_encoder: "bert-base-uncased"
  hidden_dim: 512
  num_layers: 4
  num_heads: 8
  action_chunk_size: 4

training:
  learning_rate: 1e-5
  batch_size: 32
  num_epochs: 100
  weight_decay: 1e-4
  gradient_clip: 1.0
  
data:
  train_split: 0.9
  val_split: 0.1
  image_size: [128, 128]
  augmentation: true
  
logging:
  wandb_project: "kitchen-robot-training"
  log_interval: 10
  save_interval: 20
```

#### Training Command

```bash
# Train baseline model on real robot data
python kinova_trainer.py \
    --config configs/training/baseline_config.yaml \
    --data_path ./demonstrations/ \
    --output_dir ./models/baseline/ \
    --gpu_id 0

# Train on LIBERO simulation data
python libero_trainer.py \
    --config configs/training/baseline_config.yaml \
    --data_path ./sim_demonstrations/ \
    --benchmark libero_spatial \
    --output_dir ./models/libero_baseline/
```

#### Training Process

```python
# Simplified training loop structure
def train_action_model(config, data_loader):
    model = ActionModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()  # For continuous actions
    
    for epoch in range(config.num_epochs):
        for batch in data_loader:
            # Forward pass
            images = batch['images']
            language = batch['language']
            robot_state = batch['robot_state']
            target_actions = batch['actions']
            
            predicted_actions = model(images, language, robot_state)
            
            # Compute loss (NLL for discrete, MSE for continuous)
            loss = criterion(predicted_actions, target_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            # Logging
            if step % config.log_interval == 0:
                wandb.log({'train_loss': loss.item(), 'epoch': epoch})
```

### 2. Mixture of Experts (MoE) Training

#### MoE Configuration

```yaml
# configs/training/moe_config.yaml
model:
  architecture: "moe_transformer"
  num_experts: 4  # Number of distinct tasks
  expert_hidden_dim: 256
  routing_strategy: "language_similarity"
  expert_weight: 0.1  # Fixed weighting for expert outputs
  
  # Base architecture same as baseline
  vision_encoder: "resnet18"
  language_encoder: "bert-base-uncased"
  hidden_dim: 512
  
training:
  # Same training parameters as baseline
  learning_rate: 1e-5
  batch_size: 32
  num_epochs: 100
  
  # MoE specific
  load_balancing_weight: 0.0  # Not used with fixed routing
  expert_dropout: 0.1
```

#### Training Command

```bash
# Train MoE model
python kinova_trainer.py \
    --config configs/training/moe_config.yaml \
    --data_path ./demonstrations/ \
    --output_dir ./models/moe/ \
    --use_moe \
    --num_experts 4 \
    --gpu_id 0
```

#### Expert Initialization

```python
class MoEActionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        
        # Shared components
        self.vision_encoder = self._build_vision_encoder(config)
        self.language_encoder = self._build_language_encoder(config)
        self.state_encoder = self._build_state_encoder(config)
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(config) for _ in range(self.num_experts)
        ])
        
        # Language-based router
        self.router = LanguageRouter(config.expert_tasks)
        
    def forward(self, images, language, robot_state):
        # Shared feature extraction
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(language)
        state_features = self.state_encoder(robot_state)
        
        # Route to appropriate expert
        expert_id = self.router.route(language)
        selected_expert = self.experts[expert_id]
        
        # Expert prediction
        expert_output = selected_expert(vision_features, language_features, state_features)
        
        # Apply fixed weighting
        final_output = expert_output * self.config.expert_weight
        
        return final_output
```

### 3. Training Monitoring

#### Metrics to Track

```python
TRAINING_METRICS = {
    'loss': 'Primary training objective (NLL or MSE)',
    'action_accuracy': 'Percentage of actions within tolerance',
    'success_rate': 'Task completion rate (if available)',
    'expert_utilization': 'Distribution of expert usage (MoE only)',
    'gradient_norm': 'Gradient magnitude for stability monitoring',
    'learning_rate': 'Current learning rate (if using scheduler)'
}
```

#### Wandb Integration

```python
# Initialize experiment tracking
import wandb

wandb.init(
    project="kitchen-robot-training",
    config=config,
    name=f"experiment_{timestamp}",
    tags=["baseline", "real_robot", "grasping"]
)

# Log metrics during training
wandb.log({
    'train_loss': loss.item(),
    'val_loss': val_loss,
    'success_rate': success_rate,
    'epoch': epoch
})

# Log model artifacts
wandb.save('./models/checkpoint_best.pth')
```

#### Early Stopping and Checkpointing

```python
class ModelTrainer:
    def __init__(self, config):
        self.best_val_loss = float('inf')
        self.patience = config.early_stopping_patience
        self.no_improve_count = 0
        
    def should_stop_early(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improve_count = 0
            return False
        else:
            self.no_improve_count += 1
            return self.no_improve_count >= self.patience
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        checkpoint_path = f'./models/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, './models/checkpoint_best.pth')
```

## Evaluation Protocols

### 1. Simulation Evaluation (LIBERO)

```bash
# Evaluate trained model on LIBERO benchmark
python eval.py \
    --model_path ./models/libero_baseline/checkpoint_best.pth \
    --benchmark libero_spatial \
    --num_episodes 50 \
    --video_output ./evaluation_videos/
```

**LIBERO Evaluation Metrics:**
- **Success Rate**: Percentage of episodes where task is completed successfully
- **Episode Length**: Average number of steps to complete task
- **Failure Analysis**: Categorization of failure modes

### 2. Real Robot Evaluation

```bash
# Evaluate on real robot hardware
python eval_kinova.py \
    --model_path ./models/baseline/checkpoint_best.pth \
    --tasks "grasp_cup,grasp_orange,grasp_blender" \
    --num_trials 10 \
    --record_video
```

**Real Robot Evaluation Setup:**
1. **Controlled Environment**: Same lighting and object positions as training
2. **Generalization Test**: Slightly different object positions
3. **New Location Test**: Different workspace setup
4. **Robustness Test**: Varying lighting conditions

### 3. Comparative Analysis

#### Model Comparison Template

```python
# Compare different model variants
MODELS_TO_COMPARE = {
    'baseline': './models/baseline/checkpoint_best.pth',
    'moe': './models/moe/checkpoint_best.pth',
    'fine_tuned': './models/fine_tuned/checkpoint_best.pth'
}

results = {}
for model_name, model_path in MODELS_TO_COMPARE.items():
    results[model_name] = evaluate_model(
        model_path=model_path,
        test_tasks=['grasp_cup', 'grasp_orange', 'grasp_blender'],
        num_trials=20
    )

# Generate comparison report
generate_comparison_report(results, output_path='./evaluation_report.pdf')
```

## Training Best Practices

### 1. Data Quality Guidelines

**Demonstration Quality:**
- Smooth, natural movements without jerky motions
- Consistent approach strategies for each task type
- Successful completion of the intended task
- Appropriate speed (not too fast or slow)

**Data Diversity:**
- Multiple approach angles for the same object
- Slight variations in object positions
- Different lighting conditions (if possible)
- Various gripper orientations at grasp

### 2. Hyperparameter Tuning

**Key Parameters to Tune:**
```python
TUNING_RANGES = {
    'learning_rate': [1e-6, 1e-5, 1e-4],
    'batch_size': [16, 32, 64],
    'hidden_dim': [256, 512, 1024],
    'num_layers': [2, 4, 6],
    'action_chunk_size': [1, 4, 8],
    'expert_weight': [0.05, 0.1, 0.2]  # For MoE
}
```

**Systematic Search:**
```bash
# Use Optuna or similar for hyperparameter optimization
python tune_hyperparameters.py \
    --config_template configs/training/base_template.yaml \
    --search_space configs/tuning/search_space.yaml \
    --num_trials 50 \
    --study_name kitchen_robot_tuning
```

### 3. Common Issues and Solutions

#### Training Instability
- **Symptom**: Loss oscillates wildly or explodes
- **Solutions**: 
  - Reduce learning rate
  - Increase gradient clipping
  - Add batch normalization
  - Reduce batch size

#### Overfitting
- **Symptom**: Training loss decreases but validation loss increases
- **Solutions**:
  - Add dropout layers
  - Reduce model capacity
  - Increase data augmentation
  - Early stopping

#### Poor Generalization
- **Symptom**: Good performance on training data but poor on new scenarios
- **Solutions**:
  - Collect more diverse training data
  - Increase data augmentation
  - Regularization techniques
  - Domain randomization

#### MoE-Specific Issues
- **Expert Imbalance**: Some experts never activated
  - Solution: Adjust routing mechanism or add load balancing
- **Router Instability**: Inconsistent expert selection
  - Solution: Use temperature scaling or add routing regularization

### 4. Reproducibility Guidelines

```python
# Set random seeds for reproducibility
import torch
import numpy as np
import random

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Save complete training configuration
def save_training_config(config, output_dir):
    import json
    with open(f'{output_dir}/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    # Also save git commit hash for code version tracking
    import subprocess
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    with open(f'{output_dir}/git_info.txt', 'w') as f:
        f.write(f'Git commit: {git_hash}\n')
```

This comprehensive training guide provides the foundation for successfully training and evaluating the Kitchen Assistive Robot system models.