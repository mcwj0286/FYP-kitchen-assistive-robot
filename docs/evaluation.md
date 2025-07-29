# Evaluation Guide

This guide covers evaluation methodologies, benchmarks, and performance metrics for the Kitchen Assistive Robot system.

## Evaluation Overview

The system supports multiple evaluation paradigms:

1. **Simulation Evaluation**: Using LIBERO benchmark tasks
2. **Real Robot Evaluation**: Physical manipulation tasks  
3. **Comparative Analysis**: Baseline vs. MoE models
4. **Ablation Studies**: Component-wise performance analysis
5. **Generalization Testing**: Robustness to environmental changes

## Simulation Evaluation (LIBERO)

### LIBERO Benchmark Suite

The system integrates with the LIBERO benchmark for standardized evaluation of robotic manipulation policies.

#### Available Benchmarks

```python
LIBERO_BENCHMARKS = {
    'libero_spatial': {
        'tasks': 10,
        'description': 'Spatial reasoning and object manipulation',
        'examples': [
            'Pick up the black bowl and put it in the top drawer',
            'Put the wine bottle on top of the cabinet',
            'Move the yellow block to the yellow zone'
        ]
    },
    'libero_object': {
        'tasks': 10, 
        'description': 'Object-centric manipulation tasks',
        'examples': [
            'Pick up the red mug',
            'Grasp the wooden block',
            'Move the metal bowl'
        ]
    },
    'libero_goal': {
        'tasks': 10,
        'description': 'Goal-conditioned manipulation',
        'examples': [
            'Put all objects in the drawer',
            'Stack all blocks',
            'Clear the table'
        ]
    },
    'libero_long': {
        'tasks': 10,
        'description': 'Long-horizon manipulation sequences',
        'examples': [
            'Open drawer, pick up object, place in cabinet, close drawer',
            'Move all red objects to left side, blue to right side'
        ]
    }
}
```

#### Running LIBERO Evaluation

```bash
# Evaluate baseline model
python eval.py \
    --model_path ./models/libero_baseline.pth \
    --benchmark libero_spatial \
    --num_episodes 50 \
    --seed 42 \
    --video_output ./evaluation_videos/baseline/

# Evaluate MoE model
python eval.py \
    --model_path ./models/libero_moe.pth \
    --benchmark libero_spatial \
    --num_episodes 50 \
    --seed 42 \
    --video_output ./evaluation_videos/moe/ \
    --use_moe

# Evaluate across all benchmarks
python eval.py \
    --model_path ./models/libero_moe.pth \
    --benchmark all \
    --num_episodes 20 \
    --output_dir ./results/comprehensive_eval/
```

#### LIBERO Evaluation Script

```python
# eval.py - Core evaluation logic
import torch
import numpy as np
from models.bc_act_policy import ActionModel
from models.moe_policy import MoEActionModel
import libero.libero as libero

class LiberoEvaluator:
    """Evaluator for LIBERO benchmark tasks."""
    
    def __init__(
        self, 
        model_path: str,
        benchmark: str = "libero_spatial",
        use_moe: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            benchmark: LIBERO benchmark name
            use_moe: Whether model uses MoE architecture
        """
        self.benchmark = benchmark
        self.use_moe = use_moe
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        if use_moe:
            self.model = MoEActionModel(checkpoint['config'])
        else:
            self.model = ActionModel(checkpoint['config'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize LIBERO environment
        self.env = libero.make(benchmark)
        
    def evaluate_task(
        self, 
        task_id: int,
        num_episodes: int = 10,
        max_steps: int = 500,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on specific task.
        
        Args:
            task_id: LIBERO task identifier
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics dictionary
        """
        success_count = 0
        episode_lengths = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset(task_id=task_id)
            episode_reward = 0
            
            for step in range(max_steps):
                # Prepare observation for model
                model_obs = self._prepare_observation(obs)
                
                # Get model prediction
                with torch.no_grad():
                    action = self.model.predict_action(model_obs)
                
                # Execute action
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if render:
                    self.env.render()
                
                if done:
                    if info.get('success', False):
                        success_count += 1
                    episode_lengths.append(step + 1)
                    break
            else:
                # Episode timed out
                episode_lengths.append(max_steps)
            
            episode_rewards.append(episode_reward)
        
        # Compute metrics
        metrics = {
            'success_rate': success_count / num_episodes,
            'avg_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }
        
        return metrics
    
    def evaluate_benchmark(
        self,
        num_episodes_per_task: int = 10,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on full benchmark.
        
        Args:
            num_episodes_per_task: Episodes per task
            output_path: Path to save results
            
        Returns:
            Complete benchmark results
        """
        task_results = {}
        overall_metrics = []
        
        num_tasks = self.env.get_num_tasks()
        
        for task_id in range(num_tasks):
            print(f"Evaluating task {task_id + 1}/{num_tasks}...")
            
            task_metrics = self.evaluate_task(
                task_id=task_id,
                num_episodes=num_episodes_per_task
            )
            
            task_name = self.env.get_task_name(task_id)
            task_results[task_name] = task_metrics
            overall_metrics.append(task_metrics['success_rate'])
            
            print(f"Task {task_name}: {task_metrics['success_rate']:.1%} success")
        
        # Compute overall statistics
        results = {
            'benchmark': self.benchmark,
            'overall_success_rate': np.mean(overall_metrics),
            'success_rate_std': np.std(overall_metrics),
            'task_results': task_results,
            'num_tasks': num_tasks,
            'episodes_per_task': num_episodes_per_task
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def _prepare_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare LIBERO observation for model input."""
        return {
            'images': obs['agentview_rgb'],  # Primary camera view
            'language': obs['task_description'],
            'robot_state': obs['robot_state']
        }
```

### LIBERO Results Analysis

```python
# analyze_libero_results.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_benchmark_results(results_path: str):
    """Analyze and visualize LIBERO benchmark results."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract task-level results
    task_names = []
    success_rates = []
    
    for task_name, metrics in results['task_results'].items():
        task_names.append(task_name)
        success_rates.append(metrics['success_rate'])
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Task-wise success rates
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(task_names)), success_rates)
    plt.title('Task-wise Success Rates')
    plt.xlabel('Task ID')
    plt.ylabel('Success Rate')
    plt.xticks(range(len(task_names)), [f'T{i+1}' for i in range(len(task_names))])
    
    # Color bars based on success rate
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        if rate >= 0.8:
            bar.set_color('green')
        elif rate >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Overall statistics
    plt.subplot(2, 2, 2)
    overall_rate = results['overall_success_rate']
    plt.pie([overall_rate, 1-overall_rate], 
            labels=[f'Success\n{overall_rate:.1%}', f'Failure\n{1-overall_rate:.1%}'],
            colors=['green', 'red'],
            autopct='%1.1f%%')
    plt.title('Overall Success Rate')
    
    # Success rate distribution
    plt.subplot(2, 2, 3)
    plt.hist(success_rates, bins=10, alpha=0.7, color='blue')
    plt.title('Success Rate Distribution')
    plt.xlabel('Success Rate')
    plt.ylabel('Number of Tasks')
    
    # Task difficulty analysis
    plt.subplot(2, 2, 4)
    difficulty_categories = []
    for rate in success_rates:
        if rate >= 0.8:
            difficulty_categories.append('Easy')
        elif rate >= 0.5:
            difficulty_categories.append('Medium')
        else:
            difficulty_categories.append('Hard')
    
    difficulty_counts = pd.Series(difficulty_categories).value_counts()
    plt.pie(difficulty_counts.values, labels=difficulty_counts.index, 
            colors=['green', 'orange', 'red'], autopct='%1.1f%%')
    plt.title('Task Difficulty Distribution')
    
    plt.tight_layout()
    plt.savefig(results_path.replace('.json', '_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# Generate analysis report
def generate_libero_report(baseline_results: str, moe_results: str, output_path: str):
    """Generate comparative analysis report."""
    
    with open(baseline_results, 'r') as f:
        baseline = json.load(f)
    
    with open(moe_results, 'r') as f:
        moe = json.load(f)
    
    report = f"""
# LIBERO Benchmark Evaluation Report

## Overall Performance

| Model | Success Rate | Standard Deviation | Tasks Evaluated |
|-------|-------------|-------------------|-----------------|
| Baseline | {baseline['overall_success_rate']:.1%} | {baseline['success_rate_std']:.3f} | {baseline['num_tasks']} |
| MoE | {moe['overall_success_rate']:.1%} | {moe['success_rate_std']:.3f} | {moe['num_tasks']} |
| **Improvement** | **{moe['overall_success_rate'] - baseline['overall_success_rate']:.1%}** | - | - |

## Task-by-Task Comparison

| Task | Baseline | MoE | Improvement |
|------|----------|-----|-------------|
"""
    
    for task_name in baseline['task_results'].keys():
        baseline_rate = baseline['task_results'][task_name]['success_rate']
        moe_rate = moe['task_results'][task_name]['success_rate']
        improvement = moe_rate - baseline_rate
        
        report += f"| {task_name} | {baseline_rate:.1%} | {moe_rate:.1%} | {improvement:+.1%} |\n"
    
    report += f"""
## Key Findings

- MoE model achieved {moe['overall_success_rate']:.1%} overall success rate vs {baseline['overall_success_rate']:.1%} for baseline
- Performance improvement of {moe['overall_success_rate'] - baseline['overall_success_rate']:.1%} absolute
- {sum(1 for task in baseline['task_results'].keys() if moe['task_results'][task]['success_rate'] > baseline['task_results'][task]['success_rate'])} out of {len(baseline['task_results'])} tasks showed improvement with MoE

## Recommendations

Based on the evaluation results:
- {'MoE architecture provides significant benefits' if moe['overall_success_rate'] > baseline['overall_success_rate'] + 0.05 else 'MoE provides modest improvements'}
- {'Consider further hyperparameter tuning for underperforming tasks' if min(moe['task_results'][task]['success_rate'] for task in moe['task_results'].keys()) < 0.5 else 'Performance is consistent across tasks'}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report
```

## Real Robot Evaluation

### Real Robot Task Suite

```python
REAL_ROBOT_TASKS = {
    'grasp_cup_side': {
        'description': 'Grasp cup from the side approach',
        'object': 'cup',
        'approach': 'side',
        'difficulty': 'medium'
    },
    'grasp_cup_overhead': {
        'description': 'Grasp cup from overhead approach', 
        'object': 'cup',
        'approach': 'overhead',
        'difficulty': 'medium'
    },
    'grasp_orange': {
        'description': 'Grasp spherical orange object',
        'object': 'orange',
        'approach': 'optimal',
        'difficulty': 'easy'
    },
    'grasp_blender_handle': {
        'description': 'Grasp blender handle',
        'object': 'blender',
        'approach': 'handle',
        'difficulty': 'hard'
    }
}
```

### Real Robot Evaluation Script

```bash
# eval_kinova.py - Real robot evaluation
python eval_kinova.py \
    --model_path ./models/kinova_baseline.pth \
    --tasks "grasp_cup_side,grasp_orange,grasp_blender_handle" \
    --num_trials 10 \
    --record_video \
    --output_dir ./real_robot_results/

# Evaluate with different environmental conditions
python eval_kinova.py \
    --model_path ./models/kinova_moe.pth \
    --tasks all \
    --num_trials 5 \
    --conditions "standard,low_light,cluttered,new_location" \
    --output_dir ./robustness_test/
```

```python
# eval_kinova.py implementation
import time
import cv2
import numpy as np
from typing import Dict, List, Any
from llm_ai_agent.hardware_tools import KinovaArmController, CameraController

class RealRobotEvaluator:
    """Evaluator for real robot manipulation tasks."""
    
    def __init__(
        self,
        model_path: str,
        robot_ip: str = "192.168.1.10",
        use_moe: bool = False
    ):
        """Initialize real robot evaluator."""
        
        # Load model
        self.model = self._load_model(model_path, use_moe)
        
        # Initialize hardware
        self.robot = KinovaArmController(ip_address=robot_ip)
        self.camera = CameraController()
        
        # Safety limits
        self.workspace_bounds = {
            'x': [0.2, 0.8],
            'y': [-0.4, 0.4], 
            'z': [0.05, 0.5]
        }
        
    def evaluate_task(
        self,
        task_name: str,
        num_trials: int = 10,
        record_video: bool = False,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Evaluate model on real robot task.
        
        Args:
            task_name: Name of task to evaluate
            num_trials: Number of trials to run
            record_video: Whether to record videos
            timeout: Maximum time per trial (seconds)
            
        Returns:
            Task evaluation results
        """
        
        task_config = REAL_ROBOT_TASKS[task_name]
        successes = 0
        trial_results = []
        
        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials} - {task_name}")
            
            # Setup trial
            trial_result = {
                'trial_id': trial,
                'task_name': task_name,
                'success': False,
                'execution_time': 0,
                'failure_reason': None,
                'video_path': None
            }
            
            # Initialize video recording
            if record_video:
                video_writer = self._setup_video_recording(
                    f"{task_name}_trial_{trial}.mp4"
                )
                trial_result['video_path'] = f"{task_name}_trial_{trial}.mp4"
            
            try:
                # Reset robot to starting position
                self.robot.move_home()
                time.sleep(2)
                
                # Wait for human to position object
                input(f"Position {task_config['object']} and press Enter to start trial...")
                
                # Execute trial
                start_time = time.time()
                success = self._execute_trial(
                    task_config, 
                    video_writer if record_video else None,
                    timeout
                )
                execution_time = time.time() - start_time
                
                trial_result['success'] = success
                trial_result['execution_time'] = execution_time
                
                if success:
                    successes += 1
                    print(f"✅ Trial {trial + 1} SUCCESS")
                else:
                    print(f"❌ Trial {trial + 1} FAILED")
                
            except Exception as e:
                trial_result['failure_reason'] = str(e)
                print(f"❌ Trial {trial + 1} ERROR: {e}")
                
                # Emergency stop if needed
                if "safety" in str(e).lower():
                    self.robot.emergency_stop()
                    break
                    
            finally:
                if record_video and 'video_writer' in locals():
                    video_writer.release()
            
            trial_results.append(trial_result)
            
            # Safety pause between trials
            time.sleep(5)
        
        # Compute task metrics
        task_metrics = {
            'task_name': task_name,
            'success_rate': successes / num_trials,
            'num_trials': num_trials,
            'num_successes': successes,
            'avg_execution_time': np.mean([r['execution_time'] for r in trial_results if r['success']]),
            'trial_results': trial_results
        }
        
        return task_metrics
    
    def _execute_trial(
        self, 
        task_config: Dict[str, Any], 
        video_writer: Any = None,
        timeout: int = 30
    ) -> bool:
        """Execute single manipulation trial."""
        
        start_time = time.time()
        
        # Phase 1: Move to approach position
        approach_position = self._get_approach_position(task_config)
        self.robot.move_to_position(approach_position)
        
        # Phase 2: Execute grasping with action model
        grasp_success = False
        model_steps = 0
        max_model_steps = 200  # ~10 seconds at 20Hz
        
        while model_steps < max_model_steps and time.time() - start_time < timeout:
            # Capture current observation
            image = self.camera.capture_image("wrist", resolution=(128, 128))
            robot_state = self.robot.get_current_position() + self.robot.get_joint_angles()
            
            # Prepare observation for model
            obs = {
                'images': image,
                'language': task_config['description'],
                'robot_state': robot_state
            }
            
            # Get action from model
            action = self.model.predict_action(obs)
            
            # Safety check
            if not self._is_action_safe(action):
                print("⚠️ Unsafe action detected, stopping trial")
                return False
            
            # Execute action
            self._execute_action(action)
            
            # Record video frame
            if video_writer:
                frame = self.camera.capture_image("environment", resolution=(640, 480))
                video_writer.write(frame)
            
            # Check for grasp success (simplified)
            if self._check_grasp_success(task_config):
                grasp_success = True
                break
                
            model_steps += 1
            time.sleep(0.05)  # 20Hz control loop
        
        # Phase 3: Lift and verify grasp
        if grasp_success:
            lift_position = approach_position.copy()
            lift_position[2] += 0.1  # Lift 10cm
            self.robot.move_to_position(lift_position)
            time.sleep(2)
            
            # Final verification
            final_success = self._verify_final_grasp(task_config)
            return final_success
        
        return False
    
    def _is_action_safe(self, action: np.ndarray) -> bool:
        """Check if action is within safety bounds."""
        current_pos = self.robot.get_current_position()
        
        # Check workspace bounds
        new_pos = [
            current_pos[0] + action[0] * 0.01,  # Scale velocity to position
            current_pos[1] + action[1] * 0.01,
            current_pos[2] + action[2] * 0.01
        ]
        
        for i, (coord, bounds) in enumerate(zip(new_pos, ['x', 'y', 'z'])):
            if not (self.workspace_bounds[bounds][0] <= coord <= self.workspace_bounds[bounds][1]):
                return False
        
        # Check velocity limits
        max_velocity = 0.2  # m/s
        if np.linalg.norm(action[:3]) > max_velocity:
            return False
        
        return True
    
    def _check_grasp_success(self, task_config: Dict[str, Any]) -> bool:
        """Check if grasp attempt was successful."""
        # Simplified grasp detection based on gripper state and force feedback
        gripper_state = self.robot.get_gripper_state()
        
        # Check if gripper is closed but not fully
        if 0.2 < gripper_state['position'] < 0.8:
            # Check if there's appropriate force feedback
            if gripper_state['force'] > 10:  # Minimum force threshold
                return True
        
        return False
```

### Evaluation Conditions

#### Standard Conditions
- Consistent lighting (500+ lux)
- Clean workspace
- Objects in trained positions
- No environmental distractions

#### Robustness Testing

```python
EVALUATION_CONDITIONS = {
    'standard': {
        'lighting': 'normal',
        'workspace': 'clean',
        'object_position': 'trained',
        'distractions': 'none'
    },
    'low_light': {
        'lighting': 'dimmed (200 lux)',
        'workspace': 'clean', 
        'object_position': 'trained',
        'distractions': 'none'
    },
    'cluttered': {
        'lighting': 'normal',
        'workspace': 'cluttered with additional objects',
        'object_position': 'trained',
        'distractions': 'visual_clutter'
    },
    'new_location': {
        'lighting': 'normal', 
        'workspace': 'clean',
        'object_position': 'shifted ±5cm from training',
        'distractions': 'none'
    },
    'challenging': {
        'lighting': 'variable',
        'workspace': 'cluttered',
        'object_position': 'random within workspace',
        'distractions': 'visual_and_auditory'
    }
}
```

## Comparative Analysis

### Model Comparison Framework

```python
def compare_models(
    model_paths: Dict[str, str],
    evaluation_tasks: List[str],
    num_trials: int = 10
) -> Dict[str, Any]:
    """
    Compare multiple models on same task set.
    
    Args:
        model_paths: Dictionary mapping model names to checkpoint paths
        evaluation_tasks: List of tasks to evaluate
        num_trials: Number of trials per task per model
        
    Returns:
        Comprehensive comparison results
    """
    
    comparison_results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating {model_name}...")
        
        evaluator = RealRobotEvaluator(
            model_path=model_path,
            use_moe=('moe' in model_name.lower())
        )
        
        model_results = {}
        for task in evaluation_tasks:
            task_results = evaluator.evaluate_task(
                task_name=task,
                num_trials=num_trials,
                record_video=False
            )
            model_results[task] = task_results
        
        comparison_results[model_name] = model_results
    
    # Generate comparison statistics
    comparison_stats = generate_comparison_stats(comparison_results)
    
    return {
        'model_results': comparison_results,
        'comparison_stats': comparison_stats
    }

def generate_comparison_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate statistical comparison between models."""
    
    model_names = list(results.keys())
    tasks = list(results[model_names[0]].keys())
    
    stats = {
        'overall_performance': {},
        'task_performance': {},
        'statistical_significance': {}
    }
    
    # Overall performance comparison
    for model_name in model_names:
        overall_success_rates = []
        for task in tasks:
            overall_success_rates.append(results[model_name][task]['success_rate'])
        
        stats['overall_performance'][model_name] = {
            'mean_success_rate': np.mean(overall_success_rates),
            'std_success_rate': np.std(overall_success_rates),
            'min_success_rate': np.min(overall_success_rates),
            'max_success_rate': np.max(overall_success_rates)
        }
    
    # Task-by-task comparison
    for task in tasks:
        task_comparison = {}
        for model_name in model_names:
            task_comparison[model_name] = results[model_name][task]['success_rate']
        stats['task_performance'][task] = task_comparison
    
    return stats
```

## Performance Metrics

### Success Rate Metrics

```python
def calculate_success_metrics(trial_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate comprehensive success rate metrics."""
    
    successes = [r['success'] for r in trial_results]
    n_trials = len(trial_results)
    n_successes = sum(successes)
    
    # Basic success rate
    success_rate = n_successes / n_trials if n_trials > 0 else 0
    
    # Confidence interval (Wilson score interval)
    from scipy import stats
    if n_trials > 0:
        ci_lower, ci_upper = stats.proportion_confint(
            n_successes, n_trials, alpha=0.05, method='wilson'
        )
    else:
        ci_lower = ci_upper = 0
    
    # Success rate by trial number (learning effects)
    success_by_trial = {}
    for i, result in enumerate(trial_results):
        success_by_trial[i] = result['success']
    
    return {
        'success_rate': success_rate,
        'confidence_interval_lower': ci_lower,
        'confidence_interval_upper': ci_upper,
        'n_trials': n_trials,
        'n_successes': n_successes,
        'success_by_trial': success_by_trial
    }
```

### Execution Time Analysis

```python
def analyze_execution_times(trial_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze execution time patterns."""
    
    # Filter successful trials only
    successful_times = [
        r['execution_time'] for r in trial_results 
        if r['success'] and r['execution_time'] > 0
    ]
    
    if not successful_times:
        return {'error': 'No successful trials with valid execution times'}
    
    return {
        'mean_time': np.mean(successful_times),
        'median_time': np.median(successful_times),
        'std_time': np.std(successful_times),
        'min_time': np.min(successful_times),
        'max_time': np.max(successful_times),
        'efficiency_score': 1.0 / np.mean(successful_times) if successful_times else 0
    }
```

### Failure Analysis

```python
def analyze_failures(trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns in task failures."""
    
    failed_trials = [r for r in trial_results if not r['success']]
    
    if not failed_trials:
        return {'message': 'No failures to analyze'}
    
    # Categorize failure reasons
    failure_categories = {}
    for trial in failed_trials:
        reason = trial.get('failure_reason', 'unknown')
        if reason not in failure_categories:
            failure_categories[reason] = 0
        failure_categories[reason] += 1
    
    # Timing analysis of failures
    failure_times = [r['execution_time'] for r in failed_trials if r['execution_time'] > 0]
    
    return {
        'total_failures': len(failed_trials),
        'failure_rate': len(failed_trials) / len(trial_results),
        'failure_categories': failure_categories,
        'mean_failure_time': np.mean(failure_times) if failure_times else 0,
        'early_failures': len([t for t in failure_times if t < 5]),  # < 5 seconds
        'late_failures': len([t for t in failure_times if t > 20])   # > 20 seconds
    }
```

## Reporting and Visualization

### Automated Report Generation

```python
def generate_evaluation_report(
    results: Dict[str, Any],
    output_path: str = "evaluation_report.html"
) -> str:
    """Generate comprehensive HTML evaluation report."""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Kitchen Assistive Robot - Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .success { color: green; font-weight: bold; }
        .failure { color: red; font-weight: bold; }
        .metric { background-color: #f9f9f9; }
        .chart { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Kitchen Assistive Robot Evaluation Report</h1>
    <p>Generated on: {timestamp}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric">
        <p><strong>Overall Success Rate:</strong> <span class="success">{overall_success:.1%}</span></p>
        <p><strong>Tasks Evaluated:</strong> {num_tasks}</p>
        <p><strong>Total Trials:</strong> {total_trials}</p>
        <p><strong>Models Compared:</strong> {num_models}</p>
    </div>
    
    <h2>Model Performance Comparison</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Success Rate</th>
            <th>Avg. Execution Time</th>
            <th>Best Task</th>
            <th>Worst Task</th>
        </tr>
        {model_comparison_rows}
    </table>
    
    <h2>Task-by-Task Results</h2>
    <table>
        <tr>
            <th>Task</th>
            <th>Description</th>
            <th>Difficulty</th>
            <th>Best Model</th>
            <th>Success Rate Range</th>
        </tr>
        {task_results_rows}
    </table>
    
    <h2>Key Findings</h2>
    <ul>
        {key_findings}
    </ul>
    
    <h2>Recommendations</h2>
    <ul>
        {recommendations}
    </ul>
    
</body>
</html>
    """
    
    # Process results and populate template
    # [Template population logic here]
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path
```

### Visualization Tools

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_visualizations(results: Dict[str, Any], output_dir: str):
    """Create comprehensive performance visualizations."""
    
    # Success rate comparison
    plt.figure(figsize=(15, 10))
    
    # 1. Overall success rate comparison
    plt.subplot(2, 3, 1)
    models = list(results['model_results'].keys())
    overall_rates = [results['comparison_stats']['overall_performance'][model]['mean_success_rate'] 
                    for model in models]
    
    bars = plt.bar(models, overall_rates)
    plt.title('Overall Success Rate by Model')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    # Color bars based on performance
    for bar, rate in zip(bars, overall_rates):
        if rate >= 0.8:
            bar.set_color('green')
        elif rate >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # 2. Task-wise performance heatmap
    plt.subplot(2, 3, 2)
    task_performance = results['comparison_stats']['task_performance']
    tasks = list(task_performance.keys())
    
    heatmap_data = []
    for model in models:
        model_performance = [task_performance[task][model] for task in tasks]
        heatmap_data.append(model_performance)
    
    sns.heatmap(heatmap_data, 
                xticklabels=[t.replace('_', ' ').title() for t in tasks],
                yticklabels=models,
                annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('Task Performance Heatmap')
    
    # 3. Execution time comparison
    plt.subplot(2, 3, 3)
    # [Execution time plotting logic]
    
    # 4. Success rate distribution
    plt.subplot(2, 3, 4)
    all_success_rates = []
    model_labels = []
    for model in models:
        model_rates = [task_performance[task][model] for task in tasks]
        all_success_rates.extend(model_rates)
        model_labels.extend([model] * len(model_rates))
    
    plt.boxplot([task_performance[task][model] for task in tasks] for model in models], 
                labels=models)
    plt.title('Success Rate Distribution')
    plt.ylabel('Success Rate')
    
    # 5. Learning curves (if trial-by-trial data available)
    plt.subplot(2, 3, 5)
    # [Learning curve plotting logic]
    
    # 6. Failure analysis
    plt.subplot(2, 3, 6)
    # [Failure analysis plotting logic]
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
```

This comprehensive evaluation framework provides systematic methods for assessing the Kitchen Assistive Robot system performance across simulation and real-world scenarios.