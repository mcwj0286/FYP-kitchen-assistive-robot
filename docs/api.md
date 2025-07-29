# API Reference

This document provides comprehensive API documentation for the Kitchen Assistive Robot system components.

## Core Modules

### 1. Agent System (`llm_ai_agent/`)

#### `agents.py`

The main entry point for creating and managing AI agents.

```python
def create_agent(
    agent_type: str = "base_agent",
    use_hardware: bool = False,
    capture_image: str = "",
    max_retries: int = 3,
    timeout: int = 30
) -> ConfigurableAgent:
    """
    Create a configured AI agent instance.
    
    Args:
        agent_type: Type of agent to create (matches YAML config filename)
        use_hardware: Enable real hardware connections
        capture_image: Auto-capture setting ("environment", "wrist", "both", "")
        max_retries: Maximum API call retries on failure
        timeout: Request timeout in seconds
        
    Returns:
        Configured agent instance
        
    Raises:
        AgentConfigurationError: If configuration file not found or invalid
        HardwareConnectionError: If hardware enabled but not accessible
        
    Example:
        >>> agent = create_agent("kitchen_assistant", use_hardware=True)
        >>> response = agent.process_to_string("Move to home position")
    """
```

#### `configurable_agent.py`

Core agent implementation class.

```python
class ConfigurableAgent:
    """
    Configurable AI agent with tool-based capabilities.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        use_hardware: bool = False,
        capture_image: str = "",
        api_client: Optional[Any] = None
    ):
        """
        Initialize agent with configuration.
        
        Args:
            config: Agent configuration dictionary
            use_hardware: Enable hardware tool access
            capture_image: Image capture mode
            api_client: Optional custom API client
        """
    
    def process_to_string(self, user_input: str) -> str:
        """
        Process user input and return string response.
        
        Args:
            user_input: Natural language input from user
            
        Returns:
            Agent's text response
            
        Raises:
            APIError: If language model API call fails
            ToolExecutionError: If tool execution fails
            
        Example:
            >>> response = agent.process_to_string("What is 2 + 2?")
            >>> print(response)  # "4"
        """
    
    def process_with_image(
        self, 
        user_input: str, 
        image_path: str
    ) -> str:
        """
        Process input with image context.
        
        Args:
            user_input: Text input
            image_path: Path to image file
            
        Returns:
            Agent response incorporating visual analysis
        """
    
    def call_tool(
        self, 
        tool_name: str, 
        tool_input: str
    ) -> str:
        """
        Call a specific tool directly.
        
        Args:
            tool_name: Name of tool to execute
            tool_input: Input data for tool
            
        Returns:
            Tool execution result
            
        Raises:
            ToolNotFoundError: If tool name not recognized
            ToolExecutionError: If tool execution fails
        """
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools for this agent.
        
        Returns:
            List of tool names available to agent
        """
    
    def update_memory(
        self, 
        memory_type: str, 
        key: str, 
        value: Any
    ) -> bool:
        """
        Update persistent agent memory.
        
        Args:
            memory_type: Type of memory ("action_plans", "positions", "locations")
            key: Memory key
            value: Value to store
            
        Returns:
            True if update successful
        """
```

### 2. Hardware Interface (`hardware_tools.py`)

#### `KinovaArmController`

Interface for Kinova Gen2 robotic arm control.

```python
class KinovaArmController:
    """
    Controller for Kinova Gen2 JACO robotic arm.
    """
    
    def __init__(
        self,
        ip_address: str = "192.168.1.10",
        username: str = "admin",
        password: str = "admin"
    ):
        """
        Initialize robot connection.
        
        Args:
            ip_address: Robot IP address
            username: Robot login username
            password: Robot login password
            
        Raises:
            RobotConnectionError: If unable to connect to robot
        """
    
    def move_to_position(
        self,
        position: List[float],
        velocity: float = 0.1,
        blocking: bool = True
    ) -> bool:
        """
        Move robot to Cartesian position.
        
        Args:
            position: [x, y, z, rx, ry, rz] in meters and radians
            velocity: Movement velocity (0.0-1.0)
            blocking: Wait for movement completion
            
        Returns:
            True if movement initiated successfully
            
        Example:
            >>> arm.move_to_position([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
        """
    
    def get_current_position(self) -> List[float]:
        """
        Get current robot end-effector position.
        
        Returns:
            Current position as [x, y, z, rx, ry, rz]
        """
    
    def move_home(self) -> bool:
        """
        Move robot to home position.
        
        Returns:
            True if successful
        """
    
    def set_gripper_position(
        self,
        position: float,
        force: float = 50.0
    ) -> bool:
        """
        Control gripper position.
        
        Args:
            position: Gripper position (0.0=open, 1.0=closed)
            force: Gripper force limit (0-100%)
            
        Returns:
            True if command successful
        """
    
    def emergency_stop(self) -> bool:
        """
        Emergency stop robot motion.
        
        Returns:
            True if stop successful
        """
    
    def get_joint_angles(self) -> List[float]:
        """
        Get current joint angles.
        
        Returns:
            List of joint angles in radians
        """
    
    def is_moving(self) -> bool:
        """
        Check if robot is currently moving.
        
        Returns:
            True if robot is in motion
        """
```

#### Camera Interface

```python
class CameraController:
    """
    Interface for camera system management.
    """
    
    def __init__(
        self,
        environment_camera_id: int = 0,
        wrist_camera_id: int = 1
    ):
        """
        Initialize camera system.
        
        Args:
            environment_camera_id: USB index for environment camera
            wrist_camera_id: USB index for wrist camera
        """
    
    def capture_image(
        self,
        camera: str = "environment",
        resolution: Tuple[int, int] = (128, 128)
    ) -> np.ndarray:
        """
        Capture image from specified camera.
        
        Args:
            camera: Camera to use ("environment", "wrist", "both")
            resolution: Output image resolution
            
        Returns:
            Captured image as numpy array
            
        Raises:
            CameraError: If camera not accessible
        """
    
    def get_camera_info(self, camera: str) -> Dict[str, Any]:
        """
        Get camera information and capabilities.
        
        Args:
            camera: Camera identifier
            
        Returns:
            Dictionary with camera specifications
        """
    
    def calibrate_camera(
        self,
        camera: str,
        calibration_images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate camera using checkerboard images.
        
        Args:
            camera: Camera to calibrate
            calibration_images: List of calibration images
            
        Returns:
            Tuple of (camera_matrix, distortion_coefficients)
        """
```

### 3. Model Interface (`models/`)

#### Action Models

```python
class ActionModel(nn.Module):
    """
    Base class for robotic action models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize action model.
        
        Args:
            config: Model configuration dictionary
        """
    
    def forward(
        self,
        images: torch.Tensor,
        language: torch.Tensor,
        robot_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through action model.
        
        Args:
            images: Visual input (B, C, H, W)
            language: Language encoding (B, L, D)
            robot_state: Robot state vector (B, S)
            
        Returns:
            Predicted actions (B, T, A) where T=action_chunk_size, A=action_dim
        """
    
    def predict_action(
        self,
        observation: Dict[str, Any]
    ) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            observation: Dictionary containing:
                - 'images': RGB image array
                - 'language': Task instruction string
                - 'robot_state': Current robot state
                
        Returns:
            Predicted action vector [vx, vy, vz, wx, wy, wz, gripper]
        """

class MoEActionModel(ActionModel):
    """
    Mixture of Experts action model.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        num_experts: int,
        expert_tasks: List[str]
    ):
        """
        Initialize MoE model.
        
        Args:
            config: Base model configuration
            num_experts: Number of expert networks
            expert_tasks: Task descriptions for each expert
        """
    
    def route_to_expert(self, language_instruction: str) -> int:
        """
        Route instruction to appropriate expert.
        
        Args:
            language_instruction: Task instruction
            
        Returns:
            Expert ID (0 to num_experts-1)
        """
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """
        Get utilization statistics for each expert.
        
        Returns:
            Dictionary mapping expert_id to utilization_percentage
        """
```

#### Training Interface

```python
class ModelTrainer:
    """
    Training interface for action models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any]
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> Dict[str, List[float]]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Training history dictionary
        """
    
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data_loader: Evaluation data loader
            
        Returns:
            Evaluation metrics dictionary
        """
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        optimizer_state: Dict[str, Any]
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dictionary
        """
    
    def load_checkpoint(
        self,
        filepath: str
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Tuple of (epoch, optimizer_state)
        """
```

### 4. Tools System (`tools.py`)

#### Core Tools

```python
def calculator(input_str: str) -> str:
    """
    Perform mathematical calculations.
    
    Args:
        input_str: Mathematical expression as string
        
    Returns:
        Calculation result
        
    Example:
        >>> calculator("15 * 32")
        "480"
    """

def text_processor(input_str: str) -> str:
    """
    Process and analyze text.
    
    Args:
        input_str: JSON string with processing parameters
        
    Returns:
        Processing result
        
    Example:
        >>> text_processor('{"action": "word_count", "text": "Hello world"}')
        "2"
    """

def speak(input_str: str) -> str:
    """
    Convert text to speech output.
    
    Args:
        input_str: Text to speak
        
    Returns:
        Confirmation message
        
    Example:
        >>> speak("Hello, how can I help you?")
        "Speech output: Hello, how can I help you?"
    """
```

#### Hardware Tools

```python
def move_position(input_str: str) -> str:
    """
    Move robot to specified position.
    
    Args:
        input_str: JSON string with position parameters:
            {
                "x": float, "y": float, "z": float,
                "theta_x": float, "theta_y": float, "theta_z": float,
                "fingers": float
            }
            
    Returns:
        Movement result message
        
    Example:
        >>> move_position('{"x": 0.5, "y": 0.0, "z": 0.3, "theta_x": 0.0, "theta_y": 0.0, "theta_z": 0.0, "fingers": 0.5}')
        "Robot moved to position [0.5, 0.0, 0.3, 0.0, 0.0, 0.0] with gripper at 50%"
    """

def capture(input_str: str) -> str:
    """
    Capture image from camera system.
    
    Args:
        input_str: JSON string with capture parameters:
            {"camera": "environment|wrist|both"}
            
    Returns:
        Image capture confirmation with base64 encoded image data
        
    Example:
        >>> capture('{"camera": "environment"}')
        "Image captured from environment camera: data:image/jpeg;base64,..."
    """
```

#### Memory Tools

```python
def get_action_plans(input_str: str = "") -> str:
    """
    Retrieve stored action plans.
    
    Args:
        input_str: Optional filter criteria (JSON string)
        
    Returns:
        YAML-formatted action plans
        
    Example:
        >>> get_action_plans()
        "opening_jar:\n  description: Step-by-step process...\n  steps: ..."
    """

def get_action_positions(input_str: str = "") -> str:
    """
    Retrieve stored robot positions.
    
    Args:
        input_str: Optional position name filter
        
    Returns:
        YAML-formatted position data
        
    Example:
        >>> get_action_positions()
        "jar_opening_position:\n  coordinates: [0.5, 0.3, 0.2, 0.0, 0.0, 0.0]..."
    """

def get_item_locations(input_str: str = "") -> str:
    """
    Retrieve known item locations.
    
    Args:
        input_str: Optional item name filter
        
    Returns:
        YAML-formatted location data
        
    Example:
        >>> get_item_locations("coffee")
        "coffee:\n  location: [0.2, 0.5, 0.1]\n  description: Coffee container..."
    """
```

### 5. Configuration System (`config_loader.py`)

```python
def load_agent_config(agent_type: str) -> Dict[str, Any]:
    """
    Load agent configuration from YAML file.
    
    Args:
        agent_type: Agent type identifier
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If configuration file not found or invalid
    """

def load_tool_config(tool_category: str) -> Dict[str, Any]:
    """
    Load tool configuration.
    
    Args:
        tool_category: Tool category name
        
    Returns:
        Tool configuration dictionary
    """

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """

def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override values
        
    Returns:
        Merged configuration dictionary
    """
```

### 6. Data Processing (`dataset.py`)

```python
class RobotDataset(Dataset):
    """
    PyTorch dataset for robot demonstration data.
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        sequence_length: int = 1
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to demonstration data
            transform: Optional data transforms
            sequence_length: Length of action sequences
        """
    
    def __len__(self) -> int:
        """Get dataset size."""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - 'images': Image tensor (C, H, W)
                - 'language': Language token IDs
                - 'robot_state': Robot state vector
                - 'actions': Action sequence
        """

def create_data_loader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create PyTorch data loader.
    
    Args:
        data_path: Path to dataset
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        
    Returns:
        Configured DataLoader
    """
```

## Error Handling

### Exception Classes

```python
class KitchenRobotError(Exception):
    """Base exception for kitchen robot system."""

class AgentConfigurationError(KitchenRobotError):
    """Agent configuration related errors."""

class HardwareConnectionError(KitchenRobotError):
    """Hardware connection and communication errors."""

class ToolExecutionError(KitchenRobotError):
    """Tool execution related errors."""

class RobotSafetyError(KitchenRobotError):
    """Robot safety related errors."""

class ModelInferenceError(KitchenRobotError):
    """Model inference related errors."""

class CameraError(KitchenRobotError):
    """Camera system related errors."""

class APIError(KitchenRobotError):
    """Language model API related errors."""
```

### Error Handling Patterns

```python
# Example error handling in agent methods
def process_to_string(self, user_input: str) -> str:
    """Process user input with comprehensive error handling."""
    try:
        # Main processing logic
        response = self._generate_response(user_input)
        return response
        
    except APIError as e:
        # Handle API-specific errors
        logger.error(f"API error: {e}")
        return "I'm having trouble connecting to my language model. Please try again."
        
    except ToolExecutionError as e:
        # Handle tool execution errors
        logger.error(f"Tool execution error: {e}")
        return f"I encountered an error while executing a tool: {e}"
        
    except RobotSafetyError as e:
        # Handle safety-critical errors
        logger.critical(f"Safety error: {e}")
        self.emergency_stop()
        return "Emergency stop activated due to safety concern. Please check the system."
        
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error: {e}")
        return "I encountered an unexpected error. Please check the system logs."
```

## Usage Examples

### Basic Agent Usage

```python
from llm_ai_agent.agents import create_agent

# Create and use kitchen assistant
agent = create_agent("kitchen_assistant", use_hardware=False)

# Basic interaction
response = agent.process_to_string("Hello, can you help me?")
print(response)

# Tool usage
response = agent.process_to_string("What is 15 multiplied by 32?")
print(response)  # Should output: 480

# Memory access
response = agent.process_to_string("Where is the coffee located?")
print(response)
```

### Hardware Integration

```python
# Enable hardware for real robot
hardware_agent = create_agent(
    agent_type="kitchen_assistant",
    use_hardware=True,
    capture_image="environment"
)

# Robot control
response = hardware_agent.process_to_string("Move to home position")

# Vision-based interaction
image_response = hardware_agent.process_to_string("What do you see in the workspace?")
```

### Model Training

```python
from models.bc_act_policy import ActionModel
from dataset import create_data_loader

# Load data
train_loader = create_data_loader("./demonstrations/", batch_size=32)

# Initialize model
model = ActionModel(config)

# Train model
trainer = ModelTrainer(model, training_config)
history = trainer.train(train_loader, val_loader, num_epochs=100)

# Save trained model
trainer.save_checkpoint("./models/trained_model.pth", epoch=100, optimizer.state_dict())
```

### Custom Tool Development

```python
# Define custom tool
def custom_analysis_tool(input_str: str) -> str:
    """Custom tool for specialized analysis."""
    try:
        params = json.loads(input_str)
        analysis_type = params.get('type')
        data = params.get('data')
        
        if analysis_type == 'ingredient_nutrition':
            return analyze_nutrition(data)
        elif analysis_type == 'recipe_difficulty':
            return assess_difficulty(data)
        else:
            return "Unknown analysis type"
            
    except Exception as e:
        return f"Analysis error: {str(e)}"

# Register tool
from llm_ai_agent.tools import TOOLS
TOOLS["custom_analysis_tool"] = custom_analysis_tool

# Use in agent configuration
# Add to tools.include list in YAML config
```

This API reference provides comprehensive documentation for integrating with and extending the Kitchen Assistive Robot system.