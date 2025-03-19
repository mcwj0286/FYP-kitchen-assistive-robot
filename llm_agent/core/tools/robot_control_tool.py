import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from ..base_agent import Tool

class RobotControlTool(Tool):
    """
    A tool for controlling the robot arm.
    """
    def __init__(
        self,
        name: str = "robot_control",
        description: str = "Controls the robot arm",
        enable_controller: bool = False,
        cartesian_linear_scale: float = 0.1,
        cartesian_angular_scale: float = 40.0
    ):
        """
        Initialize the robot control tool.
        
        Args:
            name: The name of the tool.
            description: A description of the tool.
            enable_controller: Whether to enable physical control of the robot.
            cartesian_linear_scale: Maximum Cartesian linear velocity in m/s.
            cartesian_angular_scale: Maximum Cartesian angular velocity in degrees/s.
        """
        super().__init__(name, description)
        self.enable_controller = enable_controller
        self.cartesian_linear_scale = cartesian_linear_scale
        self.cartesian_angular_scale = cartesian_angular_scale
        self.robot_controller = None
        
        # Initialize the robot controller
        try:
            from sim_env.Kinova_gen2.src.robot_controller import RobotController
            self.robot_controller = RobotController(enable_controller=self.enable_controller)
            self.robot_controller.initialize_devices()
            print("Robot controller initialized successfully.")
        except ImportError:
            print("Warning: RobotController not found. Robot control will be simulated.")
            self.robot_controller = None
        except Exception as e:
            print(f"Error initializing robot controller: {e}")
            self.robot_controller = None
    
    def execute(self, action: str, duration: float = 1.0, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a robot control action.
        
        Args:
            action: The action to perform (e.g., "move_left", "open_gripper").
            duration: Duration of the action in seconds.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: Result of the action.
        """
        # Map action names to functions
        action_map = {
            "move_left": self._move_left,
            "move_right": self._move_right,
            "move_forward": self._move_forward,
            "move_backward": self._move_backward,
            "move_up": self._move_up,
            "move_down": self._move_down,
            "turn_left": self._turn_left,
            "turn_right": self._turn_right,
            "pitch_up": self._pitch_up,
            "pitch_down": self._pitch_down,
            "roll_left": self._roll_left,
            "roll_right": self._roll_right,
            "open_gripper": self._open_gripper,
            "close_gripper": self._close_gripper,
            "stop": self._stop,
            "get_position": self._get_position
        }
        
        if action not in action_map:
            raise ValueError(f"Unknown action: {action}")
        
        # Execute the action
        result = {"success": False, "message": ""}
        try:
            action_func = action_map[action]
            result = action_func(duration, *args, **kwargs)
        except Exception as e:
            result = {"success": False, "message": str(e)}
        
        return result
    
    def _move_left(self, duration: float) -> Dict[str, Any]:
        """Move the arm left."""
        return self._send_cartesian_velocity(
            linear_velocity=(self.cartesian_linear_scale, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            duration=duration
        )
    
    def _move_right(self, duration: float) -> Dict[str, Any]:
        """Move the arm right."""
        return self._send_cartesian_velocity(
            linear_velocity=(-self.cartesian_linear_scale, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            duration=duration
        )
    
    def _move_forward(self, duration: float) -> Dict[str, Any]:
        """Move the arm forward."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, -self.cartesian_linear_scale, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            duration=duration
        )
    
    def _move_backward(self, duration: float) -> Dict[str, Any]:
        """Move the arm backward."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, self.cartesian_linear_scale, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            duration=duration
        )
    
    def _move_up(self, duration: float) -> Dict[str, Any]:
        """Move the arm up."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, self.cartesian_linear_scale),
            angular_velocity=(0.0, 0.0, 0.0),
            duration=duration
        )
    
    def _move_down(self, duration: float) -> Dict[str, Any]:
        """Move the arm down."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, -self.cartesian_linear_scale),
            angular_velocity=(0.0, 0.0, 0.0),
            duration=duration
        )
    
    def _turn_left(self, duration: float) -> Dict[str, Any]:
        """Turn the arm left (yaw)."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, -self.cartesian_angular_scale, 0.0),
            duration=duration
        )
    
    def _turn_right(self, duration: float) -> Dict[str, Any]:
        """Turn the arm right (yaw)."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, self.cartesian_angular_scale, 0.0),
            duration=duration
        )
    
    def _pitch_up(self, duration: float) -> Dict[str, Any]:
        """Pitch the arm up."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(-self.cartesian_angular_scale, 0.0, 0.0),
            duration=duration
        )
    
    def _pitch_down(self, duration: float) -> Dict[str, Any]:
        """Pitch the arm down."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(self.cartesian_angular_scale, 0.0, 0.0),
            duration=duration
        )
    
    def _roll_left(self, duration: float) -> Dict[str, Any]:
        """Roll the arm left."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, -self.cartesian_angular_scale),
            duration=duration
        )
    
    def _roll_right(self, duration: float) -> Dict[str, Any]:
        """Roll the arm right."""
        return self._send_cartesian_velocity(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, self.cartesian_angular_scale),
            duration=duration
        )
    
    def _open_gripper(self, duration: float) -> Dict[str, Any]:
        """Open the gripper."""
        if self.robot_controller and hasattr(self.robot_controller, 'arm'):
            try:
                self.robot_controller.arm.send_angular_velocity(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    hand_mode=1, 
                    fingers=(-3000.0, -3000.0, -3000.0), 
                    duration=duration, 
                    period=0.005
                )
                return {"success": True, "message": "Gripper opened successfully."}
            except Exception as e:
                return {"success": False, "message": f"Error opening gripper: {e}"}
        else:
            # Simulate the action
            time.sleep(min(duration, 1.0))  # Simulate shorter to make testing faster
            return {"success": True, "message": "Simulated opening gripper."}
    
    def _close_gripper(self, duration: float) -> Dict[str, Any]:
        """Close the gripper."""
        if self.robot_controller and hasattr(self.robot_controller, 'arm'):
            try:
                self.robot_controller.arm.send_angular_velocity(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    hand_mode=1, 
                    fingers=(3000.0, 3000.0, 3000.0), 
                    duration=duration, 
                    period=0.005
                )
                return {"success": True, "message": "Gripper closed successfully."}
            except Exception as e:
                return {"success": False, "message": f"Error closing gripper: {e}"}
        else:
            # Simulate the action
            time.sleep(min(duration, 1.0))  # Simulate shorter to make testing faster
            return {"success": True, "message": "Simulated closing gripper."}
    
    def _stop(self, duration: float) -> Dict[str, Any]:
        """Stop all movement."""
        time.sleep(duration)
        return {"success": True, "message": "Stopped for {:.2f} seconds.".format(duration)}
    
    def _get_position(self, *args, **kwargs) -> Dict[str, Any]:
        """Get the current Cartesian position of the robot arm."""
        if self.robot_controller and hasattr(self.robot_controller, 'arm'):
            try:
                # Get the position
                position = self.robot_controller.arm.get_cartesian_position()
                
                # Format the position
                position_dict = {
                    "position": position[:3],  # x, y, z
                    "orientation": position[3:6],  # qx, qy, qz
                    "raw": position
                }
                
                return {
                    "success": True,
                    "message": "Position retrieved successfully.",
                    "position": position_dict
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error getting position: {e}",
                    "position": None
                }
        else:
            # Return a simulated position
            return {
                "success": True,
                "message": "Simulated position.",
                "position": {
                    "position": [0.5, 0.0, 0.5],  # Simulated position
                    "orientation": [0.0, 0.0, 0.0],  # Simulated orientation
                    "raw": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]  # Simulated raw position
                }
            }
    
    def _send_cartesian_velocity(
        self,
        linear_velocity: Tuple[float, float, float],
        angular_velocity: Tuple[float, float, float],
        duration: float
    ) -> Dict[str, Any]:
        """
        Send Cartesian velocity to the robot arm.
        
        Args:
            linear_velocity: Linear velocity components (x, y, z).
            angular_velocity: Angular velocity components (x, y, z).
            duration: Duration of the movement in seconds.
            
        Returns:
            Dict[str, Any]: Result of the action.
        """
        if self.robot_controller and hasattr(self.robot_controller, 'arm'):
            try:
                self.robot_controller.arm.send_cartesian_velocity(
                    linear_velocity=linear_velocity, 
                    angular_velocity=angular_velocity, 
                    fingers=(0.0, 0.0, 0.0), 
                    duration=duration, 
                    period=0.01
                )
                return {"success": True, "message": "Movement executed successfully."}
            except Exception as e:
                return {"success": False, "message": f"Error executing movement: {e}"}
        else:
            # Simulate the action
            time.sleep(min(duration, 1.0))  # Simulate shorter to make testing faster
            return {"success": True, "message": "Simulated movement."}
    
    def close(self) -> None:
        """
        Close the robot controller.
        """
        if self.robot_controller and hasattr(self.robot_controller, "close"):
            try:
                self.robot_controller.close()
                print("Robot controller closed successfully.")
            except Exception as e:
                print(f"Error closing robot controller: {e}") 