# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the Kinova JACO2 (6-Dof) robotic arm.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/kinova_env.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the Kinova JACO2 (6-Dof) robotic arm.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors.camera import Camera, CameraCfg  # Add camera imports

# Add this new import to access local assets
import os

##
# Pre-defined configs
##
# isort: off
from isaaclab_assets import (
    KINOVA_JACO2_N6S300_CFG,
)

# isort: on


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create just a single origin for the Kinova arm
    origins = define_origins(num_origins=1, spacing=2.0)

    # Set up the Kinova JACO2 (6-Dof) arm
    prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])
    
    # -- Kitchen Environment
    kitchen_usd_path = os.path.join(os.getcwd(), "assets/Kitchen_set/Kitchen_set.usd")
    cfg = sim_utils.UsdFileCfg(usd_path=kitchen_usd_path)
    cfg.func("/World/Origin/Kitchen", cfg, translation=(0.5, 0.5, 0.0))
    # Apply scale to the kitchen prim after creation
    prim_utils.set_prim_property("/World/Origin/Kitchen", "xformOp:scale", (0.01, 0.01, 0.01))
    # Make sure the scale operation is in the transform order
    prim_utils.set_prim_property("/World/Origin/Kitchen", "xformOpOrder", ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
    
    # -- Table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    cfg.func("/World/Origin/Table", cfg, translation=(0.0, 0.0, 0.8))
    
    # -- Robot
    kinova_arm_cfg = KINOVA_JACO2_N6S300_CFG.replace(prim_path="/World/Origin/Robot")
    kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    kinova_j2n6s300 = Articulation(cfg=kinova_arm_cfg)
    
    # -- Wrist Camera
    # Create a camera on the wrist link (j2n6s300_link_6)
    wrist_camera_cfg = CameraCfg(
        prim_path="/World/Origin/Robot/j2n6s300_link_6/wrist_camera",
        update_period=0.0,  # Set to 0.0 to update every frame
        width=320,
        height=240,
        data_types=["rgb"],  # Only RGB data
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.01, 10.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.02),  # Small offset from the wrist link
            rot=(0.5, -0.5, 0.5, -0.5),  # Orientation with camera pointing forward
            convention="ros"
        ),
    )
    wrist_camera = Camera(cfg=wrist_camera_cfg)

    # return the scene information
    scene_entities = {
        "kinova_j2n6s300": kinova_j2n6s300,
        "wrist_camera": wrist_camera,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            
            # Reset camera if present - do this before resetting the robot
            # as the camera might be attached to the robot
            if "wrist_camera" in entities:
                try:
                    entities["wrist_camera"].reset()
                except Exception as e:
                    print(f"Camera reset error: {e}")
            
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                if isinstance(robot, Articulation):
                    # root state
                    root_state = robot.data.default_root_state.clone()
                    root_state[:, :3] += origins[index]
                    robot.write_root_pose_to_sim(root_state[:, :7])
                    robot.write_root_velocity_to_sim(root_state[:, 7:])
                    # set joint positions
                    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                    robot.write_joint_state_to_sim(joint_pos, joint_vel)
                    # clear internal buffers
                    robot.reset()
            
            print("[INFO]: Resetting robot state...")
        # apply random actions to the robot
        robot = entities["kinova_j2n6s300"]
        # Get velocity limits from robot configuration
        max_velocity_limits = 100.0  # As defined in kinova.py
        
        # Joint velocity control
        joint_vel_target = torch.randint(-100, 101, size=robot.data.joint_vel.shape, device=robot.data.joint_vel.device)*0.3  # Random integer velocity commands from -100 to 100
        print(joint_vel_target)
        joint_vel_target = joint_vel_target.clamp_(
            -max_velocity_limits, max_velocity_limits  # Using the defined limits
        )
        robot.set_joint_velocity_target(joint_vel_target)  # This method should exist in the API
        robot.write_data_to_sim()
        
        # Step physics first
        sim.step()
        
        # Update camera after physics step
        if "wrist_camera" in entities:
            try:
                # Update camera without any arguments
                entities["wrist_camera"].update()
                
                # Print camera info every 50 frames
                if count % 50 == 0:
                    print("-------------------------------")
                    print(f"Camera info at frame {count}:")
                    print(entities["wrist_camera"])
                    
                    # Safely check and print data attributes
                    if hasattr(entities["wrist_camera"], "data"):
                        if hasattr(entities["wrist_camera"].data, "output"):
                            if "rgb" in entities["wrist_camera"].data.output:
                                print("Camera RGB shape:", entities["wrist_camera"].data.output["rgb"].shape)
                    print("-------------------------------")
            except Exception as e:
                print(f"Camera update error: {e}")
        
        # update sim-time
        sim_time += sim_dt
        count += 1
        
        # Update robot after camera update
        if "kinova_j2n6s300" in entities:
            entities["kinova_j2n6s300"].update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
