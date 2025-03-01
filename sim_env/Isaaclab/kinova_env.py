# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script creates a basic environment with a plane and the Kinova Gen 2 JACO 6-DOF arm.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p kinova_env.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script creates a basic environment with the Kinova JACO2 6-DOF arm.")
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

##
# Pre-defined configs
##
# Import the Kinova JACO2 6-DOF configuration
from isaaclab_assets import KINOVA_JACO2_N6S300_CFG


def design_scene() -> dict:
    """Designs the scene with a ground plane and Kinova JACO2 6-DOF arm."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create root for robot and table
    prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))
    
    # Create table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd")
    cfg.func("/World/Origin/Table", cfg, translation=(0.0, 0.0, 0.8))
    
    # Setup Kinova JACO2 (6-Dof) arm
    kinova_arm_cfg = KINOVA_JACO2_N6S300_CFG.replace(prim_path="/World/Origin/Robot")
    kinova_arm_cfg.init_state.pos = (0.0, 0.0, 0.8)
    kinova_j2n6s300 = Articulation(cfg=kinova_arm_cfg)

    # return the scene information
    scene_entities = {
        "kinova_j2n6s300": kinova_j2n6s300,
    }
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Get the Kinova arm
    kinova_arm = entities["kinova_j2n6s300"]
    
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the arm
            # root state
            root_state = kinova_arm.data.default_root_state.clone()
            kinova_arm.write_root_pose_to_sim(root_state[:, :7])
            kinova_arm.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions
            joint_pos, joint_vel = kinova_arm.data.default_joint_pos.clone(), kinova_arm.data.default_joint_vel.clone()
            kinova_arm.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            kinova_arm.reset()
            print("[INFO]: Resetting robot state...")
            
        # Apply random actions to the robot
        # generate random joint positions
        joint_pos_target = kinova_arm.data.default_joint_pos + torch.randn_like(kinova_arm.data.joint_pos) * 0.1
        joint_pos_target = joint_pos_target.clamp_(
            kinova_arm.data.soft_joint_pos_limits[..., 0], kinova_arm.data.soft_joint_pos_limits[..., 1]
        )
        # apply action to the robot
        kinova_arm.set_joint_position_target(joint_pos_target)
        # write data to sim
        kinova_arm.write_data_to_sim()
        
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        kinova_arm.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.8])
    # design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
