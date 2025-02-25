import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)

scene = gs.Scene()

plane = scene.add_entity(
    gs.morphs.Plane(),
)
# kitchen = scene.add_entity(
#     gs.morphs.Mesh(file="/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/Genesis/assets/kitchen.glb",
#                        scale=1.0,
#                        collision=True,
#                        ),
    # pose=gs.transformations.make_pose(translation=(0, 0, 0)),
   
# )
kinova = scene.add_entity(
    gs.morphs.URDF(
        file='/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/Genesis/assets/kinova_j2s7s300_ign/urdf/j2s7s300.urdf',
        fixed=True,
    ),
    # gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()

# Define joint names for the Kinova Jaco 2 robot
# These joint names should match those in the URDF file
joint_names = [
    "j2s7s300_joint_1",
    "j2s7s300_joint_2",
    "j2s7s300_joint_3",
    "j2s7s300_joint_4",
    "j2s7s300_joint_5",
    "j2s7s300_joint_6",
    "j2s7s300_joint_7",
    # Finger joints - we'll exclude these for control
    # "j2s7s300_joint_finger_1",
    # "j2s7s300_joint_finger_2",
    # "j2s7s300_joint_finger_3",
]

# Get the DOF indices for each joint
dof_indices = []
for name in joint_names:
    try:
        joint = kinova.get_joint(name)
        dof_indices.append(joint.dof_idx_local)
        print(f"Found joint: {name}, DOF index: {joint.dof_idx_local}")
    except Exception as e:
        print(f"Error getting joint {name}: {e}")

if not dof_indices:
    print("No valid joints found. Please check joint names in the URDF file.")
    exit(1)

num_arm_joints = len(dof_indices)
print(f"Controlling {num_arm_joints} arm joints")

# Set control gains for better joint control
kinova.set_dofs_kp(
    kp=np.array([4500] * num_arm_joints),
    dofs_idx_local=dof_indices,
)
kinova.set_dofs_kv(
    kv=np.array([450] * num_arm_joints),
    dofs_idx_local=dof_indices,
)

# Set force range for safety
kinova.set_dofs_force_range(
    lower=np.array([-87] * num_arm_joints),
    upper=np.array([87] * num_arm_joints),
    dofs_idx_local=dof_indices,
)

# Prepare velocity array
joint_velocities = np.zeros(num_arm_joints)

# Main simulation loop
for i in range(100000):
    # Example: Oscillating joint velocities using sine waves
    if i < 5000:  # Let the robot settle first
        joint_velocities = np.zeros(num_arm_joints)
    else:
        for j in range(num_arm_joints):
            # Create a sinusoidal motion with different phase for each joint
            # Use 30% of maximum velocity for safety
            joint_velocities[j] = 0.3 * np.sin(0.001 * i + j * 0.5)
    
    # Apply joint velocities using the proper control method
    kinova.control_dofs_velocity(joint_velocities, dof_indices)
    
    # Step the simulation
    scene.step()
    
    # Print joint positions every 1000 steps
    if i % 1000 == 0:
        positions = kinova.get_dofs_position(dof_indices)
        print(f"Step {i}, Joint positions: {positions}")
