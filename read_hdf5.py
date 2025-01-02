# from libero.libero import benchmark , get_libero_path
# from libero.libero.envs import OffScreenRenderEnv
# import os

# benchmark_dict = benchmark.get_benchmark_dict()
# task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
# task_suite = benchmark_dict[task_suite_name]()

# # retrieve a specific task
# task_id = 0
# task = task_suite.get_task(task_id)
# task_name = task.name
# task_description = task.language
# task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
# print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
#       f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# # step over the environment
# env_args = {
#     "bddl_file_name": task_bddl_file,
#     "camera_heights": 128,
#     "camera_widths": 128
# }
# env = OffScreenRenderEnv(**env_args)
# env.seed(0)
# env.reset()
# init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
# init_state_id = 0
# env.set_init_state(init_states[init_state_id])

# dummy_action = [0.] * 7
# for step in range(10):
#     obs, reward, done, info = env.step(dummy_action)
#     print(f"[info] step {step}, reward: {reward}, done: {done}")
# env.close()

# read hdf5 file
path = '/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robotic/sim_env/LIBERO/libero/datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5'
import h5py

# Open the HDF5 file
with h5py.File(path, 'r') as f:
    # Display all top-level groups and datasets
    def printname(name):
        print(name)
    
    f.visit(printname)  # This will print all names in the file
    actions = f['data/demo_8/states'][:]
    print("Actions:", actions.shape)   
     # Navigate to the demo_9 group
    # demo_group = f['data/demo_9']
    
    # # List all the datasets in the demo_9 group
    # for key in demo_group.keys():
    #     if isinstance(demo_group[key], h5py.Dataset):
    #         # If it's a dataset, print its name and shape
    #         print(f"{key}: {demo_group[key].shape}")
    #     elif isinstance(demo_group[key], h5py.Group):
    #         # If it's a subgroup, print the datasets inside it
    #         for sub_key in demo_group[key].keys():
    #             dataset = demo_group[key][sub_key]
    #             if isinstance(dataset, h5py.Dataset):
    #                 print(f"{key}/{sub_key}: {dataset.shape}")