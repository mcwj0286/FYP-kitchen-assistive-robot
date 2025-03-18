#!/usr/bin/env python3

import h5py
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import shutil


def preprocess_hdf5(input_file_path, output_file_path=None):
    """
    Preprocess the HDF5 file by removing frames with all-zero actions.
    
    Parameters:
        input_file_path (str): Path to the input HDF5 file.
        output_file_path (str): Path for the output HDF5 file. If None, a default name will be created.
    
    Returns:
        str: Path to the processed HDF5 file.
    """
    # Create output filename if not provided
    if output_file_path is None:
        dir_name = os.path.dirname(input_file_path)
        base_name = os.path.basename(input_file_path)
        name, ext = os.path.splitext(base_name)
        output_file_path = os.path.join(dir_name, f"{name}_processed{ext}")
    
    print(f"Processing file: {input_file_path}")
    print(f"Output will be saved to: {output_file_path}")
    
    # Create a new HDF5 file for the processed data
    with h5py.File(input_file_path, 'r') as input_file, h5py.File(output_file_path, 'w') as output_file:
        # Find all demo groups
        demo_groups = [key for key in input_file.keys() if key.startswith("demo_")]
        
        # Process each demo group
        for demo_name in tqdm(demo_groups, desc="Processing demos"):
            input_demo_group = input_file[demo_name]
            
            # Skip if no actions dataset
            if "actions" not in input_demo_group:
                print(f"No actions dataset found in {demo_name}, copying as is")
                # Copy the entire group as is
                input_file.copy(input_demo_group, output_file, name=demo_name)
                continue
            
            # Get actions dataset
            actions_dataset = input_demo_group["actions"]
            total_frames = actions_dataset.shape[0]
            
            # Find indices of non-zero action frames
            non_zero_indices = []
            for i in range(total_frames):
                action = actions_dataset[i]
                if not np.all(action == 0):  # If not all elements are zero
                    non_zero_indices.append(i)
            
            # Skip if all actions are zero or no non-zero actions found
            if not non_zero_indices:
                print(f"All actions in {demo_name} are zero, skipping")
                continue
            
            # Create a new demo group in the output file
            output_demo_group = output_file.create_group(demo_name)
            
            # Process and copy actions dataset (keeping only non-zero action frames)
            output_actions = output_demo_group.create_dataset(
                "actions", 
                shape=(len(non_zero_indices), actions_dataset.shape[1]),
                dtype=actions_dataset.dtype
            )
            for new_idx, old_idx in enumerate(non_zero_indices):
                output_actions[new_idx] = actions_dataset[old_idx]
            
            # Copy attributes if any
            for attr_name, attr_value in input_demo_group["actions"].attrs.items():
                output_actions.attrs[attr_name] = attr_value
            
            # Process and copy other datasets (joint_angles, cartesian_pose, etc.)
            for dataset_name, dataset in input_demo_group.items():
                if dataset_name == "actions":
                    continue  # Already handled
                
                if isinstance(dataset, h5py.Dataset):
                    # Dataset has the same first dimension as actions (frame count)
                    if dataset.shape[0] == total_frames:
                        output_dataset = output_demo_group.create_dataset(
                            dataset_name,
                            shape=(len(non_zero_indices),) + dataset.shape[1:],
                            dtype=dataset.dtype
                        )
                        for new_idx, old_idx in enumerate(non_zero_indices):
                            output_dataset[new_idx] = dataset[old_idx]
                        
                        # Copy attributes
                        for attr_name, attr_value in dataset.attrs.items():
                            output_dataset.attrs[attr_name] = attr_value
                    else:
                        # If the dataset doesn't have the same frame count, copy as is
                        input_demo_group.copy(dataset, output_demo_group, name=dataset_name)
                elif isinstance(dataset, h5py.Group):
                    # Handle groups like "images" which may contain multiple camera datasets
                    output_subgroup = output_demo_group.create_group(dataset_name)
                    
                    for subitem_name, subitem in dataset.items():
                        if isinstance(subitem, h5py.Dataset) and subitem.shape[0] == total_frames:
                            # This is a camera dataset with frame count matching actions
                            output_subitem = output_subgroup.create_dataset(
                                subitem_name,
                                shape=(len(non_zero_indices),) + subitem.shape[1:],
                                dtype=subitem.dtype
                            )
                            for new_idx, old_idx in enumerate(non_zero_indices):
                                output_subitem[new_idx] = subitem[old_idx]
                            
                            # Copy attributes
                            for attr_name, attr_value in subitem.attrs.items():
                                output_subitem.attrs[attr_name] = attr_value
                        else:
                            # If not matching frame count, copy as is
                            dataset.copy(subitem, output_subgroup, name=subitem_name)
            
            # Print stats for this demo
            print(f"  {demo_name}: Kept {len(non_zero_indices)}/{total_frames} frames " 
                  f"({len(non_zero_indices)/total_frames*100:.2f}%)")
    
    return output_file_path


def process_directory(input_dir):
    """
    Process all HDF5 files in the input directory.
    
    Parameters:
        input_dir (str): Path to the directory containing HDF5 files.
    """
    # Create an output directory for processed files if it doesn't exist
    output_dir = os.path.join(input_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files in the directory
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    hdf5_files.extend(glob.glob(os.path.join(input_dir, "*.h5")))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Process each file
    for input_file_path in hdf5_files:
        base_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, base_name)
        preprocess_hdf5(input_file_path, output_file_path)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess HDF5 files by removing frames with all-zero actions")
    parser.add_argument("input_path", type=str, help="Path to the directory containing HDF5 files or a single HDF5 file")
    args = parser.parse_args()
    
    input_path = args.input_path
    
    # Check if the input path exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist")
        return
    
    # Process a directory or a single file
    if os.path.isdir(input_path):
        process_directory(input_path)
    else:
        # Process a single file
        preprocess_hdf5(input_path)


if __name__ == "__main__":
    main() 