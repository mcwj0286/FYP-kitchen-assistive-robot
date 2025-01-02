from torch.utils.data import Dataset
import numpy as np
class TransformedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return {
            "pixels": data["obs"]["agentview_rgb"],
            "pixels_egocentric": data["obs"]["eye_in_hand_rgb"],
            "proprioceptive": np.concatenate([
                data["obs"]["gripper_states"],
                data["obs"]["joint_states"]
            ], axis=-1),
            "task_emb": data["task_emb"],
            # "actions": data["actions"]

        } ,  data["actions"]