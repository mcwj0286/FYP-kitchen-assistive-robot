import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from mpi import load_mpi
import os

# Set cache directory to a location where you have write permissions
cache_dir = os.path.expanduser("~/.cache/huggingface")
os.makedirs(cache_dir, exist_ok=True)

root_dir = "/home/john/project/FYP-kitchen-assistive-robot/models/networks/utils/MPI/mpi/checkpoints/mpi-small"
# Get device from environment or default to cuda if available
device = os.environ.get("MPI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model = load_mpi(root_dir, device, freeze=True, cache_dir=cache_dir)
model.eval()
transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])

# Generate a random image of size 224x224x3
random_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = transforms(Image.fromarray(random_image)).unsqueeze(0)
visual_input = torch.stack((image, image), dim=1) # simply repeat the current observation in downstream downstask
visual_input = visual_input.to(device=device)
lang_input = ("turn on the knob", )
embedding_with_lang_tokens = model.get_representations(visual_input, lang_input, with_lang_tokens = True)
embedding_without_lang_tokens = model.get_representations(visual_input, None, with_lang_tokens = False)
print(embedding_with_lang_tokens.shape, embedding_without_lang_tokens.shape) # (1, 218, 384), (1, 197, 384)
