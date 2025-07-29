# Installation Guide

This guide provides detailed instructions for setting up the Kitchen Assistive Robot system on your development environment.

## System Requirements

### Hardware Requirements

#### Minimum Configuration
- **CPU**: Intel i5 or AMD Ryzen 5 equivalent (4+ cores)
- **RAM**: 16GB (32GB recommended for training)
- **Storage**: 50GB available space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better recommended)

#### For Physical Robot Deployment
- **Kinova Gen2 JACO 6-DOF robotic arm**
- **USB Cameras**: 2x compatible USB cameras (Rapoo or similar)
  - One for wrist mounting
  - One for overhead/environment view
- **Intel NUC 11 or equivalent mini PC** (for on-robot computing)
- **PS4 Controller** (for teleoperation and data collection)

### Software Requirements

- **Operating System**: Ubuntu 20.04+ or Windows 10+ (Ubuntu recommended)
- **Python**: 3.9 or higher
- **CUDA**: 11.8+ (if using GPU acceleration)
- **Git**: Latest version

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/kitchen-assistive-robot.git
cd kitchen-assistive-robot
```

### 2. Set Up Python Environment

#### Using Conda (Recommended)

```bash
# Create new conda environment
conda create -n kitchen-robot python=3.9
conda activate kitchen-robot

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Using Virtual Environment

```bash
# Create virtual environment
python -m venv kitchen-robot-env

# Activate environment
# On Linux/Mac:
source kitchen-robot-env/bin/activate
# On Windows:
kitchen-robot-env\Scripts\activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install additional requirements for development
pip install -r requirements-dev.txt  # if available
```

### 4. Install LIBERO Simulation Environment

```bash
cd sim_env/LIBERO
pip install -e .
cd ../..
```

### 5. Install Kinova SDK (For Real Robot)

If you plan to use the physical Kinova arm:

#### Linux Installation

```bash
# Download Kinova SDK
wget https://artifactory.kinovaapps.com/artifactory/generic-local/kortex/API/2.3.0/kortex_api-2.3.0.post34-py3-none-any.whl

# Install SDK
pip install kortex_api-2.3.0.post34-py3-none-any.whl
```

#### Additional Dependencies for Hardware

```bash
# For camera support
sudo apt-get install v4l-utils
pip install opencv-python

# For audio/speech capabilities
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
pip install pyttsx3

# For PS4 controller support
pip install pyPS4Controller
```

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env  # if example file exists
# Or create manually:
touch .env
```

Edit `.env` with your configuration:

```bash
# Hardware Configuration
ENABLE_CAMERA=true
ENABLE_SPEAKER=true
ENABLE_ARM=false  # Set to true when using real robot

# API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=anthropic/claude-3-opus-20240229

# Alternative: Use OpenAI directly
# OPENAI_API_KEY=your_openai_key_here
# MODEL_NAME=gpt-4-vision-preview

# Camera Configuration (if using multiple cameras)
ENVIRONMENT_CAMERA_INDEX=0
WRIST_CAMERA_INDEX=1

# Robot Configuration
KINOVA_IP_ADDRESS=192.168.1.10  # Your robot's IP
KINOVA_USERNAME=admin
KINOVA_PASSWORD=admin
```

### 7. Download Pre-trained Models (Optional)

If pre-trained models are available:

```bash
# Create models directory
mkdir models

# Download models (replace with actual download links)
wget -O models/libero_baseline.pth https://your-model-hosting/libero_baseline.pth
wget -O models/libero_moe.pth https://your-model-hosting/libero_moe.pth
wget -O models/kinova_baseline.pth https://your-model-hosting/kinova_baseline.pth
```

### 8. Verify Installation

#### Test Basic Functionality

```bash
# Test Python imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Test LIBERO environment
python -c "import libero; print('LIBERO installed successfully')"

# Test AI agent framework
python -m llm_ai_agent.interactive --list-configs
```

#### Test Hardware (If Available)

```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera accessible:', cap.isOpened()); cap.release()"

# Test robot connection (with robot powered on)
python -c "from llm_ai_agent.hardware_tools import test_robot_connection; test_robot_connection()"
```

## Platform-Specific Setup

### Ubuntu/Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran

# For audio support
sudo apt-get install -y alsa-utils pulseaudio
```

### Windows

```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install additional dependencies via conda
conda install -c conda-forge opencv
conda install -c anaconda pyaudio
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake pkg-config
brew install opencv
brew install portaudio
```

## Troubleshooting

### Common Issues

#### CUDA/GPU Issues

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Test PyTorch GPU access
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

#### Camera Access Issues

```bash
# Linux: Check camera permissions
sudo usermod -a -G video $USER
# Log out and back in for changes to take effect

# List available cameras
v4l2-ctl --list-devices  # Linux
# or
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
```

#### Import Errors

```bash
# Ensure you're in the correct environment
conda env list  # or pip list

# Reinstall problematic packages
pip uninstall package_name
pip install package_name
```

#### API Connection Issues

```bash
# Test API connectivity
python -c "
import requests
response = requests.get('https://openrouter.ai/api/v1/models')
print('API accessible:', response.status_code == 200)
"
```

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Verify your hardware meets the minimum requirements
3. Ensure all environment variables are correctly set
4. Check that your API keys are valid and have sufficient credits
5. Create an issue on the project repository with:
   - Your operating system and version
   - Python version and virtual environment details
   - Complete error messages
   - Steps to reproduce the issue

## Next Steps

After successful installation:

1. Follow the [Hardware Setup Guide](hardware.md) if using physical robot
2. Review the [Architecture Overview](architecture.md) to understand the system
3. Try the [Quick Start examples](../README.md#quick-start) to test functionality
4. Read the [Training Guide](training.md) to train your own models