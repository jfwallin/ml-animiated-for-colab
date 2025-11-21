"""
Colab Helper Functions for ML Animated

Utility functions to set up and configure Google Colab environment
for each chapter of Machine Learning, Animated.

Original Work: "Machine Learning, Animated" by Mark Liu
https://github.com/markhliu/ml_animated
License: MIT License (2022)

Colab Adaptation: Helper functions for cloud-based execution
"""

import os
import sys
import subprocess
from pathlib import Path
import requests
from typing import Optional, List
import warnings

# Base paths for Colab environment
COLAB_BASE_PATH = "/content/ml_animated"
FILES_PATH = os.path.join(COLAB_BASE_PATH, "files")
UTILS_PATH = os.path.join(COLAB_BASE_PATH, "utils")
DATA_PATH = os.path.join(COLAB_BASE_PATH, "data")


def is_colab():
    """Check if code is running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def check_gpu():
    """
    Check if GPU is available and provide recommendations.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print("✓ GPU is available and enabled!")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            return True
        else:
            print("⚠ GPU not detected. Running on CPU.")
            if is_colab():
                print("  To enable GPU in Colab:")
                print("  1. Runtime → Change runtime type")
                print("  2. Hardware accelerator → GPU")
                print("  3. Save")
            return False
    except ImportError:
        print("⚠ TensorFlow not installed. Cannot check GPU.")
        return False


def create_directory_structure():
    """
    Create the necessary directory structure for ML Animated.

    Creates:
    - /content/ml_animated/files/ch01/ through ch20/
    - /content/ml_animated/data/
    - /content/ml_animated/utils/
    """
    print("Creating directory structure...")

    # Create base directory
    os.makedirs(COLAB_BASE_PATH, exist_ok=True)

    # Create chapter directories
    for i in range(1, 22):  # Ch01 through Ch21
        chapter_dir = os.path.join(FILES_PATH, f"ch{i:02d}")
        os.makedirs(chapter_dir, exist_ok=True)

    # Create data directories
    os.makedirs(os.path.join(DATA_PATH, "pretrained_models"), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, "small_datasets"), exist_ok=True)

    # Create utils directory
    os.makedirs(UTILS_PATH, exist_ok=True)

    print(f"✓ Created directory structure at {COLAB_BASE_PATH}")


def setup_matplotlib_colab():
    """Configure matplotlib for optimal Colab rendering"""
    try:
        import matplotlib.pyplot as plt

        # Set backend for Colab
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['animation.html'] = 'jshtml'  # For animations in notebooks

        # Use inline backend
        if is_colab():
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython:
                ipython.run_line_magic('matplotlib', 'inline')

        print("✓ Matplotlib configured for Colab")
        return True
    except ImportError:
        print("⚠ Matplotlib not installed")
        return False


def install_package(package_name: str, quiet: bool = True):
    """
    Install a Python package using pip.

    Args:
        package_name: Name of package to install (e.g., 'gym==0.15.7')
        quiet: If True, suppress installation output
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.append(package_name)

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL if quiet else None)
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}")
        return False


def install_baselines():
    """
    Install OpenAI Baselines (required for Ch16-20).
    """
    print("Installing OpenAI Baselines...")
    try:
        # Clone baselines repository
        if not os.path.exists('/content/baselines'):
            subprocess.check_call(
                ['git', 'clone', 'https://github.com/openai/baselines.git', '/content/baselines'],
                stdout=subprocess.DEVNULL
            )

        # Install baselines
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', '-e', '/content/baselines'],
            stdout=subprocess.DEVNULL
        )

        print("✓ OpenAI Baselines installed")
        return True
    except Exception as e:
        print(f"✗ Failed to install Baselines: {e}")
        return False


def install_atari_roms():
    """
    Install Atari ROMs (required for Ch16-20).
    """
    print("Installing Atari ROMs...")
    try:
        # Install atari-py
        install_package('atari-py')

        # Install ROMs
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', 'gym[atari]'],
            stdout=subprocess.DEVNULL
        )

        print("✓ Atari ROMs installed")
        return True
    except Exception as e:
        print(f"✗ Failed to install Atari ROMs: {e}")
        return False


def download_file(url: str, destination: str, description: str = "file"):
    """
    Download a file from URL to destination.

    Args:
        url: URL to download from
        destination: Local path to save file
        description: Description of file for progress messages
    """
    print(f"Downloading {description}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Create parent directory if needed
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Download file
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

        print(f"✓ Downloaded {description}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return False


def download_pretrained_model(chapter: int, model_name: str,
                              github_base_url: Optional[str] = None):
    """
    Download a pre-trained model for a specific chapter.

    Args:
        chapter: Chapter number (1-21)
        model_name: Name of model file (e.g., 'trained_cartpole.h5')
        github_base_url: Base URL for GitHub releases (if None, uses placeholder)
    """
    if github_base_url is None:
        print(f"⚠ GitHub URL not configured. Please set up model hosting.")
        print(f"  Model needed: {model_name} for Chapter {chapter}")
        return False

    url = f"{github_base_url}/ch{chapter:02d}/{model_name}"
    destination = os.path.join(FILES_PATH, f"ch{chapter:02d}", model_name)

    return download_file(url, destination, f"model for Ch{chapter}")


def setup_colab_env(chapter: int, gpu_required: bool = False,
                   install_atari: bool = False):
    """
    One-function setup for a specific chapter.

    Args:
        chapter: Chapter number (1-21)
        gpu_required: If True, check for GPU and warn if not available
        install_atari: If True, install Atari dependencies (Ch16-20)

    Returns:
        bool: True if setup successful
    """
    print(f"=" * 60)
    print(f"Setting up environment for Chapter {chapter}")
    print(f"=" * 60)

    # Check if in Colab
    if not is_colab():
        print("⚠ Not running in Google Colab. Some features may not work.")
        print("  This setup is optimized for Colab environment.")

    # Create directory structure
    create_directory_structure()

    # Setup matplotlib
    setup_matplotlib_colab()

    # Check GPU if required
    if gpu_required:
        has_gpu = check_gpu()
        if not has_gpu:
            print(f"⚠ Chapter {chapter} benefits from GPU acceleration.")
            print("  Consider enabling GPU for better performance.")

    # Install Atari dependencies if needed
    if install_atari:
        if chapter >= 16:
            install_baselines()
            install_atari_roms()
        else:
            print("⚠ Atari installation requested but not required for this chapter")

    # Add utils to Python path
    if UTILS_PATH not in sys.path:
        sys.path.insert(0, UTILS_PATH)

    # Add parent directory to path for imports
    parent_path = os.path.dirname(COLAB_BASE_PATH)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)

    print(f"=" * 60)
    print(f"✓ Setup complete for Chapter {chapter}!")
    print(f"=" * 60)
    return True


def get_chapter_requirements(chapter: int) -> dict:
    """
    Get specific requirements for each chapter.

    Returns:
        dict: Configuration for chapter setup
    """
    configs = {
        1: {"gpu_required": False, "install_atari": False},
        2: {"gpu_required": False, "install_atari": False},
        3: {"gpu_required": False, "install_atari": False},
        4: {"gpu_required": False, "install_atari": False},
        5: {"gpu_required": False, "install_atari": False},
        6: {"gpu_required": False, "install_atari": False},
        7: {"gpu_required": True, "install_atari": False},
        8: {"gpu_required": True, "install_atari": False},
        9: {"gpu_required": True, "install_atari": False},
        10: {"gpu_required": False, "install_atari": False},
        11: {"gpu_required": True, "install_atari": False},
        12: {"gpu_required": True, "install_atari": False},
        13: {"gpu_required": False, "install_atari": False},
        14: {"gpu_required": False, "install_atari": False},
        15: {"gpu_required": True, "install_atari": False},
        16: {"gpu_required": True, "install_atari": True},
        17: {"gpu_required": True, "install_atari": True},
        18: {"gpu_required": True, "install_atari": True},
        19: {"gpu_required": True, "install_atari": True},
        20: {"gpu_required": True, "install_atari": True},
        21: {"gpu_required": False, "install_atari": False},
    }
    return configs.get(chapter, {"gpu_required": False, "install_atari": False})


def create_reduced_dataset(original_data, reduction_factor: float = 0.5,
                          maintain_balance: bool = True):
    """
    Create a smaller version of a dataset for faster training.

    Args:
        original_data: Original dataset (numpy array or similar)
        reduction_factor: Fraction of data to keep (0.5 = 50%)
        maintain_balance: If True, maintain class balance in reduction

    Returns:
        Reduced dataset
    """
    import numpy as np

    if not isinstance(original_data, (list, tuple)):
        # Simple case: just sample
        n_samples = int(len(original_data) * reduction_factor)
        indices = np.random.choice(len(original_data), n_samples, replace=False)
        return original_data[indices]

    # For (X, y) tuples
    X, y = original_data

    if maintain_balance:
        # Stratified sampling
        unique_classes = np.unique(y)
        selected_indices = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            n_samples = int(len(cls_indices) * reduction_factor)
            selected = np.random.choice(cls_indices, n_samples, replace=False)
            selected_indices.extend(selected)

        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)

        return X[selected_indices], y[selected_indices]
    else:
        # Simple random sampling
        n_samples = int(len(X) * reduction_factor)
        indices = np.random.choice(len(X), n_samples, replace=False)
        return X[indices], y[indices]


def render_gym_env_colab(env, mode='rgb_array'):
    """
    Render OpenAI Gym environment for Colab display.

    Args:
        env: OpenAI Gym environment
        mode: Rendering mode (default: 'rgb_array' for Colab)

    Returns:
        Rendered frame (numpy array)
    """
    try:
        import matplotlib.pyplot as plt

        frame = env.render(mode=mode)

        if frame is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(frame)
            plt.axis('off')
            plt.show()
            plt.close()

        return frame
    except Exception as e:
        print(f"⚠ Rendering error: {e}")
        return None


def test_imports():
    """Test that all critical imports work"""
    print("Testing critical imports...")

    packages = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'tensorflow': 'TensorFlow',
        'PIL': 'Pillow',
        'imageio': 'ImageIO',
        'sklearn': 'Scikit-learn',
    }

    all_good = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - not installed")
            all_good = False

    return all_good


# Convenience function for quick setup
def quick_setup(chapter: int):
    """
    Quick setup with auto-configuration based on chapter.

    Args:
        chapter: Chapter number (1-21)
    """
    config = get_chapter_requirements(chapter)
    return setup_colab_env(
        chapter=chapter,
        gpu_required=config['gpu_required'],
        install_atari=config['install_atari']
    )


if __name__ == "__main__":
    # Test the module
    print("ML Animated Colab Helpers")
    print("=" * 60)
    test_imports()
    print("=" * 60)
    print("Ready to use! Import this module in your notebooks.")
