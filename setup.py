"""
Setup configuration for ML Animated - Colab Edition

Original Work: "Machine Learning, Animated" by Mark Liu
https://github.com/markhliu/ml_animated
License: MIT License (2022)

This setup.py enables pip installation:
!pip install git+https://github.com/jfwallin/ml-animiated-for-colab.git
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description_file = this_directory / "README_COLAB.md"

if long_description_file.exists():
    long_description = long_description_file.read_text(encoding='utf-8')
else:
    long_description = "Machine Learning, Animated - Google Colab Edition"

# Read requirements
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    # Fallback requirements
    requirements = [
        'gym==0.15.7',
        'tensorflow>=2.0',
        'matplotlib>=3.0',
        'imageio>=2.0',
        'pandas>=1.0',
        'scikit-learn>=1.0',
        'Pillow>=9.0',
        'cloudpickle~=1.2.0',
        'pyglet<=1.5.0,>=1.4.0',
        'numpy>=1.10.4',
        'requests>=2.20',
    ]

setup(
    name='ml-animated-colab',
    version='2.0.0',
    description='Machine Learning, Animated - Google Colab Edition',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Original author attribution
    author='Mark Liu (Original), Colab Adaptation',
    author_email='',  # Add your email if desired

    # Repository information
    url='https://github.com/jfwallin/ml-animiated-for-colab',

    # License
    license='MIT',

    # Python version requirement
    python_requires='>=3.7',

    # Packages to include
    packages=find_packages(),

    # Include non-Python files
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.yml', '*.yaml'],
        'utils': ['*.py'],
    },

    # Dependencies
    install_requires=requirements,

    # Optional dependencies for advanced features
    extras_require={
        'atari': [
            'gym[atari]==0.15.7',
            'atari-py',
        ],
        'baselines': [
            # Baselines must be installed separately via git
        ],
        'dev': [
            'pytest>=6.0',
            'black',
            'flake8',
        ],
    },

    # Entry points
    entry_points={
        'console_scripts': [
            # Add command-line scripts if needed
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Jupyter',
    ],

    # Keywords
    keywords=[
        'machine-learning',
        'deep-learning',
        'reinforcement-learning',
        'education',
        'google-colab',
        'jupyter',
        'tensorflow',
        'keras',
        'animation',
        'visualization',
    ],

    # Zip safety
    zip_safe=False,
)
