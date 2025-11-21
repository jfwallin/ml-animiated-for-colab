# Notebook Modification Template

This template shows exactly how to modify each notebook for Colab compatibility.

## Step-by-Step Modification Process

### 1. Add Attribution Cell (First Cell)

Add this as a **Markdown cell** at the very top of every notebook:

```markdown
# Chapter X: [Original Chapter Title]

**Original Work:** "Machine Learning, Animated" by Mark Liu
- Original Repository: https://github.com/markhliu/ml_animated
- License: MIT License (2022)

**Colab Adaptation:**
- Modified for Google Colab compatibility
- Optimized for cloud execution
- See [ATTRIBUTION.md](../ATTRIBUTION.md) for detailed modifications

**Chapter Overview:**
[Copy the original chapter description]

**Runtime Estimate:** ~XX minutes
**GPU Required:** Yes/No
**Topics Covered:** [List 3-5 key topics]

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-repo]/ml-animated-colab/blob/main/ChXX_Title.ipynb)
```

### 2. Add Setup Cell (First Code Cell)

Add this as the **first code cell**:

```python
# =========================================================
# SETUP CELL - Run this first!
# =========================================================
# This cell installs all dependencies and sets up the environment
# Runtime: ~1-2 minutes

# Install ML Animated Colab package
!pip install -q git+https://github.com/[your-repo]/ml-animated-colab.git

# Setup environment for this chapter
from utils.colab_helpers import quick_setup
quick_setup(chapter=X)  # Replace X with chapter number

# Verify GPU (for chapters that need it)
from utils.colab_helpers import check_gpu
check_gpu()

print("=" * 60)
print("✓ Setup complete! You're ready to start learning.")
print("=" * 60)
```

### 3. Update All Import Statements

**Before (original):**
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
```

**After (add error handling):**
```python
# Core libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TensorFlow/Keras (with backward compatibility)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Flatten
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Installing TensorFlow...")
    !pip install -q tensorflow
    import tensorflow as tf
    from tensorflow import keras
```

### 4. Update File Paths

**Find and replace all file paths:**

```python
# BEFORE (original relative paths)
'files/ch05/horsedeer.p'
'files/ch12/trained_conn_model.h5'

# AFTER (Colab absolute paths)
'/content/ml_animated/files/ch05/horsedeer.p'
'/content/ml_animated/files/ch12/trained_conn_model.h5'
```

**Better: Use helper function:**
```python
from utils.colab_helpers import COLAB_BASE_PATH, FILES_PATH
import os

# Construct paths dynamically
data_file = os.path.join(FILES_PATH, 'ch05', 'horsedeer.p')
model_file = os.path.join(FILES_PATH, 'ch12', 'trained_conn_model.h5')

# Ensure directory exists before saving
os.makedirs(os.path.dirname(model_file), exist_ok=True)
```

### 5. Fix Rendering Code

**For OpenAI Gym environments:**

```python
# BEFORE (requires display window)
env.render()

# AFTER (Colab-compatible inline rendering)
import matplotlib.pyplot as plt

frame = env.render(mode='rgb_array')
if frame is not None:
    plt.figure(figsize=(8, 6))
    plt.imshow(frame)
    plt.axis('off')
    plt.title(f"Step {step}")
    plt.show()
    plt.close()
```

**Or use helper:**
```python
from utils.colab_helpers import render_gym_env_colab
render_gym_env_colab(env)
```

### 6. Replace Turtle Graphics (Ch10-13 only)

**BEFORE:**
```python
import turtle

screen = turtle.Screen()
screen.setup(width=600, height=600)
t = turtle.Turtle()
t.forward(100)
t.left(90)
```

**AFTER:**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Note for users
print("📝 Note: Turtle graphics are not supported in Colab.")
print("   Using matplotlib visualization instead.")
print("   Game logic remains identical to original.")

def draw_board(state):
    """Draw game board using matplotlib"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw grid
    for i in range(4):
        ax.plot([0, 3], [i, i], 'k-', linewidth=2)
        ax.plot([i, i], [0, 3], 'k-', linewidth=2)

    # Draw state
    # ... (chapter-specific drawing code)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

# Use the function
draw_board(current_state)
```

### 7. Add Training Progress Indicators

**For long-running training loops:**

```python
# BEFORE (no progress indicator)
for episode in range(n_episodes):
    # ... training code

# AFTER (with progress and checkpoints)
from tqdm.notebook import tqdm  # Use notebook version for Colab

checkpoint_interval = 500
total_episodes = n_episodes

for episode in tqdm(range(total_episodes), desc="Training"):
    # ... training code

    # Save checkpoint
    if episode % checkpoint_interval == 0 and episode > 0:
        checkpoint_path = os.path.join(FILES_PATH, f'ch{chapter:02d}',
                                       f'checkpoint_{episode}.h5')
        model.save(checkpoint_path)
        print(f"\n✓ Checkpoint saved at episode {episode}")

print("✓ Training complete!")
```

### 8. Add Quick Mode Option (Ch16-20)

```python
# Add this cell before training
print("Training Mode Selection")
print("=" * 60)
print("QUICK MODE: ~45-90 minutes, reduced dataset")
print("FULL MODE: ~3-6 hours, original dataset")
print("PRE-TRAINED: Load pre-trained model (instant)")
print("=" * 60)

mode = input("Choose mode (quick/full/pretrained): ").lower()

if mode == "pretrained":
    # Load pre-trained model
    from utils.colab_helpers import download_pretrained_model
    model_path = download_pretrained_model(chapter=16, model_name='pong.h5')
    model.load_weights(model_path)
    print("✓ Loaded pre-trained model")
    SKIP_TRAINING = True

elif mode == "quick":
    # Reduced training parameters
    n_episodes = 2000
    buffer_size = 10000
    print("⚡ Quick mode selected")
    SKIP_TRAINING = False

else:  # full
    # Original parameters
    n_episodes = 10000
    buffer_size = 50000
    print("🐌 Full mode selected (this will take several hours)")
    SKIP_TRAINING = False
```

### 9. Add Runtime Warnings

**For computationally intensive chapters:**

```python
# Add this cell at the beginning (after setup)
print("⚠️ RUNTIME WARNING")
print("=" * 60)
print("This chapter involves training a deep neural network.")
print(f"Estimated time: {estimated_time}")
print()
print("Recommendations:")
print("1. Enable GPU: Runtime → Change runtime type → GPU")
print("2. Keep this tab active to prevent disconnection")
print("3. Consider using 'quick mode' or pre-trained models")
print("=" * 60)

# Check GPU
from utils.colab_helpers import check_gpu
has_gpu = check_gpu()

if not has_gpu:
    print("\n⚠️ WARNING: GPU not detected!")
    print("Training will be VERY slow without GPU.")
    print("Enable GPU: Runtime → Change runtime type → GPU")
    proceed = input("\nContinue anyway? (yes/no): ")
    if proceed.lower() != 'yes':
        raise SystemExit("Please enable GPU and restart.")
```

### 10. Add Final Summary Cell

**Add at the end of each notebook:**

```python
print("=" * 60)
print("🎉 Chapter X Complete!")
print("=" * 60)
print()
print("What you learned:")
print("- [Key concept 1]")
print("- [Key concept 2]")
print("- [Key concept 3]")
print()
print("Files created:")
print(f"- {model_path}")
print(f"- {animation_path}")
print()
print("Next steps:")
print("- Try modifying hyperparameters")
print("- Experiment with different architectures")
print("- Move to Chapter X+1")
print()
print("Questions or issues?")
print("- Original work: https://github.com/markhliu/ml_animated")
print("- Colab version: https://github.com/[your-repo]/ml-animated-colab")
print("=" * 60)
```

---

## Complete Example: Chapter 1 Modification

Here's a complete example showing all modifications for Chapter 1:

### Cell 1: Attribution (Markdown)
```markdown
# Chapter 1: Create Animations

**Original Work:** "Machine Learning, Animated" by Mark Liu
- Original Repository: https://github.com/markhliu/ml_animated
- License: MIT License (2022)

**Colab Adaptation:**
- Modified for Google Colab compatibility
- See [ATTRIBUTION.md](../ATTRIBUTION.md) for details

**Chapter Overview:**
Learn to create graphs and plots, convert them into animations, and combine multiple animations into one. This foundational chapter teaches the visualization techniques used throughout the course.

**Runtime Estimate:** ~5 minutes
**GPU Required:** No
**Topics Covered:**
- Creating plots with matplotlib
- Converting plots to images
- Creating GIF animations with imageio
- Combining multiple animations

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-repo]/ml-animated-colab/blob/main/Ch01CreateAnimation.ipynb)
```

### Cell 2: Setup (Code)
```python
# =========================================================
# SETUP CELL - Run this first!
# =========================================================

!pip install -q git+https://github.com/[your-repo]/ml-animated-colab.git

from utils.colab_helpers import quick_setup
quick_setup(chapter=1)

print("✓ Ready to create animations!")
```

### Cell 3: Imports (Code)
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import imageio
import os

# Setup paths
from utils.colab_helpers import FILES_PATH
output_dir = os.path.join(FILES_PATH, 'ch01')
os.makedirs(output_dir, exist_ok=True)

print(f"Files will be saved to: {output_dir}")
```

### Cell 4-N: Original Content (with path updates)
```python
# Original code continues, but with updated paths

# BEFORE
# filename = 'files/ch01/animation.gif'

# AFTER
filename = os.path.join(output_dir, 'animation.gif')

# Rest of original code...
```

### Final Cell: Summary (Code)
```python
print("=" * 60)
print("🎉 Chapter 1 Complete!")
print("=" * 60)
print()
print("What you learned:")
print("- Create plots with matplotlib")
print("- Convert plots to images with PIL")
print("- Create animations with imageio")
print("- Combine multiple animations")
print()
print(f"Files created in: {output_dir}")
print()
print("Next: Chapter 2 - Gradient Descent")
print("=" * 60)
```

---

## Quick Checklist for Each Notebook

When modifying a notebook, verify:

- [ ] Attribution cell added at top (markdown)
- [ ] Setup cell added (first code cell)
- [ ] All imports have error handling
- [ ] All file paths updated to `/content/ml_animated/...`
- [ ] Directory creation added before file saves
- [ ] `env.render()` → `env.render(mode='rgb_array')`
- [ ] Turtle graphics replaced with matplotlib (Ch10-13)
- [ ] Progress bars added for long training
- [ ] Checkpoint saving added (Ch16-20)
- [ ] Quick mode option added (Ch16-20)
- [ ] Runtime warnings added (where appropriate)
- [ ] GPU check added (Ch7+)
- [ ] Summary cell added at end
- [ ] Colab badge updated in attribution cell
- [ ] Tested in fresh Colab session

---

## Chapter-Specific Notes

**Ch1-6:** Minimal changes, mostly paths
**Ch7:** Add reduced dataset option
**Ch8-9:** Update gym rendering
**Ch10-13:** Replace turtle graphics with matplotlib
**Ch14-15:** Update gym rendering, add checkpoints
**Ch16-20:** Add quick mode, pre-trained option, Baselines setup
**Ch21:** Update turtle graphics

---

## After Modification

1. **Test in Colab**
   - Open in fresh Colab session
   - Run all cells
   - Verify outputs

2. **Update Metadata**
   - Ensure kernel is set to Python 3
   - Clear all outputs before committing
   - Add to table of contents in README

3. **Create Colab Badge Link**
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[user]/[repo]/blob/main/ChXX_Title.ipynb)
   ```

4. **Commit to Repository**
   ```bash
   git add ChXX_Title.ipynb
   git commit -m "Add Colab-compatible ChXX: [Title]"
   git push
   ```

---

This template ensures consistency across all 21 chapters while maintaining the educational value of the original work.
