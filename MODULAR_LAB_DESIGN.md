# Modular Lab 1 Design - LMS-Integrated Approach

## Overview

Break Lab 1 into **separate, focused notebooks** that students access via LMS links. Each module covers one interactive exercise with minimal narrative.

## Structure

### LMS Page: "Lab 1: Models, Errors, Loss, Optimization"

The LMS page contains:
- **Narrative text** (learning objectives, explanations, instructions)
- **Links to Colab notebooks** (one per exercise)
- **Embedded answer boxes** (in LMS, not in notebooks)
- **Submission button** (at end of LMS page)

### Notebook Files (Colab)

Each notebook is:
- **Self-contained** - Can run independently
- **Focused** - One main interactive exercise
- **Minimal text** - Just essential instructions
- **No answer collection** - Students answer in LMS

---

## Proposed Module Breakdown

### Module 0: Setup and Group Code
**File:** `lab_1_module_0_setup.ipynb`

**Content:**
- Brief intro
- Group code input
- Data generation
- Store parameters for later verification

**LMS Integration:**
- First link on LMS page
- Students run once at start
- Download their group parameters file

---

### Module 1: Understanding Global Error
**File:** `lab_1_module_1_global_error.ipynb`

**Content:**
- Generate data with true line (visible)
- Plot with residuals
- Calculate SSE
- Show global error value

**Interactive Element:**
- None (visualization only)

**LMS Questions (after module):**
- Q1: What does global error measure?
- Q2: How would changing slope/intercept affect loss?

**Duration:** ~5 minutes

---

### Module 2: Interactive Line Fitting
**File:** `lab_1_module_2_line_fitting.ipynb`

**Content:**
- Interactive sliders for m and b
- Real-time plot with residuals
- SSE feedback
- Warm/cold feedback
- Attempt history table

**Interactive Element:**
- Sliders to adjust line
- Try to minimize error

**LMS Questions (after module):**
- Q3: How do residual lines help understand local error?
- Q4: Can global error be small with some large individual errors?
- Q5: How would an outlier affect the best-fit line?

**Duration:** ~10-15 minutes

---

### Module 3: Parameter Space Optimization
**File:** `lab_1_module_3_parameter_space.ipynb`

**Content:**
- Submit guess interface (m, b sliders + button)
- MSE feedback only (no data visible)
- History of guesses in (m, b) space
- Color-coded by MSE
- "Done" button reveals landscape

**Interactive Element:**
- Guess (m, b) values using only MSE feedback
- Click "Done" to see full landscape

**LMS Questions (after module):**
- Q6: Describe your strategy for choosing (m, b)
- Q7: How close was your best guess to the true minimum?
- Q8: How is this similar to ML training?

**Duration:** ~15-20 minutes

---

### Module 4: Hidden Function Optimization
**File:** `lab_1_module_4_hidden_function.ipynb`

**Content:**
- Slider for x in [-10, 10]
- Button to evaluate f(x)
- Plot of guesses only (not full function)
- Warm/cold feedback
- Attempt history

**Interactive Element:**
- Choose x values
- Try to minimize f(x)

**LMS Questions (after module):**
- Q9: What strategies did you use to choose x values?
- Q10: How did warm/cold feedback help?
- Q11: Could you find the minimum with only a table?

**Duration:** ~10-15 minutes

---

### Module 5: Mountain Landscape Search
**File:** `lab_1_module_5_mountain.ipynb`

**Content:**
- 2D sliders for (x, y)
- Button to sample altitude
- Plot of samples (color = altitude)
- Sample history table
- "Done" button reveals full landscape

**Interactive Element:**
- Sample (x, y) locations
- Try to find highest peak

**LMS Questions (after module):**
- Q12: Describe your sampling strategy
- Q13: How many local peaks? Where did you explore?
- Q14: How close were you to the global peak?
- Q15: How does this relate to ML optimization challenges?

**Duration:** ~15-20 minutes

---

### Module 6: Narrative Content
**File:** `lab_1_narrative.ipynb`

**Content:**
- All the explanatory text from original notebook
- Conceptual explanations
- "What is a model?" section
- "Understanding error" section
- "From error to learning" section
- No interactive elements

**Purpose:**
- Extract text for LMS pages
- Reference for instructors
- Can provide to students as reading material

---

## File Structure

```
lab_1_modules/
├── lab_1_module_0_setup.ipynb              # Setup & group code
├── lab_1_module_1_global_error.ipynb       # Visualization only
├── lab_1_module_2_line_fitting.ipynb       # Interactive sliders
├── lab_1_module_3_parameter_space.ipynb    # Parameter game
├── lab_1_module_4_hidden_function.ipynb    # 1D optimization
├── lab_1_module_5_mountain.ipynb           # 2D optimization
├── lab_1_narrative.ipynb                   # Text content for LMS
└── README_MODULAR.md                       # Instructions
```

---

## LMS Page Structure

### Example LMS Page Layout

```markdown
# Lab 1: Models, Errors, Loss, Optimization, and Learning

## Introduction
[Explanatory text about the lab goals]

## Pre-Lab: Setup (Do this first!)

**Before class, complete Module 0:**

🔗 [Open Module 0: Setup in Colab](link-to-module-0)

**Instructions:**
1. Click the link above
2. Run all cells
3. Enter your group code when prompted
4. Download your group parameters file
5. Upload the file here: [File upload box in LMS]

---

## Part 1: Understanding Global Error

### Learning Objectives
- Understand what "global error" means
- See how error is calculated across all data points

### Activity

🔗 [Open Module 1: Global Error in Colab](link-to-module-1)

Run through the notebook and observe the visualization.

### Questions

**Q1:** In your own words, what does "global error" or "loss" measure?
[Text box in LMS]

**Q2:** If we changed the slope or intercept of the line, how would that change the loss?
[Text box in LMS]

---

## Part 2: Interactive Line Fitting

### Learning Objectives
- Explore how changing parameters affects error
- Understand the relationship between local and global error

### Activity

🔗 [Open Module 2: Line Fitting in Colab](link-to-module-2)

Use the sliders to adjust the line and try to minimize the error.

### Questions

**Q3:** How do the residual lines help you understand local error at each point?
[Text box in LMS]

**Q4:** Can you make the global error small even if a few points have large errors?
[Text box in LMS]

**Q5:** How would an extreme outlier affect the best-fit line?
[Text box in LMS]

---

[Continue for all modules...]

---

## Submission

Click the Submit button below to submit your answers.
[Submit Button]

```

---

## Advantages of Modular Approach

### For Students
✅ **Focus** - One concept at a time
✅ **Flexibility** - Can pause between modules
✅ **Less overwhelming** - Smaller notebooks
✅ **Clear progress** - Can see what's done/remaining in LMS
✅ **Better mobile experience** - Smaller notebooks load faster

### For Instructors
✅ **Reusability** - Mix and match modules across different labs
✅ **Analytics** - LMS tracks which modules are completed
✅ **Partial credit** - Can grade module by module
✅ **Easy updates** - Update one module without affecting others
✅ **LMS integration** - Native answer collection and timestamps
✅ **Gradebook integration** - Automatic scoring if desired

### For Course Design
✅ **Flexibility** - Can assign subsets of modules
✅ **Scaffolding** - Can release modules progressively
✅ **Differentiation** - Advanced students can skip early modules
✅ **Assessment** - Can make some modules graded, others practice

---

## Data Collection in LMS

The LMS automatically tracks:
- **Submission timestamp** - When student submitted answers
- **Time on page** - How long they spent (if LMS supports)
- **Module completion** - Which Colab links were clicked
- **Attempt history** - If students re-submit

To get engagement from notebooks:
- Each module exports a small JSON with attempt count
- Students upload alongside answers
- Or: embed attempt count in final answer submission

---

## Implementation Strategy

### Phase 1: Create Modules (Priority Order)

1. ✅ **Module 0** (Setup) - Essential for all others
2. ✅ **Module 2** (Line Fitting) - Core interactive experience
3. ✅ **Module 3** (Parameter Space) - Most complex, test early
4. ✅ **Module 5** (Mountain) - Similar to Module 3
5. ✅ **Module 4** (Hidden Function) - Similar to Module 2
6. ✅ **Module 1** (Global Error) - Simple visualization
7. ✅ **Narrative notebook** - Extract text for LMS

### Phase 2: LMS Integration

1. Create LMS page structure
2. Upload notebooks to Colab (or GitHub for Colab links)
3. Generate shareable Colab links
4. Embed links in LMS
5. Add answer text boxes in LMS
6. Configure submission/grading

### Phase 3: Testing

1. Test each module independently
2. Test full workflow: Module 0 → ... → Module 5
3. Test LMS answer submission
4. Verify grading workflow

---

## Technical Considerations

### Group Code Consistency

**Challenge:** Each module is separate - how to maintain group code?

**Solutions:**

**Option A: Download/Upload Pattern**
```python
# In Module 0
import json
group_data = {
    "group_code": group_code,
    "true_m": true_m,
    "true_b": true_b,
    ...
}
# Download file
```

```python
# In Modules 1-5
from google.colab import files
uploaded = files.upload()  # Upload the group data file
import json
group_data = json.load(open(list(uploaded.keys())[0]))
group_code = group_data["group_code"]
```

**Option B: Simple Re-entry**
```python
# In each module
group_code = int(input("Enter your group code: "))
np.random.seed(group_code)
# Regenerate same data
```

**Recommended:** Option B (simpler for students)

### Shared Functions

**Challenge:** Avoid code duplication across modules

**Solution:** Small utility cell at top of each notebook
```python
# Standard setup cell (copy-paste to all modules)
import numpy as np
import matplotlib.pyplot as plt

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

# Module-specific code follows...
```

---

## Example: Module 2 Structure

```python
# Cell 1: Title
# Module 2: Interactive Line Fitting

# Cell 2: Setup
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Checkbox
import pandas as pd

group_code = int(input("Enter your group code: "))
np.random.seed(group_code)

# Cell 3: Generate Data
true_m = np.random.uniform(-3, 3)
true_b = np.random.uniform(-5, 5)
x = np.linspace(-5, 5, 25)
noise = np.random.normal(0, 1.0, size=len(x))
y = true_m * x + true_b + noise

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

print("✓ Data generated for your group")

# Cell 4: Instructions
print("""
INSTRUCTIONS:
1. Use the sliders below to adjust the slope (m) and intercept (b)
2. Try to minimize the Global Error (SSE)
3. Pay attention to the warm/cold feedback
4. The notebook tracks how many attempts you make
5. When done, return to the LMS to answer questions
""")

# Cell 5: Interactive Widget
attempt_history = []

def plot_guess(m, b, show_residuals=True):
    global attempt_history

    y_pred = m * x + b
    loss = sse(y, y_pred)

    prev_loss = attempt_history[-1]["loss"] if attempt_history else None
    attempt_history.append({"m": m, "b": b, "loss": loss})

    # Plotting code...

interact(...)

# Cell 6: Export Attempt Count (Optional)
print(f"You made {len(attempt_history)} attempts")
print("Remember this number for the LMS questions!")

# Or export small JSON
import json
module_data = {
    "module": 2,
    "group_code": group_code,
    "attempts": len(attempt_history)
}
with open(f"module_2_group_{group_code}.json", "w") as f:
    json.dump(module_data, f)
print(f"✓ Saved: module_2_group_{group_code}.json")
```

---

## Next Steps

1. **Create Module 0** (setup) first
2. **Create Module 2** (line fitting) as prototype
3. **Test the workflow** with both modules
4. **Refine the pattern** based on testing
5. **Create remaining modules** using established pattern
6. **Extract narrative** to separate notebook
7. **Build LMS page** structure
8. **Full integration test**

This modular approach gives maximum flexibility while maintaining simplicity for students!
