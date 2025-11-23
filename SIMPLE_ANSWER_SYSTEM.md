# Simple Answer Collection System - Direct String Variables

This document describes the SIMPLE approach to collecting answers directly in the notebook using basic Python string variables.

**Philosophy:** Students should be able to just type their answer between triple quotes. No widgets, no buttons, no complexity.

---

## The Simple Approach

### Basic Pattern:

```python
# Question 1: What does global error measure?
answer_1 = """
Put your answer here.
You can type multiple lines.
Just edit this text directly.
"""
```

That's it! Students just:
1. See the question in a comment
2. Type their answer between the triple quotes
3. Re-run the cell if they want to update it

---

## Complete Implementation

### Step 1: Modify Cell 0 (Introduction)

**REPLACE current cell 0 with:**

```markdown
# DATA 1010 – Lab 1: Models, Errors, Loss, Optimization, and Learning

**Course:** DATA 1010 – Artificial Intelligence in Action
**Lab 1 Theme:** How machines measure error, optimize models, and "learn" from data.

You will work in **small groups** for this lab. One person should share their screen and run the notebook; everyone should be involved in discussion and decisions.

This lab has two main goals:

1. **Conceptual:** Understand how we measure error, what "loss" means, and how optimization finds better models.
2. **Practical:** Get comfortable running and modifying code in Google Colab.

## ✨ Answering Questions

Throughout this lab, you'll see cells that look like this:

```python
# Question 1: What is loss?
answer_1 = """
Type your group's answer here between the triple quotes.
"""
```

To answer:
- Just click in the cell and edit the text between `"""`
- Run the cell (Shift+Enter) to save your changes
- You can edit and re-run anytime

At the end, you'll export all your answers to a file for submission.
```

---

### Step 2: Simple Setup Cell (after group code)

**INSERT this cell after the group code input (after cell 11):**

```python
# ============================================================================
# ANSWER COLLECTION SETUP
# ============================================================================
# This cell initializes the answer collection system.
# Run this cell once after entering your group code.

import json
from datetime import datetime

# Store metadata about this lab session
lab_metadata = {
    "lab_number": 1,
    "lab_title": "Models, Errors, Loss, Optimization, and Learning",
    "group_code": group_code,
    "started_at": datetime.now().isoformat()
}

print("=" * 60)
print("✓ Answer system ready!")
print("=" * 60)
print(f"Group Code: {group_code}")
print(f"Started at: {datetime.now().strftime('%I:%M:%S %p')}")
print()
print("📝 Answer questions by editing the answer_X variables")
print("📤 Export your answers at the end of the lab")
print("=" * 60)
```

---

### Step 3: Section 2 Questions

**INSERT after cell 17 (after the plot showing true line):**

```python
# ============================================================================
# SECTION 2: ANSWER QUESTIONS
# ============================================================================

print("="*70)
print("📝 SECTION 2 QUESTIONS")
print("="*70)
print()
print("Discuss these with your group, then edit the answer variables below.")
print("Run this cell (Shift+Enter) after typing each answer.")
print()
```

**Then INSERT a NEW cell:**

```python
# Question 1: In your own words, what does "global error" or "loss"
# measure in this plot?

answer_1 = """
Type your answer here.
"""

print("✓ Answer 1 saved")
```

**Then INSERT another NEW cell:**

```python
# Question 2: If we changed the slope or intercept of the line,
# how would that change the loss?

answer_2 = """
Type your answer here.
"""

print("✓ Answer 2 saved")
```

---

### Step 4: Section 3.1 Questions

**INSERT after cell 20 (after 3.1 reflection markdown):**

```python
print("="*70)
print("📝 SECTION 3.1 QUESTIONS")
print("="*70)
```

**Then three separate cells:**

```python
# Question 3: How do the residual lines (the dashed vertical lines)
# help you understand the local error at each point?

answer_3 = """
Type your answer here.
"""

print("✓ Answer 3 saved")
```

```python
# Question 4: Can you make the global error small even if a few points
# still have relatively large errors? Describe a situation where this
# happens and why.

answer_4 = """
Type your answer here.
"""

print("✓ Answer 4 saved")
```

```python
# Question 5: Suppose you add one very extreme outlier point far away
# from the others. Predict how this will affect the best-fit line and
# the global error.

answer_5 = """
Type your answer here.
"""

print("✓ Answer 5 saved")
```

---

### Step 5: Section 3.3 Questions

**INSERT after cell 23 (after 3.3 reflection):**

```python
print("="*70)
print("📝 SECTION 3.3 QUESTIONS")
print("="*70)
```

**Then three cells:**

```python
# Question 6: Describe how your guesses for (m, b) moved over time.
# Did you follow any systematic strategy?

answer_6 = """
Type your answer here.
"""

print("✓ Answer 6 saved")
```

```python
# Question 7: Look at the MSE landscape plot. How close was your best
# guess to (a) the approximate global minimum on the grid, and
# (b) the least-squares solution from the data? What does this tell you
# about optimizing only based on the global error?

answer_7 = """
Type your answer here.
"""

print("✓ Answer 7 saved")
```

```python
# Question 8: In our earlier line-fitting exercise, you could see the data,
# the line, and the residuals. In this game, you only saw the MSE. How is
# this situation similar to how many machine learning models are trained?

answer_8 = """
Type your answer here.
"""

print("✓ Answer 8 saved")
```

---

### Step 6: Section 4.1 Questions

**INSERT after cell 28:**

```python
print("="*70)
print("📝 SECTION 4.1 QUESTIONS")
print("="*70)
```

**Then three cells:**

```python
# Question 9: What strategies did your group use to choose new values
# of x within the allowed range?

answer_9 = """
Type your answer here.
"""

print("✓ Answer 9 saved")
```

```python
# Question 10: How did the "warmer/colder" feedback influence your choices?

answer_10 = """
Type your answer here.
"""

print("✓ Answer 10 saved")
```

```python
# Question 11: Imagine that even the scatter plot of your guesses was hidden,
# and you only saw the table of (x, f(x)). Would you still be able to find
# a good minimum? How?

answer_11 = """
Type your answer here.
"""

print("✓ Answer 11 saved")
```

---

### Step 7: Section 5.1 Questions

**INSERT after cell 32:**

```python
print("="*70)
print("📝 SECTION 5.1 QUESTIONS")
print("="*70)
```

**Then four cells:**

```python
# Question 12: Describe your group's strategy for choosing new (x, y) locations.
# How did you decide where to sample next after finding a high point?

answer_12 = """
Type your answer here.
"""

print("✓ Answer 12 saved")
```

```python
# Question 13: Look at the revealed landscape. How many local peaks can you see?
# Did your group spend most of its time near one peak, or did you explore
# multiple regions?

answer_13 = """
Type your answer here.
"""

print("✓ Answer 13 saved")
```

```python
# Question 14: Compare your best sample to the true global peak shown on the plot.
# Were you close to the global maximum, or did you end up stuck near a local maximum?

answer_14 = """
Type your answer here.
"""

print("✓ Answer 14 saved")
```

```python
# Question 15: Explain how this mountain-peak search is similar to what happens
# in machine learning when an algorithm is trying to optimize a loss function
# that has many "bumps" (local minima or maxima). What risks does a model face
# if it only explores one region of the loss landscape?

answer_15 = """
Type your answer here.
"""

print("✓ Answer 15 saved")
```

---

### Step 8: Export Section (Add as new Section 9 at end)

**Add markdown cell:**

```markdown
---

---

---

# Section 9: Export Your Answers for Submission

Run the cell below to generate your answer file for grading.
```

**Add code cell:**

```python
# ============================================================================
# EXPORT YOUR ANSWERS
# ============================================================================

print("="*70)
print("📤 EXPORTING YOUR ANSWERS")
print("="*70)
print()

# Collect all answers
all_answers = {
    "metadata": {
        "lab_number": lab_metadata["lab_number"],
        "lab_title": lab_metadata["lab_title"],
        "group_code": lab_metadata["group_code"],
        "started_at": lab_metadata["started_at"],
        "completed_at": datetime.now().isoformat()
    },
    "answers": {
        "Q1": answer_1.strip(),
        "Q2": answer_2.strip(),
        "Q3": answer_3.strip(),
        "Q4": answer_4.strip(),
        "Q5": answer_5.strip(),
        "Q6": answer_6.strip(),
        "Q7": answer_7.strip(),
        "Q8": answer_8.strip(),
        "Q9": answer_9.strip(),
        "Q10": answer_10.strip(),
        "Q11": answer_11.strip(),
        "Q12": answer_12.strip(),
        "Q13": answer_13.strip(),
        "Q14": answer_14.strip(),
        "Q15": answer_15.strip(),
    },
    "group_parameters": {
        "line_slope": float(true_m),
        "line_intercept": float(true_b),
        "hidden_func_a": float(a),
        "hidden_func_b": float(b_param),
        "hidden_func_c": float(c_param),
        "num_mountain_peaks": num_peaks
    },
    "questions": {
        "Q1": "In your own words, what does 'global error' or 'loss' measure in this plot?",
        "Q2": "If we changed the slope or intercept of the line, how would that change the loss?",
        "Q3": "How do the residual lines (the dashed vertical lines) help you understand the local error at each point?",
        "Q4": "Can you make the global error small even if a few points still have relatively large errors? Describe a situation where this happens and why.",
        "Q5": "Suppose you add one very extreme outlier point far away from the others. Predict how this will affect the best-fit line and the global error.",
        "Q6": "Describe how your guesses for (m, b) moved over time. Did you follow any systematic strategy?",
        "Q7": "Look at the MSE landscape plot. How close was your best guess to (a) the approximate global minimum on the grid, and (b) the least-squares solution from the data? What does this tell you about optimizing only based on the global error?",
        "Q8": "In our earlier line-fitting exercise, you could see the data, the line, and the residuals. In this game, you only saw the MSE. How is this situation similar to how many machine learning models are trained?",
        "Q9": "What strategies did your group use to choose new values of x within the allowed range?",
        "Q10": "How did the 'warmer/colder' feedback influence your choices?",
        "Q11": "Imagine that even the scatter plot of your guesses was hidden, and you only saw the table of (x, f(x)). Would you still be able to find a good minimum? How?",
        "Q12": "Describe your group's strategy for choosing new (x, y) locations. How did you decide where to sample next after finding a high point?",
        "Q13": "Look at the revealed landscape. How many local peaks can you see? Did your group spend most of its time near one peak, or did you explore multiple regions?",
        "Q14": "Compare your best sample to the true global peak shown on the plot. Were you close to the global maximum, or did you end up stuck near a local maximum?",
        "Q15": "Explain how this mountain-peak search is similar to what happens in machine learning when an algorithm is trying to optimize a loss function that has many 'bumps' (local minima or maxima). What risks does a model face if it only explores one region of the loss landscape?"
    }
}

# Count answered questions
total = 15
answered = len([a for a in all_answers["answers"].values() if a and a != "Type your answer here."])

print(f"Questions answered: {answered}/{total}")
if answered < total:
    print(f"⚠ Warning: {total - answered} questions still have default text")
    print()

# Generate human-readable text file
text_output = f"""
{'='*80}
DATA 1010 - Lab 1: Models, Errors, Loss, Optimization, and Learning
{'='*80}

GROUP INFORMATION:
  Group Code: {lab_metadata['group_code']}
  Lab Started: {lab_metadata['started_at']}
  Lab Completed: {all_answers['metadata']['completed_at']}
  Questions Answered: {answered}/{total}

{'='*80}
ANSWERS:
{'='*80}

"""

for i in range(1, total + 1):
    q_id = f"Q{i}"
    text_output += f"\n{q_id}: {all_answers['questions'][q_id]}\n"
    text_output += f"Answer:\n{all_answers['answers'][q_id]}\n"
    text_output += f"{'-'*80}\n"

text_output += f"\n\n{'='*80}\n"
text_output += "GROUP-SPECIFIC PARAMETERS:\n"
text_output += "(For instructor verification)\n"
text_output += f"{'='*80}\n"
for key, value in all_answers["group_parameters"].items():
    text_output += f"  {key}: {value:.6f}\n"

# Save files
txt_filename = f"Lab1_Answers_Group{lab_metadata['group_code']}.txt"
json_filename = f"Lab1_Answers_Group{lab_metadata['group_code']}.json"

with open(txt_filename, "w", encoding='utf-8') as f:
    f.write(text_output)

with open(json_filename, "w", encoding='utf-8') as f:
    json.dump(all_answers, f, indent=2)

print(f"✅ Files generated successfully!")
print(f"  1. {txt_filename} (human-readable)")
print(f"  2. {json_filename} (for grading system)")
print()

# Download if in Colab
try:
    from google.colab import files
    print("📥 Downloading files...")
    files.download(txt_filename)
    files.download(json_filename)
    print("✓ Files downloaded!")
except ImportError:
    print("📁 Files saved to current directory")
    print("   (Not in Colab, so no automatic download)")

print()
print("="*70)
print("📤 SUBMISSION INSTRUCTIONS")
print("="*70)
print("1. Download BOTH files (TXT and JSON)")
print("2. Submit the JSON file to your course LMS")
print("3. Keep the TXT file for your records")
print("="*70)
```

---

## Summary

**Key Benefits of This Simple Approach:**

1. **Minimal Overhead** - Students just edit text between `"""`
2. **Familiar Pattern** - Like editing any Python variable
3. **No Complex Widgets** - No buttons, no UI elements to understand
4. **Easy to Update** - Just edit and re-run the cell
5. **Works Everywhere** - Colab, Jupyter, JupyterLab, etc.
6. **Visual Clarity** - Students see exactly what they typed
7. **No State Issues** - The answer IS the variable, no hidden state

**For Students:**
```python
# This is all they need to know:
answer_1 = """
My answer goes here.
"""
```

**Total Cells to Add:** ~20 cells (much simpler cells than the widget approach)
- 1 setup cell
- 15 answer cells (one per question)
- 1 export cell

**Complexity:** MINIMAL - just Python string variables!
