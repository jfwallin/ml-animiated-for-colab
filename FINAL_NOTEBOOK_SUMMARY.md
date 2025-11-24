# Lab 1 FINAL Notebook - Complete Summary

## ✅ What Was Created

### Main File
**[lab_1_FINAL.ipynb](lab_1_FINAL.ipynb)** - Production-ready notebook with:
- Simple answer collection system
- Automatic attempt tracking
- Answer timestamps
- Engagement metrics

### Backup Files
- **[lab_1_attempt_3_ORIGINAL.ipynb](lab_1_attempt_3_ORIGINAL.ipynb)** - Original backup
- **[lab_1_attempt_3.ipynb](lab_1_attempt_3.ipynb)** - Unchanged original

### Previous Versions (for reference)
- `lab_1_with_simple_answers.ipynb` - First version (basic answers only)
- `lab_1_with_tracking.ipynb` - Second version (complex tracking)

---

## 🎯 Key Features

### 1. Simple Answer System

Students just edit Python strings:

```python
# Question 1: What does "global error" measure?
answer_1 = """
Type your answer here.
"""

# Timestamp recorded automatically when cell runs
print("✓ Answer 1 saved at 10:30:15 AM")
```

**Why this is great:**
- Zero learning curve
- Just edit text between `"""`
- Press Shift+Enter to save
- Timestamp recorded automatically

### 2. Auto-Tracking of Attempts

**No modification of existing interactive cells needed!**

The notebook already tracks attempts in these variables:
- `attempt_history` - Line fitting attempts (Section 3)
- `mse_history` - Parameter space guesses (Section 3.2)
- `opt_history` - Hidden function attempts (Section 4)
- `samples_2d` - Mountain landscape samples (Section 5)

Before export, one cell automatically calculates:

```python
# Auto-calculate from existing data
section_attempts = {
    "section_3_line_fitting": len(attempt_history),
    "section_3_2_parameter_space": len(mse_history),
    "section_4_hidden_function": len(opt_history),
    "section_5_mountain_landscape": len(samples_2d)
}

total_attempts = sum(section_attempts.values())
```

### 3. Answer Timestamps

Every time a student runs an answer cell:

```python
record_answer_update("Q1")  # Stores current timestamp
```

This tracks when each answer was last updated.

### 4. Rich Export Data

The exported JSON contains:

```json
{
  "metadata": {
    "group_code": 1234,
    "started_at": "2024-11-23T10:30:00",
    "completed_at": "2024-11-23T12:15:00"
  },
  "answers": {
    "Q1": "student answer...",
    "Q2": "student answer...",
    ...
  },
  "answer_timestamps": {
    "Q1": "2024-11-23T10:45:23",
    "Q2": "2024-11-23T10:52:10",
    ...
  },
  "engagement_metrics": {
    "section_attempts": {
      "section_3_line_fitting": 12,
      "section_3_2_parameter_space": 8,
      "section_4_hidden_function": 15,
      "section_5_mountain_landscape": 20
    },
    "total_attempts": 55
  },
  "group_parameters": {
    "line_slope": 1.234567,
    "line_intercept": -2.345678,
    ...
  },
  "questions": {
    "Q1": "full question text...",
    ...
  }
}
```

---

## 📊 What Data You Get

### For Each Student Submission

**Answers:**
- 15 questions with full text responses
- Timestamp for each answer (when last updated)

**Engagement Metrics:**
- Total number of attempts across all sections
- Attempts per section:
  - Line fitting (Section 3)
  - Parameter space optimization (Section 3.2)
  - Hidden function search (Section 4)
  - Mountain landscape search (Section 5)

**Metadata:**
- Group code
- Lab start time
- Lab completion time
- Total lab duration (calculated from start/end)

**Group Parameters:**
- True line slope/intercept (for verification)
- Hidden function parameters
- Number of mountain peaks

### Analysis Possibilities

With this data, you can:

1. **Grade answers** - Use instructor_grading_helper.py to create CSV
2. **Analyze engagement** - How many attempts did groups make?
3. **Identify struggling groups** - Very few attempts might indicate confusion
4. **Identify racing groups** - Very many attempts in short time
5. **Study answer timing** - When did they answer relative to lab start?
6. **Verify group work** - Check if parameters match group code
7. **Research pedagogy** - Correlate attempts with answer quality

---

## 🎓 Student Workflow

### Step 1: Setup (Cells 0-12)
1. Open notebook in Google Colab
2. Run cells 0-11 (setup and group code)
3. Run cell 12 (answer system setup)
4. See "✓ Answer system ready!"

### Step 2: Work Through Lab
1. Read explanations
2. Run interactive code cells
3. Experiment with sliders/widgets
4. Discuss questions with group
5. Edit answer cells and run them
6. See "✓ Answer X saved at [time]"

### Step 3: Export (Section 9)
1. Scroll to Section 9
2. Run auto-tracking cell
3. See engagement summary
4. Run export cell
5. Download TXT and JSON files
6. Submit JSON to LMS

**Total new actions for students:** Edit text, press Shift+Enter. That's it!

---

## 👨‍🏫 Instructor Workflow

### Receive Submissions
Students submit: `Lab1_Answers_GroupXXXX.json`

### Process with Grading Helper
```bash
python instructor_grading_helper.py --lab 1 --directory ./submissions
```

This creates `grading.csv` with columns:
- Group Code
- Questions Answered
- Duration
- Q1_Answer, Q2_Answer, ... Q15_Answer
- Q1_Length, Q2_Length, ... Q15_Length
- Engagement metrics columns
- Group parameters

### Grade in Spreadsheet
1. Open `grading.csv` in Excel/Google Sheets
2. Read answers directly in cells
3. Add grade columns
4. Use formulas if desired (e.g., check length minimums)

### Analyze Engagement (Optional)
```python
import json
import pandas as pd

# Load all submissions
submissions = []
for file in glob.glob("./submissions/*.json"):
    with open(file) as f:
        submissions.append(json.load(f))

# Extract engagement data
df = pd.DataFrame([
    {
        "group": s["metadata"]["group_code"],
        "total_attempts": s["engagement_metrics"]["total_attempts"],
        "section_3": s["engagement_metrics"]["section_attempts"]["section_3_line_fitting"],
        "section_4": s["engagement_metrics"]["section_attempts"]["section_4_hidden_function"],
        # ... etc
    }
    for s in submissions
])

# Analyze
print(df.describe())
df.hist(column="total_attempts")
```

---

## 🔍 Comparison to Previous Versions

| Feature | Version 1 (Simple) | Version 2 (Complex) | Version 3 (FINAL) ✅ |
|---------|-------------------|---------------------|---------------------|
| Answer method | String variables | String variables | String variables |
| Attempt tracking | ❌ None | ✓ Manual hooks | ✓ Automatic |
| Answer timestamps | ❌ None | ✓ Yes | ✓ Yes |
| Modifies interactive cells | ❌ No | ✓ Yes (many) | ❌ No |
| Code complexity | Low | High | Low |
| Student overhead | Minimal | Minimal | Minimal |
| Instructor value | Medium | High | High |
| Maintainability | Easy | Hard | Easy |

**Winner:** Version 3 (FINAL) combines simplicity with rich data!

---

## 📈 Cell Count Summary

- **Original:** 36 cells
- **FINAL:** 60 cells (+24 cells)

### Breakdown of Added Cells

| Section | Added Cells | Purpose |
|---------|------------|---------|
| Cell 0 | Modified | Updated introduction |
| After cell 11 | +1 | Answer system setup |
| Section 2 | +3 | Q1-Q2 answer cells |
| Section 3.1 | +4 | Q3-Q5 answer cells |
| Section 3.3 | +4 | Q6-Q8 answer cells |
| Section 4.1 | +4 | Q9-Q11 answer cells |
| Section 5.1 | +5 | Q12-Q15 answer cells |
| Before export | +1 | Auto-tracking calculation |
| Export section | +2 | Export markdown + code |

**Total:** 1 modified + 24 added = 25 changes

---

## ✅ Testing Checklist

Before deploying to students:

- [ ] Upload `lab_1_FINAL.ipynb` to Google Colab
- [ ] Run all cells top to bottom
- [ ] Enter test group code (e.g., 9999)
- [ ] Verify answer system setup shows "✓ Answer system ready!"
- [ ] Complete Section 3 (line fitting) - make several attempts
- [ ] Edit and run answer_1 cell
- [ ] Verify "✓ Answer 1 saved at [time]" appears
- [ ] Edit answer_1 again and re-run - verify new timestamp
- [ ] Complete all sections with test interactions
- [ ] Answer all 15 questions with test data
- [ ] Run auto-tracking cell before export
- [ ] Verify engagement metrics show correct attempt counts
- [ ] Run export cell
- [ ] Verify both files download
- [ ] Open TXT file - check readability
- [ ] Open JSON file - verify structure includes:
  - [ ] metadata
  - [ ] answers (all 15)
  - [ ] answer_timestamps (all 15)
  - [ ] engagement_metrics with section_attempts
  - [ ] group_parameters
  - [ ] questions
- [ ] Test JSON with instructor_grading_helper.py
- [ ] Verify CSV output looks correct

---

## 📝 Next Steps

1. **Review** [lab_1_FINAL.ipynb](lab_1_FINAL.ipynb) in your IDE
2. **Test** thoroughly in Google Colab
3. **Deploy** to students when ready
4. **Collect** submissions via LMS
5. **Process** with instructor_grading_helper.py
6. **Analyze** engagement metrics for insights

---

## 🎉 Summary

You now have a **production-ready notebook** that:

✅ **Students love:** Simple, intuitive, no learning curve
✅ **Instructors love:** Rich data, easy grading, engagement insights
✅ **Maintainers love:** Clean code, automatic tracking, no complex dependencies

The simple string-based answer system combined with automatic attempt tracking gives you the best of both worlds: **simplicity for students, rich data for instructors!**
