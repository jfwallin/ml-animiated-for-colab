# Lab 1 Modular Notebooks - README

## Overview

Lab 1 has been split into **6 independent modules** that students access via LMS links. Each module is a self-contained Colab notebook focusing on one interactive exercise.

## Module Files

| Module | File | Content | Time | Questions |
|--------|------|---------|------|-----------|
| **0** | `lab_1_module_0_setup.ipynb` | Setup & group code | 3 min | None |
| **1** | `lab_1_module_1_global_error.ipynb` | Error visualization | 5 min | Q1-Q2 |
| **2** | `lab_1_module_2_line_fitting.ipynb` | Interactive sliders | 10-15 min | Q3-Q5 |
| **3** | `lab_1_module_3_parameter_space.ipynb` | Parameter game | 15-20 min | Q6-Q8 |
| **4** | `lab_1_module_4_hidden_function.ipynb` | 1D optimization | 10-15 min | Q9-Q11 |
| **5** | `lab_1_module_5_mountain.ipynb` | 2D optimization | 15-20 min | Q12-Q15 |

**Total Lab Time:** ~60-75 minutes

## Additional Files

- **`lab_1_narrative.ipynb`** - All explanatory text from original notebook (for LMS extraction)

## Student Workflow

### Step 1: Module 0 (Setup)
1. Click link to Module 0
2. Run all cells
3. Enter group code when prompted
4. Remember the group code!
5. Return to LMS

### Steps 2-6: Work Through Modules
For each module (1-5):
1. Click link in LMS
2. Enter **same group code** as Module 0
3. Complete the interactive exercise
4. Note any metrics (attempts, best score, etc.)
5. Return to LMS
6. Answer questions in LMS text boxes

### Step 7: Submit
Click "Submit" button in LMS

## Instructor Setup

### Option A: Google Drive (Recommended)

1. **Upload all modules** to a shared Google Drive folder
2. **For each file:**
   - Right-click → Get link
   - Set to "Anyone with the link can view"
   - Click "Copy link"
3. **Convert to Colab links:**
   - Original: `https://drive.google.com/file/d/FILE_ID/view`
   - Colab link: `https://colab.research.google.com/drive/FILE_ID`
4. **Paste links into LMS page**

### Option B: GitHub

1. **Create a repo** (e.g., `data1010-labs`)
2. **Upload all `.ipynb` files** to repo
3. **Make repo public**
4. **Get raw URLs:**
   - Navigate to file on GitHub
   - Click "Raw" button
   - Copy URL
5. **Convert to Colab links:**
   - `https://colab.research.google.com/github/USERNAME/REPO/blob/main/FILE.ipynb`
6. **Paste links into LMS page**

### Option C: Direct Upload to Colab

1. **Go to** https://colab.research.google.com
2. **Upload each notebook** to "My Drive"
3. **Get shareable links** (same as Option A)

## LMS Page Setup

See [`LMS_PAGE_TEMPLATE.md`](LMS_PAGE_TEMPLATE.md) for a complete example LMS page structure.

### Required Elements

For each module section:
- **📖 Brief explanation** (learning objectives)
- **🔗 Colab link** (opens in new window)
- **📝 Question text boxes** (for student answers)

### Example Module Section (in LMS):

```
## Module 2: Interactive Line Fitting

### Learning Objectives
- Explore how changing parameters affects error
- Understand the relationship between local and global error

### Activity
🔗 [Open Module 2 in Google Colab](YOUR_COLAB_LINK_HERE)

Work through the interactive notebook, then return here to answer questions.

### Questions

**Q3:** How do the residual lines (dashed vertical lines) help you understand local error at each point?
[Large text box]

**Q4:** Can you make the global error small even if a few points have large errors? Explain.
[Large text box]

**Q5:** Predict how an extreme outlier would affect the best-fit line.
[Large text box]
```

## Key Features

### ✅ Module Independence
- Each module can run standalone
- No dependencies between modules (except group code)
- Students can revisit modules anytime

### ✅ Group Code Consistency
- Students enter same group code in each module
- Ensures consistent data across all exercises
- `np.random.seed(group_code)` generates same data

### ✅ LMS Integration
- Answers collected in LMS (native text boxes)
- Timestamps automatic via LMS
- Grades integrate with LMS gradebook
- No file upload/download needed

### ✅ Flexibility
- Can assign subsets of modules
- Can make some modules optional
- Can release modules progressively
- Easy to update individual modules

## Data Collection

### What LMS Tracks Automatically:
- ✅ Submission timestamp
- ✅ Answer text for all questions
- ✅ Time on page (if LMS supports)
- ✅ Module link clicks

### Optional: Collect Attempt Counts
If you want attempt counts from notebooks:

**Add to end of each module:**
```python
print(f"\\n📊 METRICS FOR LMS:")
print(f"Module: {module_number}")
print(f"Attempts: {len(attempt_history)}")  # or mse_history, etc.
print(f"\\nCopy these numbers to the LMS!")
```

Then add a question in LMS:
```
How many attempts did you make in Module 2? ___
```

## Testing Checklist

### Before Deployment:

- [ ] Upload all 6 modules to hosting (Drive/GitHub)
- [ ] Get shareable Colab links for all 6 modules
- [ ] Test each link opens in Colab correctly
- [ ] Test Module 0 workflow (group code → params)
- [ ] Test Modules 1-5 with same group code
- [ ] Verify all interactive widgets work
- [ ] Check all cells run without errors
- [ ] Build LMS page with all links
- [ ] Add all 15 question text boxes to LMS
- [ ] Test LMS submission workflow
- [ ] Verify LMS gradebook integration
- [ ] Do a complete test run: Module 0 → 5 → Submit

### During First Use:

- [ ] Monitor for technical issues
- [ ] Check LMS submission success rate
- [ ] Review student feedback
- [ ] Note any confusing instructions
- [ ] Track completion times

## Troubleshooting

### "Module won't open in Colab"
- Verify link format is correct
- Check file permissions (public/viewable)
- Try opening in incognito window

### "Different data in each module"
- Student used different group codes
- Have them re-run all modules with same code

### "Widget doesn't work"
- Student might need to run cells in order
- Check browser compatibility (Chrome recommended)
- Try refreshing the page

### "Can't submit in LMS"
- Check all required questions are answered
- Verify LMS submission is enabled
- Check student internet connection

## Advantages vs. Single Notebook

| Aspect | Modular Approach ✅ | Single Notebook |
|--------|-------------------|-----------------|
| **Focus** | One exercise at a time | All at once (overwhelming) |
| **Load time** | Fast (small files) | Slow (large file) |
| **Mobile** | Better (smaller) | Poor (too large) |
| **Reusability** | High (mix & match) | Low (all or nothing) |
| **Updates** | Easy (one module) | Risky (affects all) |
| **LMS integration** | Native | Requires export system |
| **Analytics** | Per-module tracking | All or nothing |
| **Flexibility** | Assign subsets | All or nothing |

## Future Enhancements

### Possible Additions:

1. **Pre-checks** - Verify group code matches Module 0
2. **Hints system** - Progressive hints for struggling students
3. **Leaderboards** - (Optional) show class best scores
4. **Module badges** - Visual completion tracking
5. **Jupyter widgets** - Even richer interactivity
6. **Gamification** - Points, achievements, etc.

## License & Attribution

These notebooks are based on the original Lab 1 from DATA 1010, split into modular components for LMS integration.

[Include appropriate attribution here]

---

## Quick Start Summary

1. ✅ **Upload modules** to Google Drive or GitHub
2. ✅ **Get Colab links** for all 6 modules
3. ✅ **Build LMS page** with links + question boxes
4. ✅ **Test workflow** with a group code
5. ✅ **Deploy to students**
6. ✅ **Collect answers** via LMS submission
7. ✅ **Grade in LMS** gradebook

**Time to set up:** ~2-3 hours (first time)
**Time to set up:** ~30 minutes (subsequent labs using same pattern)

You now have a **flexible, modular lab system** ready for LMS deployment!
