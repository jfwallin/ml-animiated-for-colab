# Lab 1 Answer Collection System - Implementation Guide

**For:** DATA 1010 - Lab 1: Models, Errors, Loss, Optimization, and Learning

**Purpose:** Enable students to answer questions directly in Colab notebook with automatic timestamping and easy export for grading.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Step-by-Step Integration](#step-by-step-integration)
4. [Student Experience](#student-experience)
5. [Instructor Grading Workflow](#instructor-grading-workflow)
6. [Troubleshooting](#troubleshooting)
7. [Appendix: All Question IDs](#appendix-all-question-ids)

---

## Overview

### What This System Does

- **For Students:**
  - Answer questions directly in the notebook (no separate handout)
  - See immediate visual confirmation when answers are saved
  - Edit and re-save answers anytime
  - Automatic timestamp tracking
  - One-click export to downloadable file
  - Progress tracking to know how many questions remain

- **For Instructors:**
  - Receive structured JSON files (easy to process)
  - Get human-readable TXT files as backup
  - Timestamps show when students worked on each question
  - Helper script compiles all submissions into gradebook
  - Can verify group-specific parameters were recorded

### Files Created

1. **`answer_collection_system.py`** - Student-facing answer collection system
2. **`instructor_grading_helper.py`** - Instructor grading automation
3. **`Lab1_Answer_Collection_Implementation_Guide.md`** - This guide

---

## Quick Start

### For Students (What They See)

1. Run setup cells at beginning of notebook
2. Answer boxes appear after each section
3. Type answer, click "💾 Save Answer"
4. At end of lab, click "📥 Generate Answer File"
5. Download both TXT and JSON files
6. Submit JSON file to LMS

**No separate handout needed!**

### For Instructors (What You Do)

1. Add answer collection code to notebook (see below)
2. Students submit JSON files
3. Run grading helper script:
   ```bash
   python instructor_grading_helper.py --lab 1 --directory ./submissions
   ```
4. Get CSV with all answers for grading

---

## Step-by-Step Integration

### Step 1: Add Setup Code to Notebook

**Insert after the group code cell (after cell where students enter group_code):**

```python
# Cell: Initialize Answer Collection System

import json
from datetime import datetime
from ipywidgets import Textarea, Button, VBox, HBox, HTML, Output, Layout
from IPython.display import display, clear_output

# Check Colab environment
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("⚠ Not running in Colab - some features may not work")

# Set up matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

# Initialize answer collection
lab_answers = {
    "metadata": {
        "lab_number": 1,
        "lab_title": "Models, Errors, Loss, Optimization, and Learning",
        "group_code": group_code,  # This assumes group_code was set in previous cell
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    },
    "answers": {},
    "timestamps": {},
    "group_parameters": {},
    "question_texts": {}
}

def create_answer_box(question_id, question_text, rows=3, width='95%'):
    """Create interactive answer box for a question."""

    # Store question text
    lab_answers["question_texts"][question_id] = question_text

    # Question display with styling
    q_html = HTML(
        f"<div style='background:#e8f0fe; padding:12px; margin:15px 0; "
        f"border-left:5px solid #1967d2; border-radius:4px;'>"
        f"<span style='color:#1967d2; font-weight:bold; font-size:14px;'>"
        f"Question {question_id.replace('Q', '')}</span><br/>"
        f"<span style='color:#202124; font-size:14px; line-height:1.5;'>"
        f"{question_text}</span>"
        f"</div>"
    )

    # Get existing answer
    existing_answer = lab_answers["answers"].get(question_id, "")

    # Answer text area
    answer_box = Textarea(
        value=existing_answer,
        placeholder="Type your group's answer here... (Discuss first, then type!)",
        layout=Layout(width=width, height=f'{rows*35}px'),
        style={'description_width': '0px'}
    )

    # Save button
    is_saved = question_id in lab_answers["timestamps"]
    save_btn = Button(
        description="💾 Save Answer" if not is_saved else "✓ Saved",
        button_style='success' if is_saved else 'info',
        tooltip='Click to save your answer with timestamp',
        layout=Layout(width='150px'),
        icon='check' if is_saved else 'save'
    )

    # Status display
    status_html = HTML("")
    if is_saved:
        saved_time = datetime.fromisoformat(lab_answers["timestamps"][question_id])
        status_html.value = (
            f"<span style='color:#188038; font-size:12px; margin-left:10px;'>"
            f"✓ Last saved: {saved_time.strftime('%I:%M:%S %p')}</span>"
        )

    def on_save(btn):
        """Save answer with timestamp."""
        answer_text = answer_box.value.strip()

        if not answer_text:
            status_html.value = (
                "<span style='color:#ea8600; font-size:12px; margin-left:10px;'>"
                "⚠ Answer is empty - type something first!</span>"
            )
            return

        # Save answer
        lab_answers["answers"][question_id] = answer_text
        lab_answers["timestamps"][question_id] = datetime.now().isoformat()

        # Update UI
        save_btn.button_style = 'success'
        save_btn.description = "✓ Saved"
        save_btn.icon = 'check'
        saved_time = datetime.now()
        status_html.value = (
            f"<span style='color:#188038; font-size:12px; margin-left:10px;'>"
            f"✓ Saved at {saved_time.strftime('%I:%M:%S %p')}</span>"
        )

    save_btn.on_click(on_save)

    # Helper text
    helper_html = HTML(
        "<div style='color:#5f6368; font-size:11px; margin-top:5px;'>"
        "💡 Tip: You can edit and re-save your answer anytime before final export"
        "</div>"
    )

    # Display
    display(VBox([
        q_html,
        answer_box,
        HBox([save_btn, status_html]),
        helper_html
    ], layout=Layout(margin='10px 0 20px 0')))

print("✓ Answer collection system initialized")
print("📝 Answer boxes will appear after each section")
```

### Step 2: Add Answer Boxes After Each Section

**After Section 2 (cells 16-17), add:**

```python
# Cell: Answer Questions for Section 2

display(HTML("<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px;'>📝 Section 2 Questions</h3>"))

create_answer_box(
    "Q1",
    "In your own words, what does 'global error' or 'loss' measure in this plot?",
    rows=4
)

create_answer_box(
    "Q2",
    "If we changed the slope or intercept of the line, how would that change the loss?",
    rows=4
)
```

**After Section 3.1 (cell 20), add:**

```python
# Cell: Answer Questions for Section 3.1

display(HTML("<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px;'>📝 Section 3.1 Questions</h3>"))

create_answer_box(
    "Q3",
    "How do the residual lines (the dashed vertical lines) help you understand the local error at each point?",
    rows=4
)

create_answer_box(
    "Q4",
    "Can you make the global error small even if a few points still have relatively large errors? Describe a situation where this happens and why.",
    rows=5
)

create_answer_box(
    "Q5",
    "Suppose you add one very extreme outlier point far away from the others. Predict how this will affect the best-fit line and the global error.",
    rows=5
)
```

**After Section 3.3 (cell 23), add:**

```python
# Cell: Answer Questions for Section 3.3

display(HTML("<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px;'>📝 Section 3.3 Questions</h3>"))

create_answer_box(
    "Q6",
    "Describe how your guesses for (m, b) moved over time. Did you follow any systematic strategy?",
    rows=5
)

create_answer_box(
    "Q7",
    "Look at the MSE landscape plot. How close was your best guess to the global minimum? What does this tell you about optimizing only based on global error?",
    rows=5
)

create_answer_box(
    "Q8",
    "In our earlier line-fitting exercise, you could see the data, line, and residuals. In this game, you only saw the MSE. How is this similar to how ML models are trained?",
    rows=5
)
```

**After Section 4.1 (cell 28), add:**

```python
# Cell: Answer Questions for Section 4.1

display(HTML("<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px;'>📝 Section 4.1 Questions</h3>"))

create_answer_box(
    "Q9",
    "What strategies did your group use to choose new values of x within the allowed range?",
    rows=4
)

create_answer_box(
    "Q10",
    "How did the 'warmer/colder' feedback influence your choices?",
    rows=4
)

create_answer_box(
    "Q11",
    "Imagine even the scatter plot was hidden, and you only saw the table of (x, f(x)). Would you still be able to find a good minimum? How?",
    rows=5
)
```

**After Section 5.1 (cell 32), add:**

```python
# Cell: Answer Questions for Section 5.1

display(HTML("<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px;'>📝 Section 5.1 Questions</h3>"))

create_answer_box(
    "Q12",
    "Describe your group's strategy for choosing new (x, y) locations. How did you decide where to sample next after finding a high point?",
    rows=5
)

create_answer_box(
    "Q13",
    "Look at the revealed landscape. How many local peaks can you see? Did your group spend most of its time near one peak, or did you explore multiple regions?",
    rows=5
)

create_answer_box(
    "Q14",
    "Compare your best sample to the true global peak. Were you close to the global maximum, or did you end up stuck near a local maximum?",
    rows=5
)

create_answer_box(
    "Q15",
    "Explain how this mountain-peak search is similar to ML optimization with many 'bumps' in the loss landscape. What risks does a model face?",
    rows=6
)
```

### Step 3: Add Progress Tracker (Optional but Recommended)

**Add after Section 3 or 4:**

```python
# Cell: Check Your Progress

def show_progress():
    """Display progress on answering questions."""
    total_questions = len(lab_answers["question_texts"])
    answered = len([a for a in lab_answers["answers"].values() if a.strip()])

    if total_questions == 0:
        print("⚠ No questions displayed yet")
        return

    progress_pct = (answered / total_questions) * 100

    progress_html = f"""
    <div style='margin: 20px 0; font-family: sans-serif;'>
        <h3 style='color:#202124;'>📊 Your Progress</h3>
        <div style='background: #e8eaed; border-radius: 8px; height: 35px; position: relative;'>
            <div style='background: linear-gradient(90deg, #1967d2, #188038);
                        width: {progress_pct}%; height: 100%;
                        display: flex; align-items: center; justify-content: center;'>
                <span style='color: white; font-weight: bold;'>
                    {answered}/{total_questions} ({progress_pct:.0f}%)
                </span>
            </div>
        </div>
        <p style='margin-top: 12px; color:#5f6368;'>
            ✓ Answered: <b>{answered}</b> questions<br/>
            ⏳ Remaining: <b>{total_questions - answered}</b> questions
        </p>
    </div>
    """

    display(HTML(progress_html))

    # Show missing questions
    if answered < total_questions:
        missing = []
        for q_id in lab_answers["question_texts"].keys():
            if q_id not in lab_answers["answers"] or not lab_answers["answers"][q_id].strip():
                missing.append(q_id)

        if missing:
            missing_sorted = sorted(missing, key=lambda x: int(x.replace('Q','')))
            print(f"⚠ Unanswered: {', '.join(missing_sorted[:10])}")
            if len(missing) > 10:
                print(f"   ... and {len(missing) - 10} more")

check_btn = Button(description="📊 Check Progress", button_style='info')
check_btn.on_click(lambda b: show_progress())
display(check_btn)
```

### Step 4: Add Export Section at End

**Add as new Section 9 (or after Section 6/8):**

```python
# Cell: Section 9 - Export Your Answers

display(HTML("""
<h2 style='color:#1967d2; border-bottom:3px solid #1967d2; padding-bottom:10px;'>
📤 Section 9: Export Your Answers for Submission
</h2>
<p style='color:#5f6368; font-size:14px;'>
<b>Important:</b> Make sure you've answered all questions before generating your answer file.
</p>
"""))

# Show progress first
show_progress()

# Generate answer file button
export_btn = Button(
    description="📥 Generate Answer File",
    button_style='warning',
    tooltip='Create downloadable file with all your answers',
    layout=Layout(width='300px', height='50px')
)

output_area = Output()

def export_answers(btn):
    """Generate and download answer files."""
    with output_area:
        clear_output()

        print("Generating answer files...")

        # Update metadata
        lab_answers["metadata"]["completed_at"] = datetime.now().isoformat()

        # Store revealed parameters
        lab_answers["group_parameters"] = {
            "line_slope": float(true_m),
            "line_intercept": float(true_b),
            "hidden_func_a": float(a),
            "hidden_func_b": float(b_param),
            "hidden_func_c": float(c_param),
            "num_mountain_peaks": num_peaks
        }

        # Count answered
        total = len(lab_answers["question_texts"])
        answered = len([a for a in lab_answers["answers"].values() if a.strip()])

        print(f"Questions answered: {answered}/{total}")

        if answered < total:
            print(f"\n⚠ Warning: {total - answered} questions are unanswered")
            print("  You can still export, but consider completing all questions first\n")

        # Generate text file
        text_output = f"""
{'='*80}
DATA 1010 - Lab 1: Models, Errors, Loss, Optimization, and Learning
{'='*80}

GROUP INFORMATION:
  Group Code: {group_code}
  Lab Started: {lab_answers["metadata"]["started_at"]}
  Lab Completed: {lab_answers["metadata"]["completed_at"]}
  Questions Answered: {answered}/{total}

{'='*80}
ANSWERS:
{'='*80}

"""

        # Add each answer
        for i in range(1, total + 1):
            q_id = f"Q{i}"
            if q_id not in lab_answers["question_texts"]:
                continue

            question_text = lab_answers["question_texts"][q_id]
            timestamp = lab_answers["timestamps"].get(q_id, "Not answered")
            answer = lab_answers["answers"].get(q_id, "[No answer provided]")

            text_output += f"\n{q_id}: {question_text}\n"
            text_output += f"Timestamp: {timestamp}\n"
            text_output += f"Answer:\n{answer}\n"
            text_output += f"{'-'*80}\n"

        # Add group parameters
        text_output += f"\n\n{'='*80}\n"
        text_output += "GROUP-SPECIFIC PARAMETERS:\n"
        text_output += "(For instructor verification)\n"
        text_output += f"{'='*80}\n"
        for key, value in lab_answers["group_parameters"].items():
            text_output += f"  {key}: {value:.6f}\n"

        # Save files
        txt_filename = f"Lab1_Answers_Group{group_code}.txt"
        json_filename = f"Lab1_Answers_Group{group_code}.json"

        with open(txt_filename, "w", encoding='utf-8') as f:
            f.write(text_output)

        with open(json_filename, "w", encoding='utf-8') as f:
            json.dump(lab_answers, f, indent=2)

        print(f"\n✅ Answer files generated successfully!")
        print(f"\nFiles created:")
        print(f"  1. {txt_filename} (human-readable)")
        print(f"  2. {json_filename} (for grading system)")

        # Download buttons
        print("\n📥 Click the buttons below to download your files:")

        download_txt_btn = Button(
            description="📥 Download TXT",
            button_style='primary',
            layout=Layout(width='200px', height='40px')
        )

        download_json_btn = Button(
            description="📥 Download JSON",
            button_style='success',
            layout=Layout(width='200px', height='40px')
        )

        def download_txt(b):
            try:
                from google.colab import files
                files.download(txt_filename)
                print(f"✓ Downloaded {txt_filename}")
            except:
                print("Download initiated (or not in Colab)")

        def download_json(b):
            try:
                from google.colab import files
                files.download(json_filename)
                print(f"✓ Downloaded {json_filename}")
            except:
                print("Download initiated (or not in Colab)")

        download_txt_btn.on_click(download_txt)
        download_json_btn.on_click(download_json)

        display(HBox([download_txt_btn, download_json_btn]))

        # Show preview
        print("\n" + "="*80)
        print("PREVIEW OF YOUR ANSWERS:")
        print("="*80)
        print(text_output[:1000])
        if len(text_output) > 1000:
            print("\n... (preview truncated, full content in downloaded file)")

export_btn.on_click(export_answers)

display(export_btn)
display(output_area)

# Instructions
display(HTML("""
<div style='background:#fff3cd; border-left:5px solid #ffc107; padding:15px; margin:20px 0; border-radius:4px;'>
    <h4 style='color:#856404; margin-top:0;'>📤 Submission Instructions</h4>
    <ol style='color:#856404; line-height:1.8;'>
        <li><b>Download BOTH files</b> (TXT and JSON) using the buttons above</li>
        <li><b>Submit the JSON file</b> to your course LMS/Canvas</li>
        <li><b>Keep the TXT file</b> for your own records</li>
        <li><b>Verify the file name</b> includes your correct group code</li>
        <li>If you need to make changes, re-run the export cell after editing</li>
    </ol>
    <p style='color:#856404; margin-bottom:0;'>
        <b>File naming format:</b> Lab1_Answers_GroupXXXX.json (where XXXX is your group code)
    </p>
</div>
"""))
```

---

## Student Experience

### What Students See

1. **Setup Phase:**
   - Run first few cells to initialize notebook
   - Enter group code
   - See "✓ Answer collection system initialized"

2. **During Lab:**
   - After completing each section, see blue answer boxes
   - Read question
   - Discuss with group
   - Type answer in text box
   - Click "💾 Save Answer"
   - See green "✓ Saved at XX:XX:XX" confirmation

3. **Checking Progress:**
   - Click "📊 Check Progress" button anytime
   - See progress bar and count
   - See list of unanswered questions

4. **Final Export:**
   - Navigate to Section 9
   - Click "📥 Generate Answer File"
   - See summary of answers
   - Click download buttons for TXT and JSON
   - Submit JSON to LMS

### Student FAQ

**Q: Can I edit my answers after saving?**
A: Yes! Just change the text and click "💾 Save Answer" again. The timestamp will update.

**Q: What if I accidentally close the notebook?**
A: Your answers are stored in the notebook session. If you restart the runtime, you'll need to answer again. Save your notebook file (`File → Save`) frequently!

**Q: Do I need to save after every word I type?**
A: No, type your complete answer, then click save once.

**Q: What's the difference between TXT and JSON files?**
A: TXT is human-readable (for your records). JSON is for the grading system (submit this one).

**Q: Can I see my answers before downloading?**
A: Yes, a preview appears when you generate the file.

---

## Instructor Grading Workflow

### Step 1: Collect Submissions

Students submit JSON files to your LMS. Download all files to a folder (e.g., `./lab1_submissions/`).

### Step 2: Process with Grading Helper

**Option A: Command Line**

```bash
python instructor_grading_helper.py --lab 1 --directory ./lab1_submissions --output lab1_grading.csv
```

**Option B: Python Script/Jupyter**

```python
from instructor_grading_helper import GradingHelper

# Create helper
helper = GradingHelper(lab_number=1)

# Load submissions
helper.load_submissions('./lab1_submissions')

# Create grading spreadsheet
df = helper.create_grading_spreadsheet('lab1_grading.csv')

# Check statistics
helper.check_completion_stats()
helper.check_submission_times()
```

### Step 3: Grade in Spreadsheet

Open `lab1_grading.csv` in Excel/Google Sheets:

- Each row is a group
- Columns: Group Code, Timestamps, Q1_Answer, Q2_Answer, etc.
- Add grade columns as needed
- Can use formulas to calculate total points

### Step 4: Export Individual Reports (Optional)

```python
helper.export_individual_reports('./grading_reports')
```

Creates individual TXT files for each group with space for comments.

### Advanced: Check for Plagiarism

```python
# Check Q1 for similar answers
helper.find_similar_answers("Q1", threshold=0.8)
```

Shows pairs of groups with >80% similar answers (basic string matching).

---

## Troubleshooting

### Common Student Issues

**Issue:** "Widgets not displaying"
**Solution:**
```python
# Add to setup cell
!pip install ipywidgets
```

**Issue:** "Can't download files"
**Solution:**
- Make sure running in Colab (not local Jupyter)
- Check pop-up blocker settings
- Try running export cell again

**Issue:** "My answers disappeared after restart"
**Solution:**
- Runtime restart clears memory
- Always download files before closing session
- Consider saving notebook file regularly

**Issue:** "Export button doesn't work"
**Solution:**
- Make sure all previous cells ran successfully
- Check that `true_m`, `true_b`, etc. are defined
- Look for error messages in output

### Common Instructor Issues

**Issue:** "Grading helper can't find files"
**Solution:**
```python
# Check pattern matches your files
import glob
files = glob.glob('./lab1_submissions/Lab1_Answers_*.json')
print(files)
```

**Issue:** "CSV has weird characters"
**Solution:**
- Open with UTF-8 encoding
- In Excel: Data → Get External Data → From Text → UTF-8

**Issue:** "Some submissions have missing timestamps"
**Solution:**
- Students might have skipped save button
- Answer is still there, just no timestamp
- Can still grade normally

---

## Appendix: All Question IDs

For Lab 1, questions are:

```
Q1  - Section 2: Global error meaning
Q2  - Section 2: Effect of changing parameters
Q3  - Section 3.1: Residuals and local error
Q4  - Section 3.1: Small global error with large local errors
Q5  - Section 3.1: Outlier effect prediction
Q6  - Section 3.3: Strategy in parameter space
Q7  - Section 3.3: Closeness to global minimum
Q8  - Section 3.3: ML connection - loss only optimization
Q9  - Section 4.1: 1D optimization strategy
Q10 - Section 4.1: Warmer/colder feedback influence
Q11 - Section 4.1: Optimization without visualization
Q12 - Section 5.1: 2D sampling strategy
Q13 - Section 5.1: Number of local peaks
Q14 - Section 5.1: Global vs local maximum
Q15 - Section 5.1: Connection to ML local minima
```

Additional questions for Post-Lab (if added):
```
Q16 - Why square errors?
Q17 - Relationship between error, loss, optimization
Q18 - What four activities taught about optimization
Q19 - Transfer to house prices example
Q20 - Real-world optimization example
Q21 - What's still confusing?
Q22 - What would help understanding?
Q23 - Preview of gradient descent
```

---

## Testing Checklist

Before deploying to students:

- [ ] Test in fresh Colab session
- [ ] Verify all answer boxes display
- [ ] Test save functionality (multiple times)
- [ ] Test editing saved answers
- [ ] Test progress tracker
- [ ] Test export with all questions answered
- [ ] Test export with some questions unanswered
- [ ] Verify file downloads work
- [ ] Check JSON structure is valid
- [ ] Test grading helper with sample files
- [ ] Verify CSV opens correctly in Excel
- [ ] Test with group code containing special characters

---

## Maintenance Notes

**To add more questions:**
1. Add `create_answer_box("QXX", "Question text", rows=4)` after appropriate section
2. Update total question count in progress tracker
3. Update appendix list in this guide

**To change styling:**
- Modify HTML strings in `create_answer_box` function
- Color codes: `#1967d2` (blue), `#188038` (green), `#ea8600` (orange)

**To add features:**
- Auto-save (add timer callback)
- Cloud backup (Google Drive API)
- Real-time collaboration (Firebase)

---

## Summary

This system provides:
✅ Easy answer collection (no separate handout)
✅ Automatic timestamps
✅ Edit capability
✅ Progress tracking
✅ One-click export
✅ Instructor grading automation

**Students save time, instructors save time, everyone wins!**

---

**Questions?** Contact [Instructor Name/Email]

**Version:** 1.0
**Last Updated:** 2024
