# Integrated Answer Collection System - Cells to Add/Modify

This document contains all the exact cells to add or modify in `lab_1_attempt_3.ipynb` to integrate the answer collection system directly into the notebook.

---

## Step 1: Modify Cell 0 (First Markdown Cell)

**REPLACE the current cell 0 with:**

```markdown
# DATA 1010 – Lab 1: Models, Errors, Loss, Optimization, and Learning

**Course:** DATA 1010 – Artificial Intelligence in Action
**Lab 1 Theme:** How machines measure error, optimize models, and "learn" from data.

You will work in **small groups** for this lab. One person should share their screen and run the notebook; everyone should be involved in discussion and decisions.

This lab has two main goals:

1. **Conceptual:** Understand how we measure error, what "loss" means, and how optimization finds better models.
2. **Practical:** Get comfortable running and modifying code in Google Colab.

## ✨ NEW: Answer Questions Directly in This Notebook!

You will answer lab questions using **interactive answer boxes** that appear throughout this notebook. Your answers will be:
- ✅ Saved automatically with timestamps
- ✅ Editable anytime before final submission
- ✅ Exported to a downloadable file at the end

**No separate handout needed!**
```

---

## Step 2: Add New Cell After Group Code (Insert after current cell 11)

**INSERT this new code cell after the group code input:**

```python
# ============================================================================
# ANSWER COLLECTION SYSTEM SETUP
# ============================================================================
# This cell initializes the system for collecting your answers throughout the lab.
# Run this cell once after entering your group code above.

import json
from datetime import datetime
from ipywidgets import Textarea, Button, VBox, HBox, HTML, Output, Layout
from IPython.display import display, clear_output

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("⚠ Not running in Colab - answer export may work differently")

# Configure matplotlib for Colab
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

# Initialize answer storage
lab_answers = {
    "metadata": {
        "lab_number": 1,
        "lab_title": "Models, Errors, Loss, Optimization, and Learning",
        "group_code": group_code,
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

print("=" * 60)
print("✓ Answer collection system initialized!")
print("=" * 60)
print("📝 Answer boxes will appear after each section")
print("💾 Remember to save your answers as you go")
print("📤 You'll export everything at the end of the lab")
print("=" * 60)
```

---

## Step 3: Add Answer Boxes After Section 2

**INSERT this new code cell after cell 17 (after the plot showing true line):**

```python
# ============================================================================
# SECTION 2: ANSWER QUESTIONS
# ============================================================================

display(HTML("""
<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px; margin-top:30px;'>
📝 Section 2 Questions
</h3>
<p style='color:#5f6368;'>Discuss these questions with your group, then type your answers below.</p>
"""))

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

---

## Step 4: Add Answer Boxes After Section 3.1

**INSERT this new code cell after cell 20 (after 3.1 reflection markdown):**

```python
# ============================================================================
# SECTION 3.1: ANSWER QUESTIONS
# ============================================================================

display(HTML("""
<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px; margin-top:30px;'>
📝 Section 3.1 Questions
</h3>
"""))

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

---

## Step 5: Add Answer Boxes After Section 3.3

**INSERT this new code cell after cell 23 (after 3.3 reflection markdown):**

```python
# ============================================================================
# SECTION 3.3: ANSWER QUESTIONS
# ============================================================================

display(HTML("""
<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px; margin-top:30px;'>
📝 Section 3.3 Questions
</h3>
"""))

create_answer_box(
    "Q6",
    "Describe how your guesses for (m, b) moved over time. Did you follow any systematic strategy (e.g., 'move m a bit, then adjust b', or 'search in a grid', etc.)?",
    rows=5
)

create_answer_box(
    "Q7",
    "Look at the MSE landscape plot. How close was your best guess to (a) the approximate global minimum on the grid, and (b) the least-squares solution from the data? What does this tell you about optimizing only based on the global error?",
    rows=6
)

create_answer_box(
    "Q8",
    "In our earlier line-fitting exercise, you could see the data, the line, and the residuals. In this game, you only saw the MSE. How is this situation similar to how many machine learning models are trained, where the algorithm only sees a loss value and not the 'right answer' in a human-readable way?",
    rows=6
)
```

---

## Step 6: Add Answer Boxes After Section 4.1

**INSERT this new code cell after cell 28 (after 4.1 reflection markdown):**

```python
# ============================================================================
# SECTION 4.1: ANSWER QUESTIONS
# ============================================================================

display(HTML("""
<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px; margin-top:30px;'>
📝 Section 4.1 Questions
</h3>
"""))

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
    "Imagine that even the scatter plot of your guesses was hidden, and you only saw the table of (x, f(x)). Would you still be able to find a good minimum? How?",
    rows=5
)
```

---

## Step 7: Add Answer Boxes After Section 5.1

**INSERT this new code cell after cell 32 (after 5.1 reflection markdown):**

```python
# ============================================================================
# SECTION 5.1: ANSWER QUESTIONS
# ============================================================================

display(HTML("""
<h3 style='color:#1967d2; border-bottom:2px solid #1967d2; padding-bottom:5px; margin-top:30px;'>
📝 Section 5.1 Questions
</h3>
"""))

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
    "Compare your best sample to the true global peak shown on the plot. Were you close to the global maximum, or did you end up stuck near a local maximum?",
    rows=5
)

create_answer_box(
    "Q15",
    "Explain how this mountain-peak search is similar to what happens in machine learning when an algorithm is trying to optimize a loss function that has many 'bumps' (local minima or maxima). What risks does a model face if it only explores one region of the loss landscape?",
    rows=6
)
```

---

## Step 8: Add Progress Tracker (Insert before Section 6)

**INSERT this new code cell before Section 6:**

```python
# ============================================================================
# CHECK YOUR PROGRESS
# ============================================================================

display(HTML("""
<h2 style='color:#1967d2; margin-top:40px;'>📊 Check Your Progress</h2>
<p style='color:#5f6368;'>See how many questions you've answered so far.</p>
"""))

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

check_btn = Button(
    description="📊 Check Progress",
    button_style='info',
    layout=Layout(width='200px', height='40px')
)
check_btn.on_click(lambda b: show_progress())
display(check_btn)

# Show progress immediately
show_progress()
```

---

## Step 9: Add Final Export Section (Add as NEW Section 9)

**ADD this as completely new cells at the end of the notebook:**

**First, add a markdown cell:**

```markdown
---

---

---

# Section 9: Export Your Answers for Submission

**IMPORTANT:** This section generates your answer file for grading.

Make sure you've answered all questions before running the export cell below.
```

**Then add this code cell:**

```python
# ============================================================================
# SECTION 9: EXPORT YOUR ANSWERS
# ============================================================================

display(HTML("""
<h2 style='color:#1967d2; border-bottom:3px solid #1967d2; padding-bottom:10px;'>
📤 Export Your Answers for Submission
</h2>
<p style='color:#5f6368; font-size:14px;'>
<b>Important:</b> Make sure you've answered all questions before generating your answer file.
</p>
"""))

# Show final progress
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
        if IN_COLAB:
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
                from google.colab import files
                files.download(txt_filename)
                print(f"✓ Downloaded {txt_filename}")

            def download_json(b):
                from google.colab import files
                files.download(json_filename)
                print(f"✓ Downloaded {json_filename}")

            download_txt_btn.on_click(download_txt)
            download_json_btn.on_click(download_json)

            display(HBox([download_txt_btn, download_json_btn]))
        else:
            print(f"\n📁 Files saved to current directory")

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
        <li>If you need to make changes, re-run the export cell after editing answers</li>
    </ol>
    <p style='color:#856404; margin-bottom:0;'>
        <b>File naming format:</b> Lab1_Answers_GroupXXXX.json (where XXXX is your group code)
    </p>
</div>
"""))
```

---

## Summary of Changes

**Cells Modified:** 1
- Cell 0: Updated introduction to mention inline answers

**New Cells Added:** 8
- After cell 11: Answer collection system setup
- After cell 17: Q1-Q2 answer boxes
- After cell 20: Q3-Q5 answer boxes
- After cell 23: Q6-Q8 answer boxes
- After cell 28: Q9-Q11 answer boxes
- After cell 32: Q12-Q15 answer boxes
- Before Section 6: Progress tracker
- New Section 9: Export functionality

**Total:** 1 modified + 8 new = 9 cell changes

**Result:** Students answer everything in the notebook, export at the end, submit JSON file!

---

## Testing Checklist

After making these changes:

- [ ] Open notebook in fresh Colab session
- [ ] Run all cells from top to bottom
- [ ] Enter a test group code
- [ ] Verify answer boxes appear after each section
- [ ] Test typing and saving an answer
- [ ] Test editing and re-saving an answer
- [ ] Test progress tracker
- [ ] Answer all questions
- [ ] Test export functionality
- [ ] Verify both files download
- [ ] Open JSON file to verify structure
- [ ] Test with instructor grading helper script

---

This integrated approach means students have **everything in one notebook** - no separate files needed!
