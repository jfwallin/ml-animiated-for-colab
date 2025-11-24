"""
Script to add simple answer collection system to lab_1_attempt_3.ipynb
"""

import json
import copy
import sys

# Fix Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def create_code_cell(source_lines):
    """Create a code cell from a list of source lines."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def create_markdown_cell(source_lines):
    """Create a markdown cell from a list of source lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

# Read the original notebook
with open(r'c:\Users\jfwal\OneDrive\python-experiments-2023\ml-animated\ml_animated\lab_1_with_simple_answers.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Original notebook has {len(nb['cells'])} cells")

# ============================================================================
# STEP 1: Modify Cell 0 (Introduction)
# ============================================================================

new_intro = [
    "# DATA 1010 – Lab 1: Models, Errors, Loss, Optimization, and Learning\n",
    "\n",
    "**Course:** DATA 1010 – Artificial Intelligence in Action  \n",
    "**Lab 1 Theme:** How machines measure error, optimize models, and \"learn\" from data.  \n",
    "\n",
    "You will work in **small groups** for this lab. One person should share their screen and run the notebook; everyone should be involved in discussion and decisions.\n",
    "\n",
    "This lab has two main goals:\n",
    "\n",
    "1. **Conceptual:** Understand how we measure error, what \"loss\" means, and how optimization finds better models.\n",
    "2. **Practical:** Get comfortable running and modifying code in Google Colab.\n",
    "\n",
    "## ✨ NEW: Answer Questions Directly in This Notebook!\n",
    "\n",
    "You will answer lab questions using **simple Python variables** that appear throughout this notebook.\n",
    "\n",
    "To answer a question, just edit the text between the triple quotes:\n",
    "\n",
    "```python\n",
    "# Question 1: What is loss?\n",
    "answer_1 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "```\n",
    "\n",
    "Then run the cell (Shift+Enter) to save it. You can edit and re-run anytime!\n",
    "\n",
    "At the end of the lab, you'll export all your answers to a file for submission.\n",
    "\n",
    "**No separate handout needed!**"
]

nb['cells'][0]['source'] = new_intro
print("✓ Modified cell 0 (introduction)")

# ============================================================================
# STEP 2: Insert answer system setup after cell 11 (group code)
# ============================================================================

setup_cell = create_code_cell([
    "# ============================================================================\n",
    "# ANSWER COLLECTION SETUP\n",
    "# ============================================================================\n",
    "# This cell initializes the answer collection system.\n",
    "# Run this cell once after entering your group code.\n",
    "\n",
    "import json\n",
    "from datetime import datetime\n",
    "\n",
    "# Store metadata about this lab session\n",
    "lab_metadata = {\n",
    "    \"lab_number\": 1,\n",
    "    \"lab_title\": \"Models, Errors, Loss, Optimization, and Learning\",\n",
    "    \"group_code\": group_code,\n",
    "    \"started_at\": datetime.now().isoformat()\n",
    "}\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"✓ Answer system ready!\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"Group Code: {group_code}\")\n",
    "print(f\"Started at: {datetime.now().strftime('%I:%M:%S %p')}\")\n",
    "print()\n",
    "print(\"📝 Answer questions by editing the answer_X variables\")\n",
    "print(\"📤 Export your answers at the end of the lab\")\n",
    "print(\"=\" * 60)"
])

nb['cells'].insert(12, setup_cell)
print("✓ Inserted answer system setup after cell 11")

# ============================================================================
# STEP 3: Add Section 2 answer cells (after cell 17, now 18 due to insert)
# ============================================================================

section2_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 2: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(\"📝 SECTION 2 QUESTIONS\")\n",
    "print(\"=\"*70)\n",
    "print()\n",
    "print(\"Discuss these with your group, then edit the answer variables below.\")\n",
    "print(\"Run each cell (Shift+Enter) after typing your answer.\")\n",
    "print()"
])

answer1_cell = create_code_cell([
    "# Question 1: In your own words, what does \"global error\" or \"loss\"\n",
    "# measure in this plot?\n",
    "\n",
    "answer_1 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 1 saved\")"
])

answer2_cell = create_code_cell([
    "# Question 2: If we changed the slope or intercept of the line,\n",
    "# how would that change the loss?\n",
    "\n",
    "answer_2 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 2 saved\")"
])

# Insert after cell 18 (was cell 17, +1 for setup insert)
nb['cells'].insert(19, section2_header)
nb['cells'].insert(20, answer1_cell)
nb['cells'].insert(21, answer2_cell)
print("✓ Inserted Section 2 answer cells")

# ============================================================================
# STEP 4: Add Section 3.1 answer cells (after cell 20, now 24)
# ============================================================================

section31_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 3.1: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(\"📝 SECTION 3.1 QUESTIONS\")\n",
    "print(\"=\"*70)\n",
    "print()"
])

answer3_cell = create_code_cell([
    "# Question 3: How do the residual lines (the dashed vertical lines)\n",
    "# help you understand the local error at each point?\n",
    "\n",
    "answer_3 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 3 saved\")"
])

answer4_cell = create_code_cell([
    "# Question 4: Can you make the global error small even if a few points\n",
    "# still have relatively large errors? Describe a situation where this\n",
    "# happens and why.\n",
    "\n",
    "answer_4 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 4 saved\")"
])

answer5_cell = create_code_cell([
    "# Question 5: Suppose you add one very extreme outlier point far away\n",
    "# from the others. Predict how this will affect the best-fit line and\n",
    "# the global error.\n",
    "\n",
    "answer_5 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 5 saved\")"
])

# Insert after cell 24 (was 20, +4 for previous inserts)
nb['cells'].insert(25, section31_header)
nb['cells'].insert(26, answer3_cell)
nb['cells'].insert(27, answer4_cell)
nb['cells'].insert(28, answer5_cell)
print("✓ Inserted Section 3.1 answer cells")

# ============================================================================
# STEP 5: Add Section 3.3 answer cells (after cell 23, now 32)
# ============================================================================

section33_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 3.3: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(\"📝 SECTION 3.3 QUESTIONS\")\n",
    "print(\"=\"*70)\n",
    "print()"
])

answer6_cell = create_code_cell([
    "# Question 6: Describe how your guesses for (m, b) moved over time.\n",
    "# Did you follow any systematic strategy?\n",
    "\n",
    "answer_6 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 6 saved\")"
])

answer7_cell = create_code_cell([
    "# Question 7: Look at the MSE landscape plot. How close was your best\n",
    "# guess to (a) the approximate global minimum on the grid, and\n",
    "# (b) the least-squares solution from the data? What does this tell you\n",
    "# about optimizing only based on the global error?\n",
    "\n",
    "answer_7 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 7 saved\")"
])

answer8_cell = create_code_cell([
    "# Question 8: In our earlier line-fitting exercise, you could see the data,\n",
    "# the line, and the residuals. In this game, you only saw the MSE. How is\n",
    "# this situation similar to how many machine learning models are trained?\n",
    "\n",
    "answer_8 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 8 saved\")"
])

# Insert after cell 32 (was 23, +9 for previous inserts)
nb['cells'].insert(33, section33_header)
nb['cells'].insert(34, answer6_cell)
nb['cells'].insert(35, answer7_cell)
nb['cells'].insert(36, answer8_cell)
print("✓ Inserted Section 3.3 answer cells")

# ============================================================================
# STEP 6: Add Section 4.1 answer cells (after cell 28, now 41)
# ============================================================================

section41_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 4.1: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(\"📝 SECTION 4.1 QUESTIONS\")\n",
    "print(\"=\"*70)\n",
    "print()"
])

answer9_cell = create_code_cell([
    "# Question 9: What strategies did your group use to choose new values\n",
    "# of x within the allowed range?\n",
    "\n",
    "answer_9 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 9 saved\")"
])

answer10_cell = create_code_cell([
    "# Question 10: How did the \"warmer/colder\" feedback influence your choices?\n",
    "\n",
    "answer_10 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 10 saved\")"
])

answer11_cell = create_code_cell([
    "# Question 11: Imagine that even the scatter plot of your guesses was hidden,\n",
    "# and you only saw the table of (x, f(x)). Would you still be able to find\n",
    "# a good minimum? How?\n",
    "\n",
    "answer_11 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 11 saved\")"
])

# Insert after cell 41 (was 28, +13 for previous inserts)
nb['cells'].insert(42, section41_header)
nb['cells'].insert(43, answer9_cell)
nb['cells'].insert(44, answer10_cell)
nb['cells'].insert(45, answer11_cell)
print("✓ Inserted Section 4.1 answer cells")

# ============================================================================
# STEP 7: Add Section 5.1 answer cells (after cell 32, now 49)
# ============================================================================

section51_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 5.1: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(\"📝 SECTION 5.1 QUESTIONS\")\n",
    "print(\"=\"*70)\n",
    "print()"
])

answer12_cell = create_code_cell([
    "# Question 12: Describe your group's strategy for choosing new (x, y) locations.\n",
    "# How did you decide where to sample next after finding a high point?\n",
    "\n",
    "answer_12 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 12 saved\")"
])

answer13_cell = create_code_cell([
    "# Question 13: Look at the revealed landscape. How many local peaks can you see?\n",
    "# Did your group spend most of its time near one peak, or did you explore\n",
    "# multiple regions?\n",
    "\n",
    "answer_13 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 13 saved\")"
])

answer14_cell = create_code_cell([
    "# Question 14: Compare your best sample to the true global peak shown on the plot.\n",
    "# Were you close to the global maximum, or did you end up stuck near a local maximum?\n",
    "\n",
    "answer_14 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 14 saved\")"
])

answer15_cell = create_code_cell([
    "# Question 15: Explain how this mountain-peak search is similar to what happens\n",
    "# in machine learning when an algorithm is trying to optimize a loss function\n",
    "# that has many \"bumps\" (local minima or maxima). What risks does a model face\n",
    "# if it only explores one region of the loss landscape?\n",
    "\n",
    "answer_15 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "print(\"✓ Answer 15 saved\")"
])

# Insert after cell 49 (was 32, +17 for previous inserts)
nb['cells'].insert(50, section51_header)
nb['cells'].insert(51, answer12_cell)
nb['cells'].insert(52, answer13_cell)
nb['cells'].insert(53, answer14_cell)
nb['cells'].insert(54, answer15_cell)
print("✓ Inserted Section 5.1 answer cells")

# ============================================================================
# STEP 8: Add export section at the end
# ============================================================================

export_markdown = create_markdown_cell([
    "\n",
    "---\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "# Section 9: Export Your Answers for Submission\n",
    "\n",
    "**IMPORTANT:** This section generates your answer file for grading.\n",
    "\n",
    "Make sure you've answered all questions before running the export cell below."
])

export_cell = create_code_cell([
    "# ============================================================================\n",
    "# EXPORT YOUR ANSWERS\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(\"📤 EXPORTING YOUR ANSWERS\")\n",
    "print(\"=\"*70)\n",
    "print()\n",
    "\n",
    "# Collect all answers\n",
    "all_answers = {\n",
    "    \"metadata\": {\n",
    "        \"lab_number\": lab_metadata[\"lab_number\"],\n",
    "        \"lab_title\": lab_metadata[\"lab_title\"],\n",
    "        \"group_code\": lab_metadata[\"group_code\"],\n",
    "        \"started_at\": lab_metadata[\"started_at\"],\n",
    "        \"completed_at\": datetime.now().isoformat()\n",
    "    },\n",
    "    \"answers\": {\n",
    "        \"Q1\": answer_1.strip(),\n",
    "        \"Q2\": answer_2.strip(),\n",
    "        \"Q3\": answer_3.strip(),\n",
    "        \"Q4\": answer_4.strip(),\n",
    "        \"Q5\": answer_5.strip(),\n",
    "        \"Q6\": answer_6.strip(),\n",
    "        \"Q7\": answer_7.strip(),\n",
    "        \"Q8\": answer_8.strip(),\n",
    "        \"Q9\": answer_9.strip(),\n",
    "        \"Q10\": answer_10.strip(),\n",
    "        \"Q11\": answer_11.strip(),\n",
    "        \"Q12\": answer_12.strip(),\n",
    "        \"Q13\": answer_13.strip(),\n",
    "        \"Q14\": answer_14.strip(),\n",
    "        \"Q15\": answer_15.strip(),\n",
    "    },\n",
    "    \"group_parameters\": {\n",
    "        \"line_slope\": float(true_m),\n",
    "        \"line_intercept\": float(true_b),\n",
    "        \"hidden_func_a\": float(a),\n",
    "        \"hidden_func_b\": float(b_param),\n",
    "        \"hidden_func_c\": float(c_param),\n",
    "        \"num_mountain_peaks\": num_peaks\n",
    "    },\n",
    "    \"questions\": {\n",
    "        \"Q1\": \"In your own words, what does 'global error' or 'loss' measure in this plot?\",\n",
    "        \"Q2\": \"If we changed the slope or intercept of the line, how would that change the loss?\",\n",
    "        \"Q3\": \"How do the residual lines (the dashed vertical lines) help you understand the local error at each point?\",\n",
    "        \"Q4\": \"Can you make the global error small even if a few points still have relatively large errors? Describe a situation where this happens and why.\",\n",
    "        \"Q5\": \"Suppose you add one very extreme outlier point far away from the others. Predict how this will affect the best-fit line and the global error.\",\n",
    "        \"Q6\": \"Describe how your guesses for (m, b) moved over time. Did you follow any systematic strategy?\",\n",
    "        \"Q7\": \"Look at the MSE landscape plot. How close was your best guess to (a) the approximate global minimum on the grid, and (b) the least-squares solution from the data? What does this tell you about optimizing only based on the global error?\",\n",
    "        \"Q8\": \"In our earlier line-fitting exercise, you could see the data, the line, and the residuals. In this game, you only saw the MSE. How is this situation similar to how many machine learning models are trained?\",\n",
    "        \"Q9\": \"What strategies did your group use to choose new values of x within the allowed range?\",\n",
    "        \"Q10\": \"How did the 'warmer/colder' feedback influence your choices?\",\n",
    "        \"Q11\": \"Imagine that even the scatter plot of your guesses was hidden, and you only saw the table of (x, f(x)). Would you still be able to find a good minimum? How?\",\n",
    "        \"Q12\": \"Describe your group's strategy for choosing new (x, y) locations. How did you decide where to sample next after finding a high point?\",\n",
    "        \"Q13\": \"Look at the revealed landscape. How many local peaks can you see? Did your group spend most of its time near one peak, or did you explore multiple regions?\",\n",
    "        \"Q14\": \"Compare your best sample to the true global peak shown on the plot. Were you close to the global maximum, or did you end up stuck near a local maximum?\",\n",
    "        \"Q15\": \"Explain how this mountain-peak search is similar to what happens in machine learning when an algorithm is trying to optimize a loss function that has many 'bumps' (local minima or maxima). What risks does a model face if it only explores one region of the loss landscape?\"\n",
    "    }\n",
    "}\n",
    "\n",
    "# Count answered questions\n",
    "total = 15\n",
    "answered = len([a for a in all_answers[\"answers\"].values() if a and a != \"Type your answer here.\"])\n",
    "\n",
    "print(f\"Questions answered: {answered}/{total}\")\n",
    "if answered < total:\n",
    "    print(f\"⚠ Warning: {total - answered} questions still have default text\")\n",
    "    print()\n",
    "\n",
    "# Generate human-readable text file\n",
    "text_output = f\"\"\"\n",
    "{'='*80}\n",
    "DATA 1010 - Lab 1: Models, Errors, Loss, Optimization, and Learning\n",
    "{'='*80}\n",
    "\n",
    "GROUP INFORMATION:\n",
    "  Group Code: {lab_metadata['group_code']}\n",
    "  Lab Started: {lab_metadata['started_at']}\n",
    "  Lab Completed: {all_answers['metadata']['completed_at']}\n",
    "  Questions Answered: {answered}/{total}\n",
    "\n",
    "{'='*80}\n",
    "ANSWERS:\n",
    "{'='*80}\n",
    "\n",
    "\"\"\"\n",
    "\n",
    "for i in range(1, total + 1):\n",
    "    q_id = f\"Q{i}\"\n",
    "    text_output += f\"\\n{q_id}: {all_answers['questions'][q_id]}\\n\"\n",
    "    text_output += f\"Answer:\\n{all_answers['answers'][q_id]}\\n\"\n",
    "    text_output += f\"{'-'*80}\\n\"\n",
    "\n",
    "text_output += f\"\\n\\n{'='*80}\\n\"\n",
    "text_output += \"GROUP-SPECIFIC PARAMETERS:\\n\"\n",
    "text_output += \"(For instructor verification)\\n\"\n",
    "text_output += f\"{'='*80}\\n\"\n",
    "for key, value in all_answers[\"group_parameters\"].items():\n",
    "    text_output += f\"  {key}: {value:.6f}\\n\"\n",
    "\n",
    "# Save files\n",
    "txt_filename = f\"Lab1_Answers_Group{lab_metadata['group_code']}.txt\"\n",
    "json_filename = f\"Lab1_Answers_Group{lab_metadata['group_code']}.json\"\n",
    "\n",
    "with open(txt_filename, \"w\", encoding='utf-8') as f:\n",
    "    f.write(text_output)\n",
    "\n",
    "with open(json_filename, \"w\", encoding='utf-8') as f:\n",
    "    json.dump(all_answers, f, indent=2)\n",
    "\n",
    "print(f\"✅ Files generated successfully!\")\n",
    "print(f\"  1. {txt_filename} (human-readable)\")\n",
    "print(f\"  2. {json_filename} (for grading system)\")\n",
    "print()\n",
    "\n",
    "# Download if in Colab\n",
    "try:\n",
    "    from google.colab import files\n",
    "    print(\"📥 Downloading files...\")\n",
    "    files.download(txt_filename)\n",
    "    files.download(json_filename)\n",
    "    print(\"✓ Files downloaded!\")\n",
    "except ImportError:\n",
    "    print(\"📁 Files saved to current directory\")\n",
    "    print(\"   (Not in Colab, so no automatic download)\")\n",
    "\n",
    "print()\n",
    "print(\"=\"*70)\n",
    "print(\"📤 SUBMISSION INSTRUCTIONS\")\n",
    "print(\"=\"*70)\n",
    "print(\"1. Download BOTH files (TXT and JSON)\")\n",
    "print(\"2. Submit the JSON file to your course LMS\")\n",
    "print(\"3. Keep the TXT file for your records\")\n",
    "print(\"=\"*70)"
])

nb['cells'].append(export_markdown)
nb['cells'].append(export_cell)
print("✓ Added export section at end")

# ============================================================================
# Save the modified notebook
# ============================================================================

with open(r'c:\Users\jfwal\OneDrive\python-experiments-2023\ml-animated\ml_animated\lab_1_with_simple_answers.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print()
print("="*70)
print(f"✅ Modified notebook saved!")
print(f"   Total cells: {len(nb['cells'])} (was 36, now {len(nb['cells'])})")
print("="*70)
print()
print("Files created:")
print("  1. lab_1_attempt_3_ORIGINAL.ipynb (backup of original)")
print("  2. lab_1_with_simple_answers.ipynb (new version with answers)")
print()
print("You can now compare them side by side in your IDE!")
