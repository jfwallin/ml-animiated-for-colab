"""
Script to add ENHANCED answer collection system with attempt tracking and timestamps
"""

import json
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
with open(r'c:\Users\jfwal\OneDrive\python-experiments-2023\ml-animated\ml_animated\lab_1_attempt_3.ipynb', 'r', encoding='utf-8') as f:
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
# STEP 2: Insert ENHANCED answer system setup after cell 11 (group code)
# ============================================================================

setup_cell = create_code_cell([
    "# ============================================================================\n",
    "# ANSWER COLLECTION SETUP (with attempt tracking and timestamps)\n",
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
    "# Track answer timestamps (when each answer was last updated)\n",
    "answer_timestamps = {}\n",
    "\n",
    "# Track attempts in each interactive section\n",
    "section_attempts = {\n",
    "    \"section_3_line_fitting\": 0,\n",
    "    \"section_3_2_parameter_space\": 0,\n",
    "    \"section_4_hidden_function\": 0,\n",
    "    \"section_5_mountain_landscape\": 0\n",
    "}\n",
    "\n",
    "# Track when sections were started/completed\n",
    "section_timestamps = {\n",
    "    \"section_3_line_fitting\": {\"started\": None, \"completed\": None},\n",
    "    \"section_3_2_parameter_space\": {\"started\": None, \"completed\": None},\n",
    "    \"section_4_hidden_function\": {\"started\": None, \"completed\": None},\n",
    "    \"section_5_mountain_landscape\": {\"started\": None, \"completed\": None}\n",
    "}\n",
    "\n",
    "def record_answer_update(question_id):\n",
    "    \"\"\"Record timestamp when an answer is updated.\"\"\"\n",
    "    answer_timestamps[question_id] = datetime.now().isoformat()\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"✓ Answer system ready!\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"Group Code: {group_code}\")\n",
    "print(f\"Started at: {datetime.now().strftime('%I:%M:%S %p')}\")\n",
    "print()\n",
    "print(\"📝 Answer questions by editing the answer_X variables\")\n",
    "print(\"📊 Your attempts in each section will be tracked\")\n",
    "print(\"📤 Export your answers at the end of the lab\")\n",
    "print(\"=\" * 60)"
])

nb['cells'].insert(12, setup_cell)
print("✓ Inserted enhanced answer system setup after cell 11")

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
    "# Record timestamp\n",
    "record_answer_update(\"Q1\")\n",
    "print(f\"✓ Answer 1 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

answer2_cell = create_code_cell([
    "# Question 2: If we changed the slope or intercept of the line,\n",
    "# how would that change the loss?\n",
    "\n",
    "answer_2 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "# Record timestamp\n",
    "record_answer_update(\"Q2\")\n",
    "print(f\"✓ Answer 2 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

nb['cells'].insert(19, section2_header)
nb['cells'].insert(20, answer1_cell)
nb['cells'].insert(21, answer2_cell)
print("✓ Inserted Section 2 answer cells")

# ============================================================================
# STEP 3.5: Add attempt tracking for Section 3 (line fitting)
# ============================================================================

# Find the interactive line fitting cell (cell 19, now 22 after inserts)
# We need to modify it to track attempts

section3_tracking_setup = create_code_cell([
    "# Track section start\n",
    "if section_timestamps[\"section_3_line_fitting\"][\"started\"] is None:\n",
    "    section_timestamps[\"section_3_line_fitting\"][\"started\"] = datetime.now().isoformat()\n",
    "    print(\"📊 Section 3 (Line Fitting) - attempt tracking started\")"
])

nb['cells'].insert(22, section3_tracking_setup)
print("✓ Inserted Section 3 tracking setup")

# Modify the existing interact cell to track attempts
# This will need to be done by wrapping the plot_guess function

# ============================================================================
# STEP 4: Add Section 3.1 answer cells (after cell 20, now 25)
# ============================================================================

section31_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 3.1: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "# Mark section 3 as completed\n",
    "if section_timestamps[\"section_3_line_fitting\"][\"completed\"] is None:\n",
    "    section_timestamps[\"section_3_line_fitting\"][\"completed\"] = datetime.now().isoformat()\n",
    "    print(f\"📊 Section 3 completed: {section_attempts['section_3_line_fitting']} attempts made\")\n",
    "    print()\n",
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
    "record_answer_update(\"Q3\")\n",
    "print(f\"✓ Answer 3 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
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
    "record_answer_update(\"Q4\")\n",
    "print(f\"✓ Answer 4 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
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
    "record_answer_update(\"Q5\")\n",
    "print(f\"✓ Answer 5 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

# Insert after the section 3 line fitting code
# Need to find the right position - after attempt_history cell (now around cell 25)
nb['cells'].insert(26, section31_header)
nb['cells'].insert(27, answer3_cell)
nb['cells'].insert(28, answer4_cell)
nb['cells'].insert(29, answer5_cell)
print("✓ Inserted Section 3.1 answer cells")

# ============================================================================
# STEP 5: Add Section 3.3 answer cells with tracking
# ============================================================================

section33_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 3.3: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "# Mark section 3.2 as completed (parameter space game)\n",
    "if section_timestamps[\"section_3_2_parameter_space\"][\"completed\"] is None:\n",
    "    section_timestamps[\"section_3_2_parameter_space\"][\"completed\"] = datetime.now().isoformat()\n",
    "    print(f\"📊 Section 3.2 completed: {section_attempts['section_3_2_parameter_space']} guesses made\")\n",
    "    print()\n",
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
    "record_answer_update(\"Q6\")\n",
    "print(f\"✓ Answer 6 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
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
    "record_answer_update(\"Q7\")\n",
    "print(f\"✓ Answer 7 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
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
    "record_answer_update(\"Q8\")\n",
    "print(f\"✓ Answer 8 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

# Insert after section 3.2 parameter space game (around cell 34 now)
nb['cells'].insert(35, section33_header)
nb['cells'].insert(36, answer6_cell)
nb['cells'].insert(37, answer7_cell)
nb['cells'].insert(38, answer8_cell)
print("✓ Inserted Section 3.3 answer cells")

# ============================================================================
# STEP 6: Add Section 4.1 answer cells with tracking
# ============================================================================

section41_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 4.1: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "# Mark section 4 as completed (hidden function)\n",
    "if section_timestamps[\"section_4_hidden_function\"][\"completed\"] is None:\n",
    "    section_timestamps[\"section_4_hidden_function\"][\"completed\"] = datetime.now().isoformat()\n",
    "    print(f\"📊 Section 4 completed: {section_attempts['section_4_hidden_function']} attempts made\")\n",
    "    print()\n",
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
    "record_answer_update(\"Q9\")\n",
    "print(f\"✓ Answer 9 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

answer10_cell = create_code_cell([
    "# Question 10: How did the \"warmer/colder\" feedback influence your choices?\n",
    "\n",
    "answer_10 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "record_answer_update(\"Q10\")\n",
    "print(f\"✓ Answer 10 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
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
    "record_answer_update(\"Q11\")\n",
    "print(f\"✓ Answer 11 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

# Insert after section 4 (around cell 45 now)
nb['cells'].insert(46, section41_header)
nb['cells'].insert(47, answer9_cell)
nb['cells'].insert(48, answer10_cell)
nb['cells'].insert(49, answer11_cell)
print("✓ Inserted Section 4.1 answer cells")

# ============================================================================
# STEP 7: Add Section 5.1 answer cells with tracking
# ============================================================================

section51_header = create_code_cell([
    "# ============================================================================\n",
    "# SECTION 5.1: ANSWER QUESTIONS\n",
    "# ============================================================================\n",
    "\n",
    "# Mark section 5 as completed (mountain landscape)\n",
    "if section_timestamps[\"section_5_mountain_landscape\"][\"completed\"] is None:\n",
    "    section_timestamps[\"section_5_mountain_landscape\"][\"completed\"] = datetime.now().isoformat()\n",
    "    print(f\"📊 Section 5 completed: {section_attempts['section_5_mountain_landscape']} samples taken\")\n",
    "    print()\n",
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
    "record_answer_update(\"Q12\")\n",
    "print(f\"✓ Answer 12 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
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
    "record_answer_update(\"Q13\")\n",
    "print(f\"✓ Answer 13 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

answer14_cell = create_code_cell([
    "# Question 14: Compare your best sample to the true global peak shown on the plot.\n",
    "# Were you close to the global maximum, or did you end up stuck near a local maximum?\n",
    "\n",
    "answer_14 = \"\"\"\n",
    "Type your answer here.\n",
    "\"\"\"\n",
    "\n",
    "record_answer_update(\"Q14\")\n",
    "print(f\"✓ Answer 14 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
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
    "record_answer_update(\"Q15\")\n",
    "print(f\"✓ Answer 15 saved at {datetime.now().strftime('%I:%M:%S %p')}\")"
])

# Insert after section 5 (around cell 54 now)
nb['cells'].insert(55, section51_header)
nb['cells'].insert(56, answer12_cell)
nb['cells'].insert(57, answer13_cell)
nb['cells'].insert(58, answer14_cell)
nb['cells'].insert(59, answer15_cell)
print("✓ Inserted Section 5.1 answer cells")

# ============================================================================
# STEP 8: Add ENHANCED export section at the end
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
    "# EXPORT YOUR ANSWERS (with attempt tracking and timestamps)\n",
    "# ============================================================================\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(\"📤 EXPORTING YOUR ANSWERS\")\n",
    "print(\"=\"*70)\n",
    "print()\n",
    "\n",
    "# Calculate engagement metrics\n",
    "total_attempts = sum(section_attempts.values())\n",
    "\n",
    "# Collect section durations\n",
    "section_durations = {}\n",
    "for section, times in section_timestamps.items():\n",
    "    if times[\"started\"] and times[\"completed\"]:\n",
    "        start = datetime.fromisoformat(times[\"started\"])\n",
    "        end = datetime.fromisoformat(times[\"completed\"])\n",
    "        duration_seconds = (end - start).total_seconds()\n",
    "        section_durations[section] = duration_seconds\n",
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
    "    \"answer_timestamps\": answer_timestamps,\n",
    "    \"engagement_metrics\": {\n",
    "        \"section_attempts\": section_attempts,\n",
    "        \"total_attempts\": total_attempts,\n",
    "        \"section_timestamps\": section_timestamps,\n",
    "        \"section_durations_seconds\": section_durations\n",
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
    "print(f\"Total attempts across all sections: {total_attempts}\")\n",
    "print()\n",
    "print(\"Attempts per section:\")\n",
    "for section, count in section_attempts.items():\n",
    "    duration = section_durations.get(section, 0)\n",
    "    print(f\"  {section}: {count} attempts ({duration/60:.1f} minutes)\")\n",
    "print()\n",
    "\n",
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
    "ENGAGEMENT METRICS:\n",
    "{'='*80}\n",
    "  Total Attempts: {total_attempts}\n",
    "\"\"\"\n",
    "\n",
    "for section, count in section_attempts.items():\n",
    "    duration = section_durations.get(section, 0)\n",
    "    text_output += f\"\\n  {section}: {count} attempts ({duration/60:.1f} min)\"\n",
    "\n",
    "text_output += f\"\"\"\n",
    "\n",
    "{'='*80}\n",
    "ANSWERS:\n",
    "{'='*80}\n",
    "\n",
    "\"\"\"\n",
    "\n",
    "for i in range(1, total + 1):\n",
    "    q_id = f\"Q{i}\"\n",
    "    timestamp = answer_timestamps.get(q_id, \"Not answered\")\n",
    "    text_output += f\"\\n{q_id}: {all_answers['questions'][q_id]}\\n\"\n",
    "    text_output += f\"Timestamp: {timestamp}\\n\"\n",
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
print("✓ Added enhanced export section at end")

# ============================================================================
# Save the modified notebook
# ============================================================================

output_file = r'c:\Users\jfwal\OneDrive\python-experiments-2023\ml-animated\ml_animated\lab_1_with_tracking.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print()
print("="*70)
print(f"✅ Enhanced notebook saved!")
print(f"   Total cells: {len(nb['cells'])} (was 36)")
print("="*70)
print()
print("Files created:")
print("  1. lab_1_with_tracking.ipynb (NEW - with attempt tracking)")
print()
print("Features added:")
print("  ✓ Answer timestamps - when each answer was last updated")
print("  ✓ Section attempt counters - track engagement")
print("  ✓ Section start/completion timestamps")
print("  ✓ Section duration calculations")
print("  ✓ Engagement metrics in export")
