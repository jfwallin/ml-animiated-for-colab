# DATA 1010 Lab Framework — Design Principles, Goals, and Flow

**Course:** DATA 1010 – Artificial Intelligence in Action
**Document Version:** 2.0
**Last Updated:** 2024

---

## Table of Contents

1. [Audience & Context](#1-audience--context)
2. [Overall Goals for the Lab Sequence](#2-overall-goals-for-the-lab-sequence)
3. [Lab Flow & Student Experience](#3-lab-flow--student-experience)
4. [The Cognitive Loop: Predict → Experiment → Explain](#4-the-cognitive-loop-predict--experiment--explain)
5. [Assessment & Learning Outcomes](#5-assessment--learning-outcomes)
6. [Instructor Preparation & Resources](#6-instructor-preparation--resources)
7. [Technical Infrastructure](#7-technical-infrastructure)

---

## 1. Audience & Context

DATA 1010 serves **first-year college students** with:

- **A wide range of prior experience**
  (from zero coding exposure to strong CS backgrounds)

- **No prerequisites** in computer science, math, or programming

- **High variation** in study habits, confidence, and motivation

- **Interest** in learning how modern AI works through hands-on exploration

### Lab Component Objectives

The lab component is intended to:

- **Lower the barrier of entry** for non-CS students
- **Challenge and enrich** students with stronger backgrounds
- **Build conceptual foundation** before mathematical or formal models
- **Provide repeated experience** with feedback-driven exploration, prediction, and explanation

---

## 2. Overall Goals for the Lab Sequence

### A. Build Core AI Intuition

Students should emerge with **intuitive mastery** of:

- **Error, loss, and optimization** — understanding how machines measure and minimize mistakes
- **Model behavior and decision boundaries** — recognizing how models partition input space
- **Embeddings, similarity, and representation** — grasping how data is transformed for learning
- **Multimodal inputs and outputs** — working with text, images, audio, and structured data
- **Evaluation, uncertainty, and explainability** — assessing model quality and limitations
- **Limits, failure modes, and bias** — recognizing when and why AI systems fail
- **Prediction and learning as iterative refinement** — seeing learning as continuous feedback-driven improvement

> **Key Principle:** The labs are designed to foster *thinking like an AI practitioner* before "writing code like one."

### B. Build Computational Confidence

By the end of the semester, students should:

- **Comfortably navigate** Google Colab
- **Read, modify, and run** Python cells
- **Interpret** plots and visual output
- **Understand** how small changes propagate in models

Labs are designed to be **approachable** and **redundancy-safe** — not dependent on heavy install steps or fragile toolchains.

### C. Emphasize Conceptual Learning, Not Coding Mechanics

The course does **NOT** aim to teach Python deeply or turn students into programmers.

Instead, coding is used as a lens for:

- **Exploration** — discovering patterns and behaviors
- **Visualization** — making abstract concepts concrete
- **Experimentation** — testing ideas and hypotheses
- **Reasoning** — developing mental models of how systems work

> **Design Philosophy:** Where AI instruction often hides computation, this course *demystifies* it through controlled, guided exposure.

### D. Encourage Prediction → Experiment → Explanation

Across the entire sequence, labs share a common **cognitive loop**:

#### 1. **Prediction**
Students anticipate what a model or parameter change will do.

#### 2. **Experiment**
Students run code, adjust parameters, test ideas, gather feedback.

#### 3. **Explanation**
Students articulate reasoning about what happened and why.

This pattern reinforces **scientific reasoning** and helps students **internalize AI concepts**.

### E. Support Group Collaboration

Labs are structured so that:

- Students work in **groups of 2–4**
- Roles are **flexible** (not rigid)
- Tasks encourage **discussion**, not division of labor
- Each student participates in **prediction and explanation**

Group work prioritizes:

- **Sensemaking** — making sense of observations together
- **Shared reasoning** — thinking through problems collaboratively
- **Collaborative discovery** — finding insights as a team
- **Peer teaching** — explaining concepts to one another

---

## 3. Lab Flow & Student Experience

Each lab follows a consistent **three-part structure**:

```
┌─────────────┐
│  Pre-Lab    │  (Asynchronous, 10–20 min)
│  (at home)  │  • Reading/video
└──────┬──────┘  • Vocabulary
       │         • Micro-quiz
       ↓         • Preview notebook
┌─────────────┐
│  In-Lab     │  (Synchronous, 75–90 min)
│  (in class) │  • Group work
└──────┬──────┘  • Interactive exploration
       │         • Guided activities
       ↓         • Predict-Experiment-Explain
┌─────────────┐
│  Post-Lab   │  (Asynchronous, 20–30 min)
│  (at home)  │  • Reflection questions
└─────────────┘  • Concept synthesis
                 • Preview next lab
```

### I. Pre-Lab (Asynchronous, 10–20 minutes)

#### Goals:
- **Reduce cognitive load** in-lab
- **Introduce key terms** before they encounter them in practice
- **Activate prior knowledge**
- **Ensure environment readiness**

#### Components:

1. **Short reading or video** (5–10 min)
   - Introduces the lab's core concept
   - Provides real-world context
   - Builds motivation

2. **Explanation of new vocabulary and concepts**
   - Key terms defined clearly
   - Visual aids where helpful
   - Examples provided

3. **Preview of what the lab will explore**
   - Learning objectives stated explicitly
   - Overview of activities
   - Expected outcomes

4. **Extremely lightweight code warm-up**
   - Run a cell
   - Change a number
   - See output change
   - Build confidence

5. **Short prediction/intuition prompts**
   - "What do you think will happen if...?"
   - "Which option will give a better result?"
   - Activate predictive thinking

6. **Auto-graded micro-quiz in LMS (3–4 questions)**
   - Check vocabulary understanding
   - Verify readiness
   - Low stakes (completion-based or easy points)

7. **Preview of the unified notebook**
   - Students see the full lab structure
   - Understand the scope
   - Can ask questions before class

> **Key Outcome:** Students walk in ready to engage, not to figure out tooling.

### II. In-Lab (Synchronous, 75–90 minutes)

#### Goals:
- **Hands-on exploration** of AI concepts
- **Collaborative problem-solving** in groups
- **Immediate feedback** on understanding
- **Active engagement** with materials

#### Structure:

**Opening (5–10 min):**
- Quick recap of pre-lab concepts
- Clarify any questions
- Form groups and assign shared notebook
- Review lab goals

**Main Activities (60–70 min):**

Divided into **4–6 sections**, each following the Predict-Experiment-Explain loop:

1. **Introduction to concept** (5 min)
   - Brief context
   - Key vocabulary
   - Connection to previous work

2. **Prediction phase** (2–3 min)
   - Group discusses: "What do you think will happen?"
   - Record predictions on handout
   - Commit to hypothesis

3. **Experiment phase** (5–10 min)
   - Run interactive code cells
   - Adjust parameters with sliders
   - Observe visualizations
   - Try different values

4. **Explanation phase** (3–5 min)
   - Group discusses: "Why did that happen?"
   - Answer questions on handout
   - Connect to broader concepts
   - Prepare to share with class

5. **Brief class discussion** (optional, 2–3 min)
   - One group shares insight
   - Instructor highlights key points
   - Address misconceptions

**Closing (5–10 min):**
- Recap key concepts
- Preview post-lab questions
- Note connections to next lab

#### Interactive Elements:

- **Sliders and widgets** for parameter exploration
- **Real-time visualizations** showing model behavior
- **"Warmer/Colder" feedback** guiding optimization
- **History tables** showing progression of guesses
- **Reveal mechanics** for hidden information (e.g., true parameters)

#### Group Roles (Suggested, Not Required):

- **Navigator:** Controls mouse/keyboard, runs cells
- **Recorder:** Documents observations, answers handout questions
- **Checker:** Verifies understanding, asks "why?" questions
- **Roles rotate** between sections

### III. Post-Lab (Asynchronous, 20–30 minutes)

#### Goals:
- **Consolidate learning** from in-lab activities
- **Deepen understanding** through reflection
- **Connect concepts** to broader AI/ML themes
- **Prepare** for next lab

#### Components:

1. **Reflection Questions (10–15 min)**
   - "In your own words, explain..."
   - "How does [concept A] relate to [concept B]?"
   - "What surprised you about...?"
   - Open-ended, synthesis-focused

2. **Concept Application (5–10 min)**
   - "How might this be used in...?"
   - "What would happen if...?"
   - Transfer to new scenarios

3. **Vocabulary Reinforcement (2–3 min)**
   - Define key terms in own words
   - Provide examples from lab

4. **Preview Next Lab (2–3 min)**
   - Brief reading on next concept
   - Connection to current lab
   - Build anticipation

5. **Optional Extensions (if interested)**
   - "Try adjusting these parameters..."
   - "Explore this additional notebook..."
   - "Read about this real-world application..."

#### Assessment:
- **Completion-based** (e.g., 80% for thoughtful effort)
- **Graded on understanding**, not correctness
- Feedback focuses on **conceptual gaps**

---

## 4. The Cognitive Loop: Predict → Experiment → Explain

This three-phase loop is the **pedagogical core** of the lab sequence.

### Phase 1: Prediction

**What students do:**
- Make explicit hypotheses
- Discuss with group
- Record predictions

**Example prompts:**
- "Before running this cell, predict: will the error increase or decrease?"
- "Which parameter (slope or intercept) will have a bigger effect on the fit?"
- "If we use a larger learning rate, what will happen to the optimization?"

**Why it matters:**
- **Activates prior knowledge**
- **Creates cognitive dissonance** when predictions are wrong (learning opportunity)
- **Engages** students in the process
- **Develops intuition** before seeing results

### Phase 2: Experiment

**What students do:**
- Run code cells
- Adjust parameters
- Observe visualizations
- Gather data
- Test variations

**Design principles:**
- **Immediate feedback** (results appear quickly)
- **Safe to fail** (no wrong answers, just observations)
- **Low-friction** (sliders, not code editing)
- **Visual** (plots, animations, colors)

**Why it matters:**
- **Hands-on learning** is more memorable
- **Discovery** feels rewarding
- **Variability** shows patterns
- **Iteration** builds fluency

### Phase 3: Explanation

**What students do:**
- Articulate what happened
- Explain why it happened
- Connect to concepts
- Identify patterns

**Example prompts:**
- "Why did the loss decrease when you changed X?"
- "How does this relate to what we learned about optimization?"
- "What would you tell a friend who asked how this works?"

**Why it matters:**
- **Consolidates understanding**
- **Reveals misconceptions**
- **Develops communication skills**
- **Deepens learning** through articulation

### Facilitating the Loop

**Instructor role:**
- **Model the loop** in demonstrations
- **Use consistent language** ("Let's predict...", "Now experiment...", "Can you explain...?")
- **Validate effort** in all phases
- **Probe deeper** with follow-up questions
- **Connect** to bigger picture

**Common pitfalls:**
- Students skip prediction and jump to experiment
- Students observe but don't explain
- Groups split work instead of collaborating
- Focus on "getting it right" instead of understanding

**Solutions:**
- **Require** prediction before revealing experiment cell
- **Pause** after experiments for discussion
- **Structure** questions to require group consensus
- **Emphasize** process over correctness

---

## 5. Assessment & Learning Outcomes

### Assessment Philosophy

- **Low-stakes, high-feedback**
- **Process over product**
- **Understanding over correctness**
- **Growth-oriented**

### Assessment Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| **Pre-Lab Quiz** | 5–10% | Readiness check, vocabulary |
| **In-Lab Participation** | 20–30% | Engagement, collaboration |
| **Post-Lab Reflection** | 30–40% | Conceptual understanding |
| **Lab Practical** (midterm/final) | 30–40% | Synthesis, application |

### Learning Outcomes by Lab

Each lab should have **3–5 specific learning outcomes**:

**Example (Lab 1 on Error and Optimization):**

By the end of this lab, students should be able to:

1. **Explain** how error is measured at a single data point vs. globally
2. **Describe** why we square errors instead of adding raw residuals
3. **Predict** how changing model parameters affects total loss
4. **Demonstrate** how optimization uses only loss values (not the data directly)
5. **Connect** line-fitting optimization to broader ML training

### Measuring Success

**Formative indicators:**
- Students make thoughtful predictions
- Groups discuss reasoning actively
- Explanations show conceptual understanding
- Students connect to prior labs

**Summative indicators:**
- Post-lab reflections show depth
- Lab practicals demonstrate transfer
- Students can explain concepts to peers
- Course evaluations highlight understanding

---

## 6. Instructor Preparation & Resources

### Before Each Lab

**1 week before:**
- [ ] Review lab notebook end-to-end
- [ ] Test all interactive widgets
- [ ] Prepare any handouts
- [ ] Post pre-lab materials to LMS

**1 day before:**
- [ ] Re-run notebook in Colab to ensure functionality
- [ ] Prepare any additional visualizations for class discussion
- [ ] Review common student questions from previous offerings
- [ ] Set up breakout rooms (if remote)

**Day of:**
- [ ] Open lab notebook and test key cells
- [ ] Have backup plan for technical issues
- [ ] Prepare group assignments (or randomization method)

### During Lab

**Opening:**
- Briefly review pre-lab concept (don't re-teach)
- Address any setup issues quickly
- Set expectations for group work

**While groups work:**
- **Circulate actively** — visit each group multiple times
- **Listen before intervening** — let groups struggle productively
- **Ask probing questions** instead of giving answers
- **Note common issues** for whole-class discussion
- **Watch for groups that are stuck** vs. struggling productively

**Common situations:**

| Situation | Approach |
|-----------|----------|
| Group is off-task | Redirect with question: "What are you predicting for this next part?" |
| One student dominating | "Let's hear from [quieter student] — what do you think?" |
| Technical error | "What does the error message say? Can anyone in the group interpret it?" |
| Conceptual confusion | Don't explain directly — ask group to discuss, then guide |
| Moving too fast | "Before you move on, can you explain to each other what just happened?" |
| Stuck | Provide small hint or ask scaffolding question, not full answer |

**Closing:**
- Bring class together 10 min before end
- Highlight 2–3 key insights from the lab
- Preview post-lab questions
- Note connections to next lab

### After Lab

**Same day:**
- Note any technical issues for future fixes
- Record common misconceptions observed
- Save any good student explanations for future examples

**Before next lab:**
- Review post-lab submissions for understanding
- Identify concepts that need reinforcement
- Adjust next lab if needed

### Resources for Instructors

**Documentation:**
- Full lab notebook with instructor notes
- Answer key for handout questions
- Common misconceptions guide
- Technical troubleshooting FAQ

**Professional development:**
- Pedagogical notes on Predict-Experiment-Explain
- Group facilitation strategies
- Colab tips and tricks

---

## 7. Technical Infrastructure

### Platform: Google Colab

**Why Colab:**
- **No installation required** — runs in browser
- **Free GPU access** — for later labs with larger models
- **Familiar interface** — Jupyter-based
- **Easy sharing** — link-based distribution
- **Persistent** — auto-saves to Google Drive

**Setup Requirements:**
- Students need Google account (provided by university if needed)
- Notebook shared via link (view-only or copy)
- Students "Save a copy in Drive" to work

### Notebook Design Principles

**Structure:**
- Clear section markers
- Estimated time per section
- Visual separation (horizontal rules)
- Consistent heading hierarchy

**Code Cells:**
- **Minimize editing required** — students run, don't write
- **Clear comments** explaining what code does
- **Error handling** for common input mistakes
- **Progress indicators** for long-running cells

**Interactive Elements:**
- **Sliders** for continuous parameters
- **Dropdowns/checkboxes** for categorical choices
- **Buttons** for actions (submit, reveal, reset)
- **Output areas** that update in place (not scroll)

**Visualizations:**
- **Matplotlib** for static plots
- **Interactive plots** where helpful (but not overwhelming)
- **Consistent color schemes** across labs
- **Clear labels** on axes, legends, titles
- **Annotations** highlighting key features

### Accessibility Considerations

- **Color-blind friendly** palettes
- **Alt text** for images (in markdown cells)
- **Text descriptions** of visualizations
- **Keyboard navigation** supported
- **Screen reader** compatible where possible

### Common Technical Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Runtime disconnected" | Re-run setup cells; save work frequently |
| Widgets not displaying | Refresh page; check browser compatibility |
| Slow performance | Reduce data size; use smaller examples |
| Import errors | Ensure all !pip install commands ran successfully |
| Plots not showing | Run %matplotlib inline; check output cleared |

### Version Control

- Lab notebooks stored in GitHub repository
- Versioning: Lab1_v2.3.ipynb
- Changelog maintained
- Student-facing version generated from master

### Data & Model Hosting

- Small datasets: embedded in notebook
- Medium datasets: Google Drive links
- Large datasets: Cloud storage with public access
- Pre-trained models: Hosted and versioned

---

## Appendix A: Example Question Stems for Each Phase

### Prediction Phase
- "Before running this cell, what do you think will happen when...?"
- "Which option do you predict will give better results: A or B? Why?"
- "If we increase [parameter], will [outcome] increase or decrease?"
- "Draw what you think the graph will look like."
- "What would a good [model/parameter/solution] look like?"

### Experiment Phase
- "Try several different values and record what you observe."
- "Adjust [parameter] and watch how [outcome] changes."
- "Can you find a combination that makes [metric] as small/large as possible?"
- "Run this 5 times and see if results change."
- "Compare [condition A] vs [condition B]."

### Explanation Phase
- "Why do you think that happened?"
- "How does [observation] relate to [concept]?"
- "Explain in your own words what [code/model/algorithm] is doing."
- "What pattern did you notice? Why does that pattern make sense?"
- "If you had to teach this to a friend, how would you explain it?"
- "Connect this to something we learned in a previous lab."

---

## Appendix B: Unified Notebook Template

A "unified notebook" refers to a single Jupyter/Colab notebook that contains all three phases of the lab:

```
┌────────────────────────────────────────┐
│  DATA 1010 – Lab N: [Topic]            │
├────────────────────────────────────────┤
│  📘 Pre-Lab (Complete Before Class)    │
│     - Reading                          │
│     - Vocabulary                       │
│     - Warm-up code                     │
│     - Prediction prompts               │
├────────────────────────────────────────┤
│  🔬 In-Lab (Group Work in Class)       │
│     - Section 1                        │
│     - Section 2                        │
│     - Section 3                        │
│     - Section 4                        │
├────────────────────────────────────────┤
│  📝 Post-Lab (Complete After Class)    │
│     - Reflection questions             │
│     - Synthesis                        │
│     - Preview next lab                 │
└────────────────────────────────────────┘
```

This structure:
- Keeps all materials in one place
- Shows students the full arc
- Allows asynchronous and synchronous work
- Maintains consistency across labs

---

## Document History

- **Version 2.0** (2024): Expanded with complete sections, assessment details, instructor notes
- **Version 1.0** (2023): Initial framework document

---

**For questions or suggestions about this framework, contact:** [Instructor/Course Coordinator]
