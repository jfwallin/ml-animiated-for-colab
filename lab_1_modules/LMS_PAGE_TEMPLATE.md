# Lab 1 - LMS Page Template

## Complete Example for Canvas/Blackboard/Moodle

Copy this structure into your LMS page editor. Replace `[LINK]` placeholders with actual Colab links.

---

# Lab 1: Models, Errors, Loss, Optimization, and Learning

## Overview

**Due Date:** [Set in LMS]
**Points:** 100
**Estimated Time:** 60-75 minutes

### Learning Objectives

By the end of this lab, you will be able to:
- Define and calculate global error (loss) for a model
- Understand the relationship between local and global error
- Use optimization strategies to minimize error
- Connect these concepts to machine learning training
- Recognize challenges in optimization (local vs. global optima)

### What You'll Need

- **Computer with internet** (laptop or desktop recommended)
- **Google account** (to use Google Colab)
- **Group code** (you'll choose one in Module 0)
- **About 60-75 minutes** of focused work time

### AI Use Policy

You **may** use AI tools (ChatGPT, Claude, etc.) to:
- Help understand Python error messages
- Explain what code is doing
- Debug when something breaks

You **may not** use AI tools to:
- Generate answers to conceptual questions
- Choose parameter values for you
- Complete the exercises without group discussion

If you use AI, note what you used and how in your submission.

---

## 🚀 Getting Started

### Before You Begin

1. Make sure you can access Google Colab: https://colab.research.google.com
2. Have your group ready (or work independently if solo)
3. Set aside uninterrupted time to complete all modules

---

## Module 0: Setup (Required - Do This First!)

### Purpose

This module sets up your unique group code and generates the data you'll use throughout the lab.

### Activity

🔗 **[Open Module 0: Setup in Google Colab](YOUR_COLAB_LINK_MODULE_0)**

**Instructions:**
1. Click the link above (opens in new tab/window)
2. Run all cells in order (click play button or Shift+Enter)
3. Enter a group code when prompted (any integer, like 1234)
4. **Write down your group code** - you'll need it for all other modules!
5. Return here when complete

### ✅ Checkpoint

**Enter your group code here:** [_______]
(This helps us verify you completed Module 0)

---

## Part 1: Understanding Global Error

### Learning Objectives

- Understand what "global error" or "loss" means
- See how error is calculated across all data points
- Visualize the relationship between predictions and data

### Background Reading

When we fit a model to data, we need a way to measure how well it's doing. The **global error** (also called **loss**) is a single number that tells us how far our model's predictions are from the actual data.

For a line fit to data points:
- Each point has a **local error**: how far that specific point is from the line
- The **global error** combines all local errors into one measure
- We typically use **sum of squared errors (SSE)** to avoid cancellation

### Activity

🔗 **[Open Module 1: Global Error in Google Colab](YOUR_COLAB_LINK_MODULE_1)**

**Instructions:**
1. Click the link above
2. **Enter the same group code** you used in Module 0
3. Run all cells and observe the visualization
4. Return here to answer questions

### Questions

**Q1:** In your own words, what does "global error" or "loss" measure in this plot? (3-5 sentences)

[Large text box - 500 character minimum]

---

**Q2:** If we changed the slope or intercept of the line, how would that change the loss? Would it increase, decrease, or could it go either way? Explain your reasoning.

[Large text box - 300 character minimum]

---

---

## Part 2: Interactive Line Fitting

### Learning Objectives

- Explore how changing parameters affects error
- Understand the relationship between local and global error
- Practice minimizing error through experimentation
- See how individual data points contribute to total error

### Background Reading

In machine learning, we often adjust model parameters to minimize error. In this exercise, you'll manually adjust the slope and intercept of a line to see how they affect the global error.

The **residual lines** (dashed vertical lines) show the local error at each point - the distance from the point to your line. Watch how these change as you adjust the parameters!

### Activity

🔗 **[Open Module 2: Line Fitting in Google Colab](YOUR_COLAB_LINK_MODULE_2)**

**Instructions:**
1. Click the link above
2. Enter your group code (same as before!)
3. Use the sliders to adjust slope (m) and intercept (b)
4. Try to minimize the Global Error (SSE)
5. Pay attention to the warm/cold feedback
6. Make at least 10-15 attempts
7. Return here to answer questions

### Questions

**Q3:** How do the residual lines (dashed vertical lines) help you understand the local error at each point? What do longer vs. shorter residual lines tell you?

[Text box]

---

**Q4:** Can you make the global error small even if a few points still have relatively large errors? Describe a situation where this happens and explain why.

[Text box]

---

**Q5:** Suppose you added one very extreme outlier point far away from all the others. Predict how this would affect:
- The best-fit line position
- The global error value
- Your strategy for minimizing error

[Text box]

---

---

## Part 3: Parameter Space Optimization

### Learning Objectives

- Learn to optimize using only global error feedback
- Explore parameter space systematically
- Understand how machine learning algorithms "see" the problem
- Compare manual search to gradient-based methods

### Background Reading

In the previous module, you could see the data points and the line. **Machine learning algorithms don't see this!** They only see the error value and must adjust parameters to reduce it.

In this module, you'll experience what it's like to optimize without visualization - only seeing whether your guess is getting warmer (lower error) or colder (higher error).

The "parameter space" is the 2D plane where:
- Horizontal axis = slope (m)
- Vertical axis = intercept (b)
- Each point (m, b) represents one possible line
- Color shows the error for that line

### Activity

🔗 **[Open Module 3: Parameter Space in Google Colab](YOUR_COLAB_LINK_MODULE_3)**

**Instructions:**
1. Click the link above
2. Enter your group code
3. Use sliders to choose (m, b) values
4. Click "Submit guess" to see the error
5. Try to find low-error values using only the feedback
6. Make at least 8-10 guesses
7. Click "Done" when ready to see the full landscape
8. Return here to answer questions

### Questions

**Q6:** Describe how your guesses for (m, b) moved over time. Did you follow any systematic strategy (like "move m first, then b") or was it more exploratory? What strategy worked best?

[Text box]

---

**Q7:** Look at the revealed MSE landscape plot. How close was your best guess to:
- (a) The approximate global minimum on the grid?
- (b) The least-squares solution from the data?

What does this tell you about optimizing using only the global error value?

[Text box]

---

**Q8:** In our earlier line-fitting exercise, you could see the data, the line, and the residuals. In this game, you only saw the MSE (error value). How is this situation similar to how many machine learning models are trained? Why might ML training be even harder?

[Text box]

---

---

## Part 4: Hidden Function Optimization

### Learning Objectives

- Optimize a 1D function you cannot see
- Develop and apply search strategies
- Use warm/cold feedback effectively
- Understand gradient-free optimization

### Background Reading

Many real-world optimization problems involve functions we can't visualize or understand fully. We can only evaluate the function at specific points and must find the optimum using this limited information.

In this module, there's a hidden mathematical function. You can choose any x value in the range [-10, 10] and see what f(x) is, but you won't see the function's shape until the end.

Your goal: Find the x that gives the smallest f(x) value!

### Activity

🔗 **[Open Module 4: Hidden Function in Google Colab](YOUR_COLAB_LINK_MODULE_4)**

**Instructions:**
1. Click the link above
2. Enter your group code
3. Choose x values using the slider
4. Click button to evaluate f(x)
5. Use warm/cold feedback to guide your search
6. Make at least 10-15 attempts
7. Try to find the minimum value
8. Return here to answer questions

### Questions

**Q9:** What strategies did your group use to choose new values of x within the allowed range [-10, 10]? Did you use a systematic approach (like binary search, grid search) or something else?

[Text box]

---

**Q10:** How did the "warmer/colder" feedback influence your choices? Was it helpful, misleading, or both? Explain with specific examples from your search.

[Text box]

---

**Q11:** Imagine that even the scatter plot of your guesses was hidden, and you only saw a table of (x, f(x)) pairs. Would you still be able to find a good minimum? How? What additional challenges would this create?

[Text box]

---

---

## Part 5: Mountain Landscape Search (Final Challenge!)

### Learning Objectives

- Search in 2D space for an optimal point
- Distinguish between local and global maxima
- Understand optimization challenges in ML
- Apply search strategies in higher dimensions

### Background Reading

In the final challenge, you'll search for the highest peak in a mountain landscape. But there's a twist: there are **multiple peaks** of different heights!

This is similar to loss landscapes in machine learning:
- Multiple local minima (or maxima)
- One global minimum (or maximum)
- Risk of getting "stuck" in a local optimum
- Need strategies to explore broadly

Your task: Sample (x, y) locations to measure altitude and try to find the **highest peak** in the landscape.

### Activity

🔗 **[Open Module 5: Mountain Landscape in Google Colab](YOUR_COLAB_LINK_MODULE_5)**

**Instructions:**
1. Click the link above
2. Enter your group code
3. Use sliders to choose (x, y) coordinates
4. Click button to sample the altitude
5. Try to find the highest point
6. Make at least 15-20 samples
7. Click "Done" to reveal the full landscape
8. Return here for final questions

### Questions

**Q12:** Describe your group's strategy for choosing new (x, y) locations. How did you decide where to sample next after finding a high point? Did your strategy change as you explored?

[Text box]

---

**Q13:** Look at the revealed landscape. How many local peaks can you see? Did your group spend most of its time near one peak, or did you explore multiple regions? Why?

[Text box]

---

**Q14:** Compare your best sample to the true global peak shown on the plot. Were you close to the global maximum, or did you end up stuck near a local maximum? What made it hard to find the true peak?

[Text box]

---

**Q15:** Explain how this mountain-peak search is similar to what happens in machine learning when an algorithm is trying to optimize a loss function that has many "bumps" (local minima or maxima).

What risks does a model face if it only explores one region of the loss landscape? How might ML practitioners address this challenge?

[Text box]

---

---

## 🎉 Lab Complete!

### Before You Submit

Make sure you've:
- ✅ Completed all 6 modules (Module 0-5)
- ✅ Answered all 15 questions
- ✅ Used the same group code throughout
- ✅ Reviewed your answers for completeness

### Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Completion** | 25 pts | All modules completed, all questions answered |
| **Understanding** | 40 pts | Answers demonstrate conceptual understanding |
| **Depth** | 20 pts | Detailed explanations with examples |
| **Connection** | 15 pts | Clear connections to ML concepts |
| **Total** | **100 pts** | |

### Specific Question Rubrics

**Questions 1-2 (Global Error):**
- Correctly explains what global error measures
- Understands how parameters affect error
- Uses appropriate terminology

**Questions 3-5 (Local vs. Global):**
- Distinguishes local and global error
- Explains residuals and their meaning
- Predicts outlier effects with reasoning

**Questions 6-8 (Parameter Space):**
- Describes search strategy clearly
- Analyzes results from landscape visualization
- Connects to ML training processes

**Questions 9-11 (1D Optimization):**
- Explains search strategies used
- Evaluates effectiveness of feedback
- Reasons about information limitations

**Questions 12-15 (2D Optimization):**
- Describes 2D search strategy
- Analyzes local vs. global optima
- Makes clear ML connections
- Discusses practical implications

---

## Getting Help

If you encounter technical issues:
1. Try refreshing the Colab page
2. Make sure you're entering the correct group code
3. Check that you're running cells in order
4. Contact your TA or instructor if problems persist

For conceptual questions:
- Review the background reading sections
- Discuss with your group
- Attend office hours
- Post on the course forum

---

## Submission

Click the **Submit** button below when you're ready.

[LMS Submit Button]

---

*Remember: The goal is to understand how optimization works, not to achieve perfect scores. Focus on the learning process and connecting concepts to machine learning!*
