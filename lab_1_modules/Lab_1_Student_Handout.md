# Lab 1: Models, Errors, Loss, Optimization, and Learning
## Student Handout

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In this lab, you'll explore the fundamental concepts behind how machines learn from data:

1. **How machines measure error** – Understanding what "loss" means
2. **What makes a good model** – Adjusting parameters to reduce error
3. **How optimization works** – Finding the best parameters through search
4. **Why learning is hard** – Multiple solutions and getting stuck

### Lab Structure

This lab consists of **6 modules** that you'll complete in order using **the same group code**:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | Setup and Group Code | ~3 min | Prelab |
| 1 | Understanding Global Error | ~5 min | Prelab |
| 2 | Interactive Line Fitting | ~15 min | In-class |
| 3 | Parameter Space Optimization | ~20 min | In-class |
| 4 | Hidden Function Optimization | ~15 min | In-class |
| 5 | Mountain Landscape Search | ~20 min | In-class |

**Total Time:** ~75 minutes

### Working in Groups

- Work in **small groups** (2-4 people)
- One person shares their screen running the notebooks
- Everyone participates in discussion
- All group members use the **same group code**

### Key Concept

> **A MODEL is a system that:**
> - Makes predictions about data
> - Has adjustable **parameters**
> - Can be improved by reducing its total **error** (or **loss**)

### AI Use Policy

You may use AI tools (ChatGPT, Claude, etc.) to:
- ✅ Explain concepts in different ways
- ✅ Help understand error messages
- ✅ Suggest debugging approaches

You should NOT use AI tools to:
- ❌ Generate complete answers to lab questions
- ❌ Write code for you without understanding it
- ❌ Do the thinking for you

**Remember:** The goal is to develop your own understanding, not to get perfect answers quickly.

---

## Module 0: Setup and Group Code

**Time:** ~3 minutes
**Type:** Prelab

### What You'll Do

Set up your Google Colab environment and generate unique parameters for your group's lab experience.

### Important Steps

1. **Open the notebook** in Google Colab
2. **Run the test cell** to verify Colab is working
3. **Enter your group code** – This is a unique integer (e.g., 42, 1234) that your group chooses
4. **Keep this number!** You'll need to enter the **same group code** in every module

### What Gets Created

Your group code generates unique parameters that will be used throughout the lab:
- A line with specific slope and intercept (for Modules 2-3)
- A hidden function shape (for Module 4)
- A mountain landscape with several peaks (for Module 5)

### Why This Matters

Using the same group code ensures:
- All group members see the same data and problems
- You can compare results with your group
- The instructor can verify your work

---

## Module 1: Understanding Global Error

**Time:** ~5 minutes
**Type:** Prelab

### Learning Objectives

- Understand what "global error" or "loss" means
- See how error is measured across all data points
- Distinguish between local (point-wise) and global error

### What You'll Do

View a scatter plot showing data points and a line. The notebook will calculate and display the total error between the data and the line.

### Key Concepts

**Error at a Single Point:**
```
error = actual value - predicted value
```
- Positive error = model predicted too low
- Negative error = model predicted too high

**Why Square Errors?**

Raw errors can cancel out:
```
Point 1: error = +5 (predicted too low)
Point 2: error = -5 (predicted too high)
Total raw error = 0  ← Looks perfect but isn't!
```

Squaring fixes this:
```
Point 1: squared error = 25
Point 2: squared error = 25
Total squared error = 50  ← Actually shows the error!
```

**Sum of Squared Errors (SSE):**
```
SSE = (error₁)² + (error₂)² + ... + (errorₙ)²
```

**What This Means:**
- Small SSE = good fit (model is close to data)
- Large SSE = poor fit (model is far from data)
- Squaring makes big mistakes more "expensive"

### Why This Matters for Machine Learning

Optimization algorithms **only see the total error**. They don't see:
- The shape of the data
- The individual data points
- What the model looks like visually

The total error (SSE or MSE) is the **entire signal** that guides learning.

### Questions

**Q1.** In your own words, what does "global error" or "loss" measure in this plot?

**Answer:**

<br><br><br>

**Q2.** If we changed the slope or intercept of the line, how would that change the loss?

**Answer:**

<br><br><br>

---

## Module 2: Interactive Line Fitting

**Time:** ~15 minutes
**Type:** In-class

### Learning Objectives

- Explore how changing parameters affects error
- Understand the relationship between local and global error
- Practice minimizing error through experimentation
- Develop intuition for optimization

### What You'll Do

Use interactive sliders to adjust a line's slope (`m`) and intercept (`b`), trying to minimize the total error (SSE).

### Key Concepts

**Parameters:**
- **Slope (m):** How steep the line is
- **Intercept (b):** Where the line crosses the y-axis
- Together they define: `y = m*x + b`

**Residuals:**
- Dashed vertical lines showing error at each point
- Visual representation of local error

**Feedback System:**
- 🔥 **Warmer:** Your change reduced the error (good!)
- 🧊 **Colder:** Your change increased the error (try another direction)
- **Neutral:** No change in error

### Interactive Elements

1. **Slope slider (m):** Range -5 to 5, step 0.1
2. **Intercept slider (b):** Range -5 to 5, step 0.1
3. **Show residuals checkbox:** Toggle error lines on/off
4. **History table:** Shows your recent attempts

### Strategy Tips

Think about:
- Should you adjust slope or intercept first?
- What happens when you move one parameter while keeping the other fixed?
- Can you develop a systematic strategy to find the minimum?

### Questions

**Q3.** How do the residual lines (the dashed vertical lines) help you understand the **local error** at each point?

**Answer:**

<br><br><br>

**Q4.** Can you make the global error small even if a few points still have relatively large errors? Describe a situation where this happens and why.

**Answer:**

<br><br><br>

**Q5 (PREDICTION).** Suppose you add one very extreme outlier point far away from the others. **Predict:** How will this affect the best-fit line and the global error?

**Your Prediction:**

<br><br>

**What Actually Happened (if you tried it):**

<br><br>

---

## Module 3: Parameter Space Optimization

**Time:** ~20 minutes
**Type:** In-class

### Learning Objectives

- Optimize using **only global error feedback** (no visual of data or line)
- Explore parameter space systematically
- Understand how ML algorithms "see" only loss values
- Compare your search strategy to gradient-based optimization

### What You'll Do

Search for the best slope (`m`) and intercept (`b`) by submitting guesses and seeing only the Mean Squared Error (MSE) value – **you won't see the data or the line!**

### Key Concepts

**Parameter Space:**
- A 2D plane where:
  - x-axis = slope `m`
  - y-axis = intercept `b`
- Each point in this space represents a different line
- We want to find the point with the lowest MSE

**MSE (Mean Squared Error):**
```
MSE = SSE / number of points
```
- Normalized version of SSE
- Easier to compare across different datasets

**Blind Optimization:**
- You only see MSE numbers, not the data
- This is how most ML algorithms work!
- They navigate by the "slope" of the error landscape

### What You'll See

**During Exploration:**
- Sliders to choose (m, b)
- A table showing your attempts and their MSE values
- A scatter plot showing your guesses in parameter space (colored by MSE)
- NO view of the data or the actual line

**After Clicking "Done":**
- Full MSE contour landscape
- Your exploration path overlaid
- Grid global minimum (white circle)
- Least-squares solution (yellow X) – the "correct" answer from math
- True parameters used to generate data

### Questions

**Q6.** Describe how your guesses for (m, b) moved over time. Did you follow any systematic strategy (e.g., "move m a bit, then adjust b", or "search in a grid", etc.)?

**Answer:**

<br><br><br>

**Q7.** Look at the MSE landscape plot after clicking "Done". How close was your best guess to:
- (a) the approximate global minimum on the grid?
- (b) the least-squares solution from the data?

What does this tell you about optimizing only based on the global error?

**Answer:**

<br><br><br>

**Q8.** In our earlier line-fitting exercise (Module 2), you could see the data, the line, and the residuals. In this game, you only saw the MSE. How is this situation similar to how many machine learning models are trained, where the algorithm only sees a **loss value** and not the "right answer" in a human-readable way?

**Answer:**

<br><br><br>

---

## Module 4: Hidden Function Optimization

**Time:** ~15 minutes
**Type:** In-class

### Learning Objectives

- Optimize a 1D function without seeing its shape
- Use warm/cold feedback strategically
- Develop systematic search strategies
- Understand that optimization applies beyond line fitting

### What You'll Do

Try to find the **minimum** of a hidden function by choosing x values in the range -10 to 10. You'll only see the function values at the points you try.

### Key Concepts

**Hidden Function:**
- A smooth curve (actually a parabola, but you don't know this!)
- Has one minimum somewhere in the range -10 to 10
- You can only "see" the points where you sample it

**1D Optimization:**
- Simpler than 2D (only one parameter to adjust)
- But still requires strategy!

**Search Strategies You Might Use:**
- **Grid search:** Try evenly spaced points
- **Binary search:** Narrow down the range by halves
- **Random search:** Try random points
- **Hill climbing:** Move in the direction that reduces f(x)

### Interactive Elements

1. **Slider for x:** Range -10 to 10, step 0.2
2. **Try button:** Evaluates the hidden function at your chosen x
3. **Scatter plot:** Shows only your sampled points (not the underlying function!)
4. **Warm/Cold feedback:**
   - 🔥 Warmer: f(x) decreased (closer to minimum)
   - 🧊 Colder: f(x) increased (farther from minimum)
5. **History table:** Your recent attempts

### Questions

**Q9.** What strategies did your group use to choose new values of x within the allowed range?

**Answer:**

<br><br><br>

**Q10.** How did the "warmer/colder" feedback influence your choices?

**Answer:**

<br><br><br>

**Q11.** Imagine that even the scatter plot of your guesses was hidden, and you only saw the table of (x, f(x)). Would you still be able to find a good minimum? How?

**Answer:**

<br><br><br>

---

## Module 5: Mountain Landscape Search

**Time:** ~20 minutes
**Type:** In-class

### Learning Objectives

- Search in 2D space for an optimal point
- Understand **local vs. global maxima** (peaks)
- Recognize that landscapes can have multiple optima
- Connect to ML loss landscapes with many local minima

### What You'll Do

Explore a mountain landscape by sampling (x, y) coordinates and measuring the altitude. Your goal is to find the **highest peak** – but there are **multiple peaks** of different heights!

### Key Concepts

**The Challenge:**
- The landscape has 3-5 peaks (you don't know how many)
- Some peaks are higher than others
- You can only "see" the altitude at points you sample
- You don't see the full landscape until the end

**Local Maximum:**
- The highest point in a local region
- If you're standing on a local peak, moving in any direction goes down
- But there might be a higher peak somewhere else!

**Global Maximum:**
- The highest point in the **entire** landscape
- The peak we're ultimately trying to find

**The Trap:**
- Easy to find **a** peak
- Hard to be sure it's **the highest** peak
- Need to balance exploring new areas vs. refining your current best

### Interactive Elements

1. **Two sliders:**
   - x: Range -5 to 5, step 0.2
   - y: Range -5 to 5, step 0.2
2. **Measure altitude button:** Records (x, y, height) sample
3. **Scatter plot in (x, y) space:**
   - Shows where you've sampled
   - Color indicates altitude (warm colors = higher)
   - Does NOT show the underlying landscape
4. **Best sample tracker:** Highlights your current best find
5. **Done button reveals:**
   - Full mountain landscape (contour plot)
   - All your sampled points
   - Your best sample (cyan circle)
   - True global peak (red star)

### Strategy Considerations

**Exploration vs. Exploitation:**
- **Exploration:** Search new areas to find other peaks
- **Exploitation:** Sample near your current best to refine it
- Balance is crucial!

**Think About:**
- How do you know when to stop searching near one peak and explore elsewhere?
- How can you tell if you've found the global maximum?
- What if the global peak is in a region you never explored?

### Questions

**Q12.** Describe your group's strategy for choosing new (x, y) locations. How did you decide where to sample next after finding a high point?

**Answer:**

<br><br><br>

**Q13 (PREDICTION).** Before clicking "Done": How many local peaks do you think there are? Where do you think the global peak is?

**Your Prediction:**

<br><br>

**After Revealing:** Look at the revealed landscape. How many local peaks can you actually see? Did your group spend most of its time near one peak, or did you explore multiple regions?

**Answer:**

<br><br><br>

**Q14.** Compare your best sample to the true global peak shown on the plot. Were you close to the global maximum, or did you end up stuck near a local maximum?

**Answer:**

<br><br><br>

**Q15.** Explain how this mountain-peak search is similar to what happens in machine learning when an algorithm is trying to optimize a loss function that has many "bumps" (local minima or maxima). What risks does a model face if it only explores one region of the loss landscape?

**Answer:**

<br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 6 modules using the **same group code**
- [ ] Answered all 15 questions (Q1-Q15)
- [ ] Included predictions where asked (Q5, Q13)
- [ ] Described your strategies and observations
- [ ] Discussed your answers with your group members

---

## Key Takeaways

By completing this lab, you should understand:

1. **Error/Loss Measurement**
   - Models are evaluated by total error across all data
   - Squared errors prevent cancellation and penalize large mistakes
   - SSE and MSE are common loss functions

2. **Parameters and Optimization**
   - Models have adjustable parameters
   - Optimization means finding parameter values that minimize loss
   - Different parameter values → different predictions → different loss

3. **Blind Optimization**
   - ML algorithms typically only see loss values
   - They navigate parameter space using loss gradients
   - This is similar to searching in the dark with only a "warmer/colder" signal

4. **Local vs. Global Optima**
   - Complex loss landscapes have multiple "good" solutions
   - Local optima can trap optimization algorithms
   - Finding the global optimum is hard when you can't see the whole landscape

5. **Search Strategies Matter**
   - Systematic exploration beats random guessing
   - Balance between exploitation (refining) and exploration (searching new areas)
   - Real ML uses sophisticated algorithms (gradient descent, Adam, etc.) to navigate efficiently

---

## Connecting to AI and Machine Learning

Everything you did in this lab is what happens inside ML models:

| Lab Activity | ML Equivalent |
|--------------|---------------|
| Adjusting line parameters | Training neural network weights |
| Measuring SSE/MSE | Computing loss function |
| Trying different (m, b) values | Gradient descent iterations |
| Parameter space landscape | Weight space / loss landscape |
| Finding local peak instead of global | Getting stuck in local minimum |
| Warm/cold feedback | Loss gradient direction |

**Next Steps in Your AI Journey:**
- Real ML models have **millions** of parameters
- Loss landscapes are **high-dimensional** (not just 2D or 3D)
- Algorithms use **gradients** (calculus) to navigate efficiently
- Researchers develop clever tricks to avoid local minima

Great work! 🎉

---

**Questions or Issues?**
- Check the LMS discussion board
- Ask your instructor or TA
- Review the lab narrative for additional context
