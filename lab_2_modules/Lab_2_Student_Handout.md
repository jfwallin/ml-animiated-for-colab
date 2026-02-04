# Lab 2: Gradient Descent - Automated Optimization
## Student Handout

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In this lab, you'll discover how gradient descent automates the optimization process:

1. **The universal update rule** – How one formula powers all of machine learning
2. **Gradient descent in action** – Watching algorithms navigate automatically
3. **Learning rate effects** – Understanding the most critical hyperparameter
4. **Limitations of gradient descent** – Why local optima are still a problem

### Lab Structure

This lab consists of **5 modules** that you'll complete in order:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | Setup and The Update Rule | ~5 min | Prelab |
| 1 | GD on Hidden Parabola | ~15 min | In-class |
| 2 | GD on Parameter Space (Line Fitting) | ~20 min | In-class |
| 3 | Learning Rate Exploration | ~20 min | In-class |
| 4 | Mountain Landscape - GD Limitations | ~15 min | In-class |

**Total Time:** ~75 minutes

### Working in Groups

- Work in **small groups** (2-4 people)
- Use the **same group code from Lab 1**
- One person shares their screen running the notebooks
- Everyone participates in discussion
- All group members answer questions together

### Key Concept

> **GRADIENT DESCENT is an algorithm that:**
> - Automatically finds parameter values that minimize loss
> - Uses the **gradient** (slope/direction of steepest increase) to decide which way to move
> - Follows the universal rule: `new = old - learning_rate × gradient`
> - Powers the training of virtually all modern machine learning models

### AI Use Policy

You may use AI tools (ChatGPT, Claude, etc.) to:
- Explain concepts in different ways
- Help understand error messages
- Clarify what gradients mean mathematically

You should NOT use AI tools to:
- Generate complete answers to lab questions
- Write code for you without understanding it
- Do the thinking for you

**Remember:** The goal is to develop your own understanding of how gradient descent works.

---

## Module 0: Setup & The Update Rule

**Time:** ~5 minutes
**Type:** Prelab

### What You'll Do

Learn the universal update rule that powers all gradient descent algorithms and see a simple visualization of how it works.

### The Universal Update Rule

All gradient descent follows one simple formula:

```
new_parameter = old_parameter - learning_rate × gradient
```

**Breaking it down:**

- **old_parameter**: Where you currently are
- **gradient**: The slope/direction of steepest increase in loss
- **learning_rate**: How big a step to take
- **negative sign**: Move opposite to the gradient (downhill!)
- **new_parameter**: Where you'll be after this update

### Key Concepts

**What is a Gradient?**
- The gradient is the **slope** of the loss function at your current location
- Mathematically: the derivative (rate of change)
- Intuitively: which direction is "uphill"
- **Positive gradient** → Loss increases to the right → Move left
- **Negative gradient** → Loss increases to the left → Move right

**Why the Negative Sign?**
- Gradient points **uphill** (direction of steepest increase)
- We want to go **downhill** (minimize loss)
- Subtracting the gradient moves us downhill

**Learning Rate:**
- Controls step size
- Too small = slow progress
- Too large = unstable, might diverge
- "Just right" = fast and stable convergence

### Visual Example

If you're at `x = 5` on a function:
1. Compute gradient: `gradient = +2` (slope is positive, pointing uphill)
2. Choose learning rate: `learning_rate = 0.1`
3. Update: `new_x = 5 - 0.1 × 2 = 4.8`
4. Result: Moved left (downhill) by 0.2 units

### Questions

**Q1.** If the gradient (slope) at a point is **positive**, which direction does gradient descent move? Why?

**Q2.** What happens to the step size if:
- (a) The learning rate is very large (e.g., 10.0)?
- (b) The gradient magnitude is very large?
- (c) Both learning rate and gradient are small?

---

## Module 1: GD on Hidden Parabola

**Time:** ~15 minutes
**Type:** In-class

### What You'll Do

Watch gradient descent automatically find the minimum of a hidden 1D function. Previously you would have had to search for this by hand — now GD does it for you.

### Key Concepts

**Numerical Gradient:**
```python
gradient ≈ (f(x + tiny_step) - f(x - tiny_step)) / (2 × tiny_step)
```
- We approximate the slope without calculus
- Direction and magnitude tell us how to move

**GD Update Process:**
1. Start at initial position (e.g., x = 0)
2. Compute gradient at current position
3. Take a step: `new_x = old_x - learning_rate × gradient`
4. Repeat until converged

**Convergence:**
- Steps get smaller as you approach the minimum (gradient approaches 0)
- Function value decreases (going downhill)
- Eventually, changes become tiny — convergence!

### Interactive Elements

1. **Starting x slider:** Choose where GD begins
2. **Learning rate input:** Control step size
3. **Run 1 Step / Run 10 Steps:** Watch GD updates
4. **Visualizations:** GD path and convergence curve

### Learning Rates to Try

**Five learning rates to explore:** 0.01, 0.05, 0.4, 1.0, 3.0

**Before running, predict what will happen with each!**

### Questions

**Q3 (PREDICTION).** Starting from x = 0.0, predict what will happen with five learning rates (0.01, 0.05, 0.4, 1.0, 3.0). Which will converge? Which might cause problems?

**Q4.** Based on the visualizations:
- How does step size relate to (a) gradient magnitude and (b) learning rate?
- Why do steps get smaller near the minimum?
- Describe what happens with LR = 1.0 and LR = 3.0.

---

## Module 2: GD on Parameter Space (Line Fitting)

**Time:** ~20 minutes
**Type:** In-class

### What You'll Do

Apply GD to optimize **two parameters simultaneously** — slope (`m`) and intercept (`b`) — to fit a line to data. Watch GD navigate the MSE landscape on a contour plot.

### Key Concepts

**Gradient in 2D:**
- Gradient is now a **vector** with two components:
  - ∂MSE/∂m: How MSE changes with slope
  - ∂MSE/∂b: How MSE changes with intercept
- The gradient has both **magnitude** (how steep) and **direction** (which way is steepest)
- GD moves opposite to this direction — the direction of steepest descent

**Update Rule (2 parameters):**
```
new_m = old_m - learning_rate × ∂MSE/∂m
new_b = old_b - learning_rate × ∂MSE/∂b
```
Both parameters update simultaneously!

**MSE Landscape:**
- Contour lines connect points with equal MSE
- Shaped like a bowl (quadratic function)
- One global minimum (optimal m, b)
- GD follows a curved path toward the minimum

### Visualizations

- **Left plot:** MSE contour map with GD path (start → end)
- **Right plot:** MSE over iterations (convergence progress)

### Questions

**Q5 (PREDICTION).** Starting from (0, 0), predict: will the GD path be straight or curved? Why?

**Q6.** Describe the GD path and the learning rate comparison:
- Is the path straight or curved? Why?
- What happens to step size near the minimum?
- Which learning rate (0.01, 0.1, 0.5) converged fastest, and what went wrong with the others?

---

## Module 3: Learning Rate Exploration

**Time:** ~20 minutes
**Type:** In-class

### What You'll Do

Deep dive into learning rate effects using a simple test function: `f(x) = 0.5 × x²`

This function has:
- Minimum at x = 0 where f(0) = 0
- Simple parabolic shape
- Clear convergence behavior

### The Goldilocks Problem

| Learning Rate | Speed | Stability | Outcome |
|--------------|-------|-----------|----------|
| Too small | Very slow | Very stable | Wastes computation |
| Optimal | Fast | Stable | Best performance |
| Too large | Fast initially | Unstable | Oscillation/divergence |

### Four Learning Rates to Compare

Starting from **x = 10.0**, you'll test:
1. **LR = 0.001** (very small) – Stable but painfully slow
2. **LR = 0.1** (just right) – Fast and stable
3. **LR = 0.8** (large) – Risky, might oscillate
4. **LR = 3.0** (very large) – Likely to diverge

### Questions

**Q7 (PREDICTION).** Starting from x = 10.0, predict: which learning rates will converge, which will oscillate, and which will diverge?

**Q8.** Describe the behavior for each LR category:
- **Too small (0.001):** What happens? Why is this wasteful?
- **Just right (0.1):** What makes this work well?
- **Too large (0.8):** What problems occur?
- **Way too large (3.0):** What does divergence look like?

**Q9.** How would you choose a learning rate for a new problem? What signs tell you it's too large or too small?

---

## Module 4: Mountain Landscape - GD Limitations

**Time:** ~15 minutes
**Type:** In-class

### What You'll Do

Run **gradient ascent** (uphill climbing) from multiple starting points on a landscape with multiple peaks.

**Gradient Ascent vs. Descent:**
- Gradient **descent** finds minima (valleys) – for minimizing loss
- Gradient **ascent** finds maxima (peaks) – for this mountain exploration
- Same algorithm, opposite sign: `new = old + learning_rate × gradient`

### Key Concepts

**Local vs. Global Optima:**

- **Local Maximum:** Highest point in a nearby region — GD stops here because the gradient is zero, but higher peaks may exist elsewhere.
- **Global Maximum:** The absolute highest point on the landscape — what we want, but GD may never reach it.

**The Fundamental Problem:**
- GD only uses **local information** (gradient at current point)
- Can't "see" the entire landscape
- Gets stuck at the first peak it climbs
- Starting position is critical!

### Strategy Considerations

**How to overcome local optima in practice:**
- **Multiple random starts:** Try many different starting points
- **Momentum:** Use velocity to push past shallow peaks
- **Adaptive learning rates:** Adam, RMSprop adjust step size automatically
- **Stochastic GD:** Random noise helps escape shallow minima

### Questions

**Q10 (PREDICTION).** Starting at (1, 1): will GD find the global maximum? Why or why not? Will different starting points reach different peaks?

**Q11.** Based on your experiments:
- Did gradient ascent find the global maximum from every starting point?
- Why can't GD "see" distant peaks?
- What strategies might help overcome this limitation?

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules using **the same group code from Lab 1**
- [ ] Answered all 11 questions (Q1-Q11)
- [ ] Included predictions where asked (Q3, Q5, Q7, Q10)
- [ ] Described your observations and insights

---

## Key Takeaways

### 1. The Universal Update Rule

**Formula:** `new = old - learning_rate × gradient`

- Powers all gradient descent algorithms
- Works for any differentiable function
- Scales from 1 parameter to billions of parameters

### 2. From 1D to 2D and Beyond

| 1D (Module 1) | 2D (Module 2) | Neural Networks |
|---------------|---------------|-----------------|
| Gradient is a single number (slope) | Gradient is a vector with direction | Gradient has millions of components |
| Move left or right | Move in any compass direction | Move in high-dimensional space |
| One parameter to update | Two parameters update together | All weights update simultaneously |

### 3. Learning Rate is Critical

- **Too small:** Wastes computation, slow progress
- **Just right:** Fast and stable convergence
- **Too large:** Oscillation, possible divergence

### 4. Local Optima Problem

- GD only uses local gradient information
- Gets stuck at the first minimum/maximum it reaches
- Starting position determines final solution
- Solutions: random restarts, momentum, adaptive methods

### Connecting to AI and Machine Learning

| Lab 2 Activity | Neural Network Training |
|---------------|-------------------------|
| GD on parabola (1 parameter) | Updating one weight |
| GD on line fitting (2 parameters) | Updating multiple weights |
| Learning rate experimentation | Hyperparameter tuning |
| Local optima on mountain | Getting stuck during training |
| Convergence curves | Training loss plots |
| Gradient computation | Backpropagation |

**In practice, neural networks** use mini-batch gradient descent, adaptive learning rates (Adam), momentum, learning rate schedules, and regularization — all building on the core ideas from this lab.

---

**Questions or Issues?**
- Check the LMS discussion board
- Ask your instructor or TA
- Experiment with the notebooks to deepen understanding
