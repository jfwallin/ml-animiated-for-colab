# Lab 2: Gradient Descent - Automated Optimization
## Student Handout

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In this lab, you'll discover how gradient descent automates the optimization process you manually performed in Lab 1:

1. **The universal update rule** – How one formula powers all of machine learning
2. **Gradient descent in action** – Watching algorithms navigate automatically
3. **Learning rate effects** – Understanding the most critical hyperparameter
4. **Limitations of gradient descent** – Why local optima are still a problem

### Lab Structure

This lab consists of **5 modules** that you'll complete in order using **the same group code from Lab 1**:

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
- ✅ Explain concepts in different ways
- ✅ Help understand error messages
- ✅ Clarify what gradients mean mathematically

You should NOT use AI tools to:
- ❌ Generate complete answers to lab questions
- ❌ Write code for you without understanding it
- ❌ Do the thinking for you

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

**Answer:**

<br><br>

**Q2.** What happens to the step size if:
- (a) The learning rate is very large (e.g., 10.0)?
- (b) The gradient magnitude is very large?
- (c) Both learning rate and gradient are small?

**Answer:**

<br><br><br>

---

## Module 1: GD on Hidden Parabola

**Time:** ~15 minutes
**Type:** In-class

### Learning Objectives

- Apply gradient descent to 1D function optimization
- Understand how learning rate affects convergence
- Compare automated GD vs. manual search from Lab 1
- See the same problem from Lab 1 solved automatically

### What You'll Do

Watch gradient descent automatically find the minimum of the **same hidden parabola from Lab 1 Module 4** – but now without you having to manually search!

### Connection to Lab 1

**In Lab 1 Module 4, you:**
- Manually chose x values with a slider
- Got warm/cold feedback
- Refined your guesses iteratively
- Tried to find the minimum through trial and error

**Today with Gradient Descent:**
- GD computes the slope (gradient) automatically
- GD moves downhill automatically
- You only control the learning rate
- Watch it find the minimum in seconds!

### Key Concepts

**Numerical Gradient:**
```python
gradient ≈ (f(x + tiny_step) - f(x - tiny_step)) / (2 × tiny_step)
```
- We approximate the slope without calculus
- Measures how the function changes around the current point
- Direction and magnitude tell us how to move

**GD Update Process:**
1. Start at initial position (e.g., x = 0)
2. Compute gradient at current position
3. Take a step: `new_x = old_x - learning_rate × gradient`
4. Repeat until converged (or max iterations reached)

**Convergence:**
- Steps get smaller as you approach the minimum (gradient → 0)
- Function value decreases (going downhill)
- Eventually, changes become tiny → convergence!

### Interactive Elements

**Interactive Tool:**
1. **Starting x slider:** Choose where GD begins
2. **Learning rate input:** Control step size
3. **Run 1 Step button:** See one GD update
4. **Run 10 Steps button:** See multiple updates
5. **Reset button:** Start over with new parameters

**Visualizations:**
- **Left plot:** GD path in (x, f(x)) space (dots connected by arrows)
- **Right plot:** Function value over iterations (convergence curve)
- **Table:** Detailed step-by-step information

### Learning Rates to Try

**Five learning rates to explore:** 0.01, 0.05, 0.4, 1.0, 3.0

**Before running, predict what will happen with each!**

### Questions

**Q3 (PREDICTION).** Starting from x = 0.0, predict what will happen with five learning rates:
- **LR = 0.01:** Will this converge quickly or slowly?
- **LR = 0.05:** Will this converge faster than 0.01?
- **LR = 0.4:** Will this converge faster? Any risks?
- **LR = 1.0:** Will this be fastest? Or cause problems?
- **LR = 3.0:** What do you expect with such a large learning rate?

**Your Prediction:**

<br><br><br>

**What Actually Happened:**

<br><br><br>

**Q4.** Compare gradient descent to your manual search in Lab 1 Module 4:
- Which was faster at finding the minimum?
- Which required more "attempts" or "steps"?
- What advantage does GD have over manual guessing?

**Answer:**

<br><br><br>

**Q5.** Based on the visualizations:
- How does step size relate to: (a) the slope (gradient) magnitude, and (b) the learning rate?
- Why do the steps get smaller as you approach the minimum?
- What happens with LR = 1.0? Does it converge smoothly?
- What happens with LR = 3.0? Can you explain this behavior?

**Answer:**

<br><br><br>

---

## Module 2: GD on Parameter Space (Line Fitting)

**Time:** ~20 minutes
**Type:** In-class

### Learning Objectives

- Apply GD to 2D parameter optimization
- Navigate the MSE landscape automatically
- Compare GD path to your manual exploration in Lab 1
- Understand multi-parameter gradient descent

### What You'll Do

Watch gradient descent automatically optimize the slope (`m`) and intercept (`b`) to fit the **same line from Lab 1 Modules 2-3**.

### Connection to Lab 1

**In Lab 1 Module 3, you:**
- Manually chose (m, b) values
- Only saw MSE numbers (no data visualization)
- Tried to navigate the parameter space by trial and error

**Today with Gradient Descent:**
- GD computes gradients for both `m` and `b`
- GD navigates the MSE landscape automatically
- You watch the path it takes on a contour plot

### Key Concepts

**Gradient in 2D:**
- Gradient is now a **vector** with two components:
  - ∂MSE/∂m: How MSE changes with slope
  - ∂MSE/∂b: How MSE changes with intercept
- Points in the direction of steepest increase
- GD moves opposite to this direction

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

**During GD:**
- **Left plot:** MSE contour map with GD path
  - Contours show MSE landscape
  - Colored path shows GD trajectory
  - Start (green) → End (red)
- **Right plot:** MSE over iterations
  - Shows convergence progress
  - Should decrease monotonically (always going down)

**Comparison with Lab 1:**
- Your manual guesses from Lab 1 (if saved)
- GD path
- Least-squares solution (optimal)

### Interactive Elements

1. **Starting point sliders:** Choose initial (m, b)
2. **Learning rate input:** Control step size
3. **Number of steps:** How long to run GD
4. **Run GD button:** Execute gradient descent
5. **Learning rate comparison:** See LR = {0.01, 0.1, 0.5} side-by-side

### Questions

**Q6 (PREDICTION).** Starting from (0, 0), predict:
- Will the GD path be straight or curved? Why?
- Will GD find the same minimum you found in Lab 1?

**Your Prediction:**

<br><br>

**What Actually Happened:**

<br><br>

**Q7.** Describe the shape of the GD path on the MSE contour plot:
- Is it straight or curved? Why?
- What happens to step size as GD approaches the minimum?

**Answer:**

<br><br><br>

**Q8.** Compare GD to your manual exploration in Lab 1 Module 3:
- Which was more efficient?
- How many guesses did you make in Lab 1 vs. GD steps needed?

**Answer:**

<br><br><br>

**Q9.** Based on the learning rate comparison (LR = 0.01, 0.1, 0.5):
- Which LR converged fastest?
- What happens with LR = 0.01? (Too slow?)
- What happens with LR = 0.5? (Oscillation? Divergence?)

**Answer:**

<br><br><br>

---

## Module 3: Learning Rate Exploration

**Time:** ~20 minutes
**Type:** In-class

### Learning Objectives

- Understand the critical role of learning rate in GD
- Predict and observe: slow convergence, fast convergence, oscillation, divergence
- Develop intuition for choosing learning rates
- Recognize learning rate as the most important hyperparameter

### What You'll Do

Deep dive into learning rate effects using a simple test function: `f(x) = 0.5 × x²`

This function has:
- Minimum at x = 0 where f(0) = 0
- Simple parabolic shape
- Clear convergence behavior

### The Goldilocks Problem

**Learning rate trade-offs:**

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

### Key Concepts

**Slow Convergence (LR too small):**
- Each step is tiny
- Makes progress but wastes time/computation
- Eventually reaches minimum if given enough steps
- In real ML: costs money, energy, time

**Optimal Convergence (LR just right):**
- Large enough steps for fast progress
- Small enough to remain stable
- Smooth monotonic decrease in loss
- "Goldilocks zone"

**Oscillation (LR too large):**
- Overshoots the minimum
- Bounces back and forth
- Might still converge but inefficiently
- Loss might increase temporarily

**Divergence (LR way too large):**
- Steps are so large you jump farther away
- Function value explodes
- Never converges
- Complete failure!

### Visualizations

**Four-panel comparison:**
- Each subplot shows one learning rate
- GD path overlaid on the function f(x) = 0.5x²
- Clear comparison of behaviors

**Convergence curves:**
- **Linear scale:** Shows absolute values
- **Log scale:** Better for seeing convergence rates
- Compare all four learning rates

**Summary table:**
- Steps taken
- Final x value
- Final f(x) value
- Convergence status

### Interactive Exploration

**Try your own learning rates:**
- Find the **largest LR that still converges**
- Find the **smallest LR that converges in < 50 steps**
- Find an LR that causes **dramatic divergence**

### Questions

**Q10 (PREDICTION).** Starting from x = 10.0, predict for each learning rate:
- **LR = 0.001:** Will it converge in 100 steps? Roughly how many steps needed?
- **LR = 0.1:** Will it converge quickly? How many steps?
- **LR = 0.8:** Will it converge? Oscillate? Diverge?
- **LR = 3.0:** Will it converge at all? What do you expect?

**Your Prediction:**

<br><br><br>

**What Actually Happened:**

<br><br><br>

**Q11.** Describe the behavior for each LR category:
- **Too small (0.001):** What happens? Why is this wasteful?
- **Just right (0.1):** What makes this optimal?
- **Too large (0.8):** What problems occur? Why?
- **Way too large (3.0):** How quickly does it diverge? What does this tell you?

**Answer:**

<br><br><br><br>

**Q12.** How would you choose a learning rate for a new optimization problem?
- What strategy would you use?
- What signs indicate your LR is too large? Too small?
- Why is learning rate called a "hyperparameter"?

**Answer:**

<br><br><br>

---

## Module 4: Mountain Landscape - GD Limitations

**Time:** ~15 minutes
**Type:** In-class

### Learning Objectives

- Understand that GD gets stuck at local optima
- See that starting position determines which peak/valley you reach
- Recognize this as a fundamental limitation of gradient-based optimization
- Connect to neural network training challenges

### What You'll Do

Run **gradient ascent** (uphill climbing) from multiple starting points on the **same mountain landscape from Lab 1 Module 5**.

**Gradient Ascent vs. Descent:**
- Gradient **descent** finds minima (valleys) – for minimizing loss
- Gradient **ascent** finds maxima (peaks) – for this mountain exploration
- Same algorithm, opposite sign: `new = old + learning_rate × gradient`

### Connection to Lab 1

**In Lab 1 Module 5, you:**
- Manually explored a mountain landscape
- Tried to find the highest peak
- Discovered multiple local peaks

**Today with Gradient Ascent:**
- GD automatically climbs uphill
- Starting position determines which peak you reach
- GD can't "see" distant peaks – only local slope

### Key Concepts

**Local vs. Global Optima:**

**Local Maximum:**
- Highest point in a nearby region
- All neighboring points are lower
- GD stops here (gradient becomes zero)
- But there might be higher peaks elsewhere!

**Global Maximum:**
- The absolute highest point on the landscape
- What we're trying to find
- GD can miss this if it starts near a different peak

**The Fundamental Problem:**
- GD only uses **local information** (gradient at current point)
- Can't "see" the entire landscape
- Gets stuck at the first peak it climbs
- Starting position is critical!

### Interactive Elements

**Multiple Starting Points:**
- Try 4-6 different starting positions
- Each runs gradient ascent independently
- See which peak each one reaches
- Compare final heights

**Visualizations:**
- **During exploration:** Path from each starting point
- **After completion:**
  - Full mountain landscape (contour plot)
  - All GD paths overlaid
  - Final positions marked
  - True global peak highlighted

### Strategy Considerations

**How to overcome local optima:**
- **Multiple random starts:** Try many different starting points
- **Simulated annealing:** Occasionally accept uphill moves
- **Momentum:** Use velocity to escape shallow peaks
- **Adaptive learning rates:** Adjust step size during search

In real neural networks:
- Researchers use **random initialization** (try many starting points)
- **Momentum** and **Adam** optimizer help escape shallow minima
- Sometimes local minima are "good enough"
- Architecture design helps create "nicer" landscapes

### Questions

**Q13 (PREDICTION).** Before running gradient ascent:
- Starting at (1, 1): Which peak will GD reach?
- Will it find the global maximum? Why or why not?
- What if you start at (-2, 3)?

**Your Prediction:**

<br><br>

**What Actually Happened:** How many different peaks did you reach from different starting points?

<br><br>

**Q14.** Based on your experiments:
- Did gradient ascent find the global maximum?
- Why or why not?
- How did starting position affect which peak was reached?
- Can GD "see" distant peaks? Explain.

**Answer:**

<br><br><br><br>

**Q15.** Connection to neural network training:
- How is this mountain problem similar to training a neural network?
- What strategies might help overcome getting stuck at local optima?
- Why is neural network optimization still successful despite local minima?

**Answer:**

<br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules using **the same group code from Lab 1**
- [ ] Answered all 15 questions (Q1-Q15)
- [ ] Included predictions where asked (Q3, Q6, Q10, Q13)
- [ ] Described your observations and insights
- [ ] Compared GD behavior to your Lab 1 manual exploration
- [ ] Discussed your answers with your group members

---

## Key Takeaways

By completing this lab, you should understand:

### 1. The Universal Update Rule

**Formula:** `new = old - learning_rate × gradient`

- Powers all gradient descent algorithms
- Works for any differentiable function
- Scales from 1 parameter to billions of parameters
- Used in training all modern neural networks

### 2. Gradient Descent Automates Lab 1

| Lab 1 Manual Process | Lab 2 Gradient Descent |
|---------------------|------------------------|
| You chose parameters by hand | GD computes optimal direction |
| Got warm/cold feedback | Uses gradient (exact slope) |
| Trial and error search | Systematic downhill movement |
| Many guesses needed | Converges in fewer steps |
| Required intuition | Purely algorithmic |

### 3. Learning Rate is Critical

**Too small:**
- Wastes computation
- Slow progress
- Eventually converges (if patient enough)

**Just right:**
- Fast convergence
- Stable descent
- Optimal performance

**Too large:**
- Oscillation
- Possible divergence
- Unstable, unpredictable

**Way too large:**
- Rapid divergence
- Complete failure
- Loss explodes

### 4. Local Optima Problem

**The Challenge:**
- GD only uses local gradient information
- Gets stuck at first minimum/maximum it reaches
- Starting position determines final solution
- Can't "see" the global landscape

**Solutions in Practice:**
- Multiple random initializations
- Momentum-based optimizers
- Adaptive learning rates (Adam, RMSprop)
- Better architecture design
- Sometimes "good enough" local minima

### 5. Trade-offs and Hyperparameters

**Hyperparameters you control:**
- Learning rate (most important!)
- Number of iterations
- Starting position
- Batch size (in mini-batch GD)

**Trade-offs to balance:**
- Speed vs. stability
- Exploration vs. exploitation
- Computation cost vs. solution quality

---

## Connecting to AI and Machine Learning

Everything you learned applies directly to training neural networks:

| Lab 2 Activity | Neural Network Training |
|---------------|-------------------------|
| GD on parabola (1 parameter) | Updating one weight |
| GD on line fitting (2 parameters) | Updating multiple weights |
| Learning rate experimentation | Hyperparameter tuning |
| Local optima on mountain | Getting stuck during training |
| Convergence curves | Training loss plots |
| Gradient computation | Backpropagation |

### Real-World Gradient Descent

**In practice, neural networks:**
- Have **millions to billions** of parameters
- Use **mini-batch gradient descent** (not full-batch)
- Apply **adaptive learning rates** (Adam, RMSprop)
- Include **momentum** to smooth updates
- Use **learning rate schedules** (start high, decay over time)
- Employ **regularization** to prevent overfitting
- Sometimes use **gradient clipping** to prevent explosions

### Why GD Works Despite Limitations

Even with local minima problems:
- High-dimensional spaces have fewer "bad" local minima
- Modern architectures create smoother landscapes
- Good local minima are often sufficient
- Practical tricks (momentum, adaptive LR) help navigation
- Stochastic gradient descent adds beneficial noise

### Next Steps in Your AI Journey

**Building on this lab:**
- Neural networks use GD to learn from data
- Deep learning = GD + clever architectures
- Understanding GD helps debug training problems
- Learning rate tuning is an art and science
- All modern AI relies on variations of gradient descent

**Future topics:**
- Stochastic gradient descent (SGD)
- Mini-batch training
- Backpropagation (computing gradients efficiently)
- Optimizers: Adam, RMSprop, AdaGrad
- Learning rate schedules and warm-up
- Regularization and generalization

Great work! You now understand the algorithm that powers modern AI! 🎉

---

## Advanced Insights (Optional)

### Why Does GD Converge?

For **convex functions** (bowl-shaped like MSE for linear regression):
- GD is **guaranteed** to reach the global minimum
- Step size decreases naturally as gradient → 0
- Convergence rate depends on learning rate and landscape shape

For **non-convex functions** (like neural network loss):
- Convergence to **global** minimum is not guaranteed
- But convergence to **a** minimum is usually achieved
- Theory is complex, but practice works well!

### Learning Rate Schedules

Instead of fixed learning rate, use schedules:

**Step decay:** Reduce LR every N epochs
```
epoch 0-10: LR = 0.1
epoch 11-20: LR = 0.01
epoch 21-30: LR = 0.001
```

**Exponential decay:** `LR = initial_LR × decay_rate^epoch`

**Cosine annealing:** LR follows cosine curve

**Warm-up:** Start with tiny LR, gradually increase

### Momentum

Adds "velocity" to GD updates:
```
velocity = momentum × old_velocity - learning_rate × gradient
new_parameter = old_parameter + velocity
```

**Benefits:**
- Smooths out oscillations
- Accelerates through flat regions
- Helps escape shallow local minima
- Used in almost all modern neural network training

### Adaptive Learning Rates

**Adam optimizer** (most popular):
- Adapts learning rate for each parameter individually
- Combines momentum + adaptive learning rates
- Usually works well with minimal tuning
- Default choice for many practitioners

---

**Questions or Issues?**
- Check the LMS discussion board
- Ask your instructor or TA
- Review the lab modules for interactive exploration
- Experiment with the notebooks to deepen understanding
