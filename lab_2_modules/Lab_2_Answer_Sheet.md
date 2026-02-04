# Lab 2: Gradient Descent
## Answer Sheet

**Course:** DATA 1010 – AI in Action
**Name(s):** _________________________________ **Group Code:** _______

---

## Overview

**What you'll learn:** How gradient descent automates optimization, the universal update rule, and why learning rate matters.

**Key Concept:** Gradient descent uses the rule `new = old - learning_rate × gradient` to automatically navigate toward minima, replacing manual search with systematic descent.

**Lab Structure:** 5 modules (0-4) using the same group code throughout.

---

## Module 0: Setup & The Update Rule (~5 min)

**Concepts:**
- **Universal update rule:** `new = old - learning_rate × gradient`
- **Gradient** = local slope = direction of steepest ascent
- **Learning rate** = step size multiplier
- Move **downhill** = negate gradient

**Q1.** If the gradient (slope) at a point is **positive**, which direction does gradient descent move? Why?

<br><br>

**Q2.** What happens to the step size if:
- (a) The learning rate is very large (e.g., 10.0)?
- (b) The gradient magnitude is very large?
- (c) Both learning rate and gradient are small?

<br><br><br>

---

## Module 1: GD on Hidden Parabola (~15 min)

**What you do:** Apply GD to a hidden 1D function. Watch it automatically find the minimum you would have had to search for by hand.

**Q3 - PREDICTION:** Starting from x = 0.0, predict what will happen with five learning rates:
- **LR = 0.01:** Will this converge quickly or slowly?
- **LR = 0.05:** Faster than 0.01?
- **LR = 0.4:** Any risks?
- **LR = 1.0:** Fastest, or problems?
- **LR = 3.0:** What do you expect?

**Prediction:**

<br><br><br>

**Result after running:**

<br><br><br>

**Q4.** Based on the visualizations:
- How does step size relate to (a) gradient magnitude and (b) learning rate?
- Why do steps get smaller near the minimum?
- Describe what happens with LR = 1.0 and LR = 3.0.

<br><br><br>

---

## Module 2: GD on Parameter Space (Line Fitting) (~20 min)

**What you do:** Apply GD to optimize two parameters (m, b) simultaneously. Watch GD navigate the MSE landscape.

**Q5 - PREDICTION:** Starting from (0, 0), predict:
- Will the GD path be straight or curved? Why?

**Prediction:**

<br><br>

**Result:**

<br><br>

**Q6.** Describe the GD path and the learning rate comparison:
- Is the path straight or curved? Why?
- What happens to step size near the minimum?
- Which learning rate (0.01, 0.1, 0.5) converged fastest, and what went wrong with the others?

<br><br><br>

---

## Module 3: Learning Rate Exploration (~20 min)

**What you do:** Deep dive into learning rate effects using a simple function. Run GD with LR = {0.001, 0.1, 0.8, 3.0}.

**Q7 - PREDICTION:** Starting from x = 10.0, predict for each learning rate:
- **LR = 0.001 (very small):** Will it converge in 100 steps?
- **LR = 0.1 (moderate):** Fast convergence?
- **LR = 0.8 (large):** Converge, oscillate, or diverge?
- **LR = 3.0 (very large):** What do you expect?

**Prediction:**

<br><br><br>

**Result after running:**

<br><br><br>

**Q8.** Describe the behavior for each LR category:
- **Too small (0.001):** What happens? Why is this wasteful?
- **Just right (0.1):** What makes this work well?
- **Too large (0.8):** What problems occur?
- **Way too large (3.0):** What does divergence look like?

<br><br><br>

**Q9.** How would you choose a learning rate for a new problem? What signs tell you it's too large or too small?

<br><br><br>

---

## Module 4: Mountain Landscape - GD Limitations (~15 min)

**What you do:** Run gradient ascent (uphill climbing) from multiple starting points on a landscape with several peaks.

**Q10 - PREDICTION:** Before running gradient ascent:
- Starting at (1, 1): Will GD find the global maximum? Why or why not?
- Will different starting points reach different peaks?

**Prediction:**

<br><br>

**Result after running:** How many different peaks did you reach from different starting points?

<br><br>

**Q11.** Based on your experiments:
- Did gradient ascent find the global maximum from every starting point?
- Why can't GD "see" distant peaks?
- What strategies might help overcome this limitation?

<br><br><br>

---

## Key Takeaways

- **Universal update rule:** `new = old - learning_rate × gradient` works for any optimization

- **GD automates search:** Replaces manual exploration with systematic gradient-following

- **Learning rate is critical:** Too small = slow, too large = unstable, "just right" = optimal

- **Local optima problem:** GD gets stuck at the first peak/valley it reaches

- **Starting point matters:** Different initializations lead to different solutions

**Connection to ML:** Everything you learned applies to training neural networks with millions of parameters navigating complex loss landscapes!
