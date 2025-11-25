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

**What you do:** Apply GD to the same hidden function from Lab 1 Module 4. Watch it automatically find the minimum.

**Q3 - PREDICTION:** ⚡ Starting from x = 0.0, predict what will happen with five learning rates:
- **LR = 0.01:** Will this converge quickly or slowly?
- **LR = 0.05:** Will this converge faster than 0.01?
- **LR = 0.4:** Will this converge faster? Any risks?
- **LR = 1.0:** Will this be fastest? Or cause problems?
- **LR = 3.0:** What do you expect with such a large learning rate?

**Prediction:**

<br><br><br>

**Result after running:**

<br><br><br>

**Q4.** Compare gradient descent to your manual search in Lab 1 Module 4:
- Which was faster at finding the minimum?
- What advantage does GD have over manual guessing?

<br><br><br>

**Q5.** Based on the visualizations:
- How does step size relate to (a) gradient magnitude and (b) learning rate?
- Why do steps get smaller as you approach the minimum?
- What happens with LR = 1.0? Does it converge smoothly?
- What happens with LR = 3.0? Can you explain this behavior?

<br><br><br>

---

## Module 2: GD on Parameter Space (Line Fitting) (~20 min)

**What you do:** Apply GD to optimize (m, b) for line fitting (same data as Lab 1 Modules 2-3). Watch GD navigate the MSE landscape.

**Q6 - PREDICTION:** ⚡ Starting from (0, 0), predict:
- Will the path be straight or curved? Why?
- Will GD find the same minimum you found in Lab 1?

**Prediction:**

<br><br>

**Result:**

<br><br>

**Q7.** Describe the shape of the GD path on the MSE contour plot:
- Is it straight or curved? Why?
- What happens to step size as GD approaches minimum?

<br><br><br>

**Q8.** Compare GD to your manual exploration in Lab 1 Module 3:
- Which was more efficient?
- How many guesses did you make in Lab 1 vs. GD steps?

<br><br><br>

**Q9.** Based on the learning rate comparison (LR = 0.01, 0.1, 0.5):
- Which LR converged fastest?
- What happens with LR = 0.01? (Too slow?)
- What happens with LR = 0.5? (Oscillation? Divergence?)

<br><br><br>

---

## Module 3: Learning Rate Exploration (~20 min)

**What you do:** Deep dive into learning rate effects using a simple function. Run GD with LR = {0.001, 0.1, 0.8, 3.0}.

**Q10 - PREDICTION:** ⚡ Starting from x = 10.0, predict for each learning rate:
- **LR = 0.001 (very small):** Will it converge in 100 steps? How many steps needed?
- **LR = 0.1 (moderate):** Will it converge quickly? How many steps?
- **LR = 0.8 (large):** Will it converge? Oscillate? Diverge?
- **LR = 3.0 (very large):** Will it converge at all? What do you expect?

**Prediction:**

<br><br><br><br>

**Result after running:**

<br><br><br><br>

**Q11.** Describe the behavior for each LR category:
- **Too small (0.001):** What happens? Why is this wasteful?
- **Just right (0.1):** What makes this optimal?
- **Too large (0.8):** What problems occur? Why?
- **Way too large (3.0):** How quickly does it diverge? What does this tell you?

<br><br><br><br>

**Q12.** How would you choose a learning rate for a new optimization problem?
- What strategy would you use?
- What signs indicate your LR is too large? Too small?

<br><br><br>

---

## Module 4: Mountain Landscape - GD Limitations (~15 min)

**What you do:** Run gradient ascent (uphill climbing) from multiple starting points on the same mountain landscape from Lab 1 Module 5.

**Q13 - PREDICTION:** ⚡ Before running gradient ascent:
- Starting at (1, 1): Which peak will GD reach?
- Will it find the global maximum? Why or why not?
- What if you start at (-2, 3)?

**Prediction:**

<br><br>

**Result after running:** How many different peaks did you reach from different starting points?

<br><br>

**Q14.** Based on your experiments:
- Did gradient ascent find the global maximum?
- Why or why not?
- How did starting position affect which peak was reached?
- Can GD "see" distant peaks? Explain.

<br><br><br><br>

**Q15.** Connection to neural network training:
- How is this mountain problem similar to training a neural network?
- What strategies might help overcome getting stuck at local optima?
- Why is neural network optimization still successful despite local minima?

<br><br><br><br>

---

## Key Takeaways

✓ **Universal update rule:** `new = old - learning_rate × gradient` works for any optimization

✓ **GD automates Lab 1:** Replaces manual search with systematic gradient-following

✓ **Learning rate is critical:** Too small = slow, too large = unstable, "just right" = optimal

✓ **Local optima problem:** GD gets stuck at first peak/valley it reaches

✓ **Starting point matters:** Different initializations lead to different local optima

✓ **Trade-offs:** Speed vs. stability, exploration vs. exploitation

**Connection to ML:** Everything you learned applies to training neural networks with millions of parameters navigating complex loss landscapes!

