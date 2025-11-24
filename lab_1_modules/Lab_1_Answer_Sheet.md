# Lab 1: Models, Errors, Loss, and Optimization
## Answer Sheet

**Course:** DATA 1010 – AI in Action
**Name(s):** _________________________________ **Group Code:** _______

---

## Overview

**What you'll learn:** How machines measure error, what "loss" means, and how optimization finds better models.

**Key Concept:** A **model** makes predictions, has adjustable **parameters**, and improves by reducing total **error** (loss).

**Lab Structure:** 6 modules (0-5) using the same group code throughout.

---

## Module 1: Understanding Global Error (~5 min)

**Concepts:**
- **Local error** = error at one data point: `actual - predicted`
- **Global error (SSE)** = sum of all squared errors
- Why square? Prevents cancellation, penalizes big mistakes

**Q1.** What does "global error" or "loss" measure?

<br><br>

**Q2.** If we change the slope or intercept, how does loss change?

<br><br>

---

## Module 2: Interactive Line Fitting (~15 min)

**What you do:** Use sliders to adjust slope (m) and intercept (b) to minimize error.

**Q3.** How do residual lines help you understand local error at each point?

<br><br><br>

**Q4.** Can you make global error small even if some points have large errors? When?

<br><br><br>

**Q5 - PREDICTION:** Add one extreme outlier far from others. How will this affect the line and error?

**Prediction:**

<br><br>

**Result:**

<br><br>

---

## Module 3: Parameter Space Optimization (~20 min)

**What you do:** Find best (m, b) seeing only MSE values—no data or line visible!

**Concepts:**
- **Parameter space** = 2D plane where each point (m, b) represents a different line
- **MSE** = SSE / number of points
- You only see MSE numbers, like real ML algorithms

**Q6.** Describe your strategy for choosing (m, b) values. Was it systematic?

<br><br><br>

**Q7.** After revealing the MSE landscape: How close was your best guess to (a) the grid minimum and (b) the least-squares solution?

<br><br><br>

**Q8.** How is seeing only MSE similar to how ML algorithms train (they only see loss, not "right answers")?

<br><br><br>

---

## Module 4: Hidden Function Optimization (~15 min)

**What you do:** Find minimum of 1D function by choosing x values (-10 to 10). Only see sampled points, not the curve!

**Q9.** What strategies did you use to choose x values?

<br><br><br>

**Q10.** How did "warmer/colder" feedback influence your choices?

<br><br><br>

**Q11.** If even the scatter plot was hidden (only table of x, f(x)), could you still find the minimum? How?

<br><br><br>

---

## Module 5: Mountain Landscape Search (~20 min)

**What you do:** Find the highest peak in a 2D landscape with multiple peaks. Sample (x, y) points to measure altitude.

**Concepts:**
- **Local maximum** = highest point nearby (but maybe not overall)
- **Global maximum** = highest point anywhere
- Risk: Finding a local peak and stopping too early

**Q12.** Describe your strategy for choosing (x, y) locations after finding a high point.

<br><br><br>

**Q13 - PREDICTION:** Before revealing: How many peaks? Where is the global peak?

**Prediction:**

<br><br>

**Result after revealing:** How many peaks? Where did you explore most?

<br><br>

**Q14.** Compare your best sample to the true global peak. Were you close or stuck at a local maximum?

<br><br><br>

**Q15.** How is this mountain search similar to ML optimization with many local minima? What risks exist?

<br><br><br>

---

## Key Takeaways

✓ **Error measurement:** Models optimized by minimizing total squared error (SSE/MSE)
✓ **Blind optimization:** Algorithms only see loss values, not the "answer"
✓ **Multiple optima:** Complex problems have many local solutions—hard to find global best
✓ **Strategy matters:** Systematic search beats random guessing

**Connection to ML:** Everything you did mirrors how neural networks train—adjusting millions of parameters to minimize loss on a complex landscape!
