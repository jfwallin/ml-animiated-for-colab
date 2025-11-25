# Lab 3: Activation Functions & Nonlinearity
## Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Student Name:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Module 0: When Straight Lines Fail

### Q1. Which dataset(s) can be separated by a straight line?

**Answer:**

<br><br><br>

### Q2. For the XOR pattern (Dataset 2), what happens no matter how you adjust the line? Why can't ANY straight line separate it?

**Answer:**

<br><br><br><br>

### Q3. Why can't a straight line separate the circular ring pattern (Dataset 3)? What kind of boundary shape would you need instead?

**Answer:**

<br><br><br><br>

---

## Module 1: Activation Functions – Bending Space

### Q4. In your own words, what happens to very large positive and very large negative inputs for sigmoid and tanh?

**Answer:**

<br><br><br>

### Q5. Which activation function changes most rapidly near x = 0? How can you tell from the graph?

**Answer:**

<br><br><br>

### Q6. For the sigmoid activation, what happens to points that were far away from the origin (large |x₁| or |x₂|)? Where do they end up in the activated space?

**Answer:**

<br><br><br>

### Q7. For ReLU, what happens to points where x₁ or x₂ is negative? What shape do you see in the activated space?

**Answer:**

<br><br><br>

### Q8. Which activation warps the grid the most (makes it look least like a square), and how can you tell?

**Answer:**

<br><br><br>

### Q9. When you look at the original space (left plot in Section 5), does the boundary between the two colors look straight or curved?

**Answer:**

<br><br><br>

### Q10. In the activated space (right plot in Section 5), what does the boundary look like?

**Answer:**

<br><br><br>

### Q11. Explain, in one or two sentences, how activation functions help us build more flexible decision rules even if the rule itself is linear after activation.

**Answer:**

<br><br><br>

---

## Module 2: Activation Functions in Detail

### Q12. Which activation function has outputs that are always between 0 and 1? Why might this be useful for a model that outputs probabilities?

**Answer:**

<br><br><br>

### Q13. Which activation function is centered at zero (has negative outputs for negative inputs, positive outputs for positive inputs)?

**Answer:**

<br><br><br>

### Q14. What does "saturation" mean? Which activation functions saturate at extreme input values?

**Answer:**

<br><br><br>

### Q15. ReLU is the most popular activation for hidden layers in modern neural networks. Looking at Section 4's interactive tool, what advantage does ReLU have for very large positive inputs compared to Sigmoid or Tanh?

**Answer:**

<br><br><br>

### Q16. Why is the Step function bad for training neural networks with gradient descent? (Hint: Think about smoothness.)

**Answer:**

<br><br><br>

---

## Module 3: Building a Perceptron

### Q17. Write down the two steps of a perceptron in your own words.

**Answer:**

<br><br><br>

### Q18. What do the weights (w₁, w₂) control about the decision boundary?

**Answer:**

<br><br><br>

### Q19. What does the bias (b) control about the decision boundary?

**Answer:**

<br><br><br>

### Q20. Compare Dataset 2 (vertical separation) and Dataset 3 (horizontal separation). For Dataset 2, which weight (w₁ or w₂) needed to be larger to get good classification? For Dataset 3, which weight needed to be larger? Explain why this makes sense.

**Answer:**

<br><br><br><br>

### Q21. For these linearly separable datasets, did the choice of activation function (Sigmoid vs Tanh vs ReLU vs Step) significantly change whether you could classify the data correctly? Why or why not?

**Answer:**

<br><br><br>

### Q22. In Section 6, we saw that neural networks are made of many perceptrons in layers. Why might having multiple layers of perceptrons allow neural networks to solve problems that a single perceptron cannot (like XOR or circles from Module 0)?

**Answer:**

<br><br><br><br>

---

## Module 4: Testing the Perceptron's Limits

### Q23. What was the best accuracy you could achieve on the XOR pattern? Was it close to 100%?

**Answer:**

<br><br><br>

### Q24. What was the best accuracy you could achieve on the concentric circles? Was it close to 100%?

**Answer:**

<br><br><br>

### Q25. For the XOR pattern, no matter how you adjust w₁, w₂, and b, why can't a single straight line (the perceptron's decision boundary) separate all four clusters correctly?

**Answer:**

<br><br><br><br>

### Q26. For the concentric circles, explain why a straight line will always have both blue and red points on both sides of it.

**Answer:**

<br><br><br><br>

### Q27. In Section 5, we saw that multi-layer networks can solve XOR and circles by combining multiple decision boundaries. How many perceptrons in the hidden layer do you think would be needed to solve XOR? Why?

**Answer:**

<br><br><br><br>

---

## Reflection (Optional)

What was the most surprising or interesting thing you learned in this lab?

**Answer:**

<br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules using the notebooks
- [ ] Answered all 27 questions (Q1-Q27)
- [ ] Tested the perceptron on all datasets in Module 3
- [ ] Attempted to classify XOR and circles in Module 4
- [ ] Written thoughtful, complete answers
- [ ] Discussed your answers with your group members (if working in a group)

---

**Submission Instructions:**

Submit this completed answer sheet according to your instructor's guidelines (PDF upload, hardcopy, etc.).
