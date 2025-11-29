# Lab 4: Real-World Machine Learning with TensorFlow/Keras
## Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Student Name:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Module 0: Lifting Dimensions

### Q1. In 2D, can you draw a straight line that separates the XOR pattern? Why or why not?

**Answer:**

<br><br><br>

### Q2. After adding the third dimension (x₃ = x₁ × x₂), describe what you observed in the 3D plot. Could you see how a flat plane separates the two classes?

**Answer:**

<br><br><br>

### Q3. When you view the 3D separating plane from directly above (bird's eye view), what shape does the decision boundary have in 2D? Is it a straight line?

**Answer:**

<br><br><br>

### Q4. How is adding a third dimension similar to what activation functions did in Lab 3? (Hint: Think about "transforming" or "warping" space)

**Answer:**

<br><br><br>

---

## Module 1: Anatomy of a Tiny Neural Network

### Q5. How many total parameters (weights + biases) does a 2-2-1 network have?

**Answer:**

<br><br><br>

### Q6. What do the two hidden neurons (h₁ and h₂) represent? How do they relate to the dimension-lifting you did in Module 0?

**Answer:**

<br><br><br>

### Q7. When you adjust the weights connecting inputs to hidden layer, what changes about the decision boundary?

**Answer:**

<br><br><br>

### Q8. Why do we need TWO hidden neurons instead of just one? What would happen with only one hidden neuron?

**Answer:**

<br><br><br>

---

## Module 2: Training a Neural Network

### Q9. What is the "loss function" and why do we want to minimize it?

**Answer:**

<br><br><br>

### Q10. Describe what happens when the learning rate is too small vs. too large.

**Answer:**

<br><br><br>

### Q11. How does momentum help gradient descent converge faster?

**Answer:**

<br><br><br>

### Q12. After running multi-start with 5 different random initializations, did all runs converge to the same final loss? What does this tell you about the stochastic nature of training?

**Answer:**

<br><br><br>

---

## Module 3: Iris Flower Classification

### Q13. Did the linear model (no hidden layer) achieve high accuracy on Iris? Why or why not?

**Answer:**

<br><br><br>

### Q14. How much did adding a hidden layer improve accuracy? Was the improvement large or small?

**Answer:**

<br><br><br>

### Q15. Experiment with different values of `hidden_units` (2, 4, 8, 16, 32, 64). At what point do you see diminishing returns?

**Answer:**

Record your results here:

| Hidden Units | Test Accuracy |
|--------------|---------------|
| 2            | ________%     |
| 4            | ________%     |
| 8            | ________%     |
| 16           | ________%     |
| 32           | ________%     |
| 64           | ________%     |

Diminishing returns appear around: \_\_\_\_\_ units

<br>

### Q16. Compare this to Module 2 XOR training. What's similar? (Hint: gradient descent, weight updates) What's different? (Hint: manual vs. automatic, dataset)

**Answer:**

**Similar:**

<br><br>

**Different:**

<br><br><br>

### Q17. Why is 95% accuracy considered "excellent" while 100% might be suspicious?

**Answer:**

<br><br><br>

### Q18. Looking at the petal scatter plot, why do you think Versicolor and Virginica are harder to separate than Setosa?

**Answer:**

<br><br><br>

### Q19. (Multiple Runs Section) After running the linear model 5 times, what was the mean accuracy ± standard deviation? Was the model consistent?

**Answer:**

Mean accuracy: \_\_\_\_\_\_% ± \_\_\_\_\_\_%

Consistent? (Circle one):  **Yes** / **No**

Explanation:

<br><br>

### Q20. Did the hidden layer model show more or less variability than the linear model? Why might this be?

**Answer:**

More / Less variability (circle one)

Explanation:

<br><br><br>

### Q21. Looking at the box plots, do the two models' accuracy distributions overlap significantly? What does this tell you about whether hidden layers truly help?

**Answer:**

<br><br><br><br>

---

## Module 4: Breast Cancer Classification

### Q22. How did the baseline linear model perform on breast cancer data? Were you surprised? Why or why not?

**Answer:**

Baseline test accuracy: \_\_\_\_\_\_%

<br><br><br>

### Q23. Did adding hidden layers significantly improve accuracy? At what architecture did you see diminishing returns?

**Answer:**

Record your architecture experiments:

| Hidden Layers | Units per Layer | Test Accuracy |
|--------------|-----------------|---------------|
| 0            | N/A             | ________%     |
| 1            | 8               | ________%     |
| 1            | 16              | ________%     |
| 1            | 32              | ________%     |
| 2            | 16              | ________%     |
| 2            | 32              | ________%     |

Best architecture: \_\_\_\_ layers, \_\_\_\_ units per layer

<br>

### Q24. Looking at your confusion matrices, did you reduce false negatives (missed cancers) with more complex models? Is there a trade-off with false positives?

**Answer:**

Baseline false negatives: \_\_\_\_
Best model false negatives: \_\_\_\_

Trade-off observed?

<br><br><br>

### Q25. In medical diagnosis, which error is more concerning: false positive (predicting cancer when there isn't any) or false negative (missing actual cancer)? Explain your reasoning.

**Answer:**

More concerning: (Circle one)  **False Positive** / **False Negative**

Reasoning:

<br><br><br><br>

### Q26. Compare Module 3 (Iris) and Module 4 (Breast Cancer). Which dataset benefited more from hidden layers? Why might this be? (Think about feature count and class separability)

**Answer:**

Dataset that benefited more: (Circle one)  **Iris** / **Breast Cancer**

Explanation:

<br><br><br><br>

### Q27. Reflect on your journey from Module 0 to now:

**a) What's the connection between manually lifting XOR to 3D (Module 0) and hidden layers in Keras?**

**Answer:**

<br><br><br>

**b) How does `.fit()` relate to the gradient descent you saw in Module 2?**

**Answer:**

<br><br><br>

**c) What's the SAME between 2-feature XOR and 30-feature cancer diagnosis?**

**Answer:**

<br><br><br>

### Q28. Given that a simple linear model achieves ~95% accuracy, why might doctors still want a more complex model? Why might they prefer the simpler one?

**Answer:**

**Reasons for complex model:**

<br><br>

**Reasons for simpler model:**

<br><br><br>

### Q29. (Multiple Runs Section) After running your baseline model 5 times, what was the mean number of false negatives ± std? If this varies from 2-4 missed cancers depending on random initialization, is that acceptable for medical deployment?

**Answer:**

Mean false negatives: \_\_\_\_\_ ± \_\_\_\_\_

Acceptable for medical deployment? (Circle one):  **Yes** / **No**

Reasoning:

<br><br><br>

### Q30. Did your custom model reduce false negatives CONSISTENTLY across all 5 runs, or was the improvement inconsistent? What does the box plot tell you?

**Answer:**

Reduction was: (Circle one)  **Consistent** / **Inconsistent**

What box plot shows:

<br><br><br>

### Q31. If the accuracy box plots for baseline and custom models overlap significantly, what does this mean about the reliability of any "improvement" you measured?

**Answer:**

<br><br><br><br>

---

## Reflection (Optional)

### What was the most surprising or interesting thing you learned in this lab?

**Answer:**

<br><br><br><br>

### How has your understanding of machine learning changed from Lab 1 (manual weight adjustment) to Lab 4 (TensorFlow/Keras on real data)?

**Answer:**

<br><br><br><br>

### If you were deploying a medical ML system for cancer diagnosis in a real hospital, what additional steps would you take beyond what we did in this lab?

**Answer:**

<br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed Module 0 (Lifting Dimensions)
- [ ] Completed Module 1 (Anatomy of a Tiny Neural Network)
- [ ] Completed Module 2 (Training a Neural Network)
- [ ] Completed Module 3 (Iris Classification)
- [ ] Completed Module 4 (Breast Cancer Classification)
- [ ] Run the multiple experiments sections in Modules 3 and 4
- [ ] Filled in all architecture experiment tables (Q15, Q23)
- [ ] Answered all 31 questions (Q1-Q31)
- [ ] Recorded mean ± std from multiple runs (Q19, Q29)
- [ ] Written thoughtful, complete answers
- [ ] Reflected on medical ethics implications (Q25, Q29)
- [ ] Connected to earlier labs (Q4, Q16, Q27)

---

## Submission Instructions

Submit this completed answer sheet according to your instructor's guidelines (PDF upload, hardcopy, etc.).

**Congratulations on completing Lab 4!** You've successfully applied neural networks to real-world data using professional tools.
