# Lab 6: Model Interpretability & Saliency Maps
## Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Student Name:** _________________________________

**Date:** _________________________________

---

## Module 0: What Is Saliency?

### Q1. In your own words, what is a saliency map and why would we want one?

**Answer:**

<br><br><br>

### Q2. Give an example of a real-world application where model explainability is ethically important.

**Answer:**

<br><br><br>

### Q3. Why might a saliency map not tell the whole story about how a model makes decisions?

**Answer:**

<br><br><br>

---

## Module 1: Text Saliency with Word Masking

### Q4. In the example sentence, which word had the highest importance score? Does this match your intuition?

**Answer:**

<br><br><br>

### Q5. Try your own sentence in the interactive section. Which words were most important? Record your sentence and the top 3 words.

**My sentence:** _______________________________________________

**Top 3 important words:**
1. _______________ (importance: ______)
2. _______________ (importance: ______)
3. _______________ (importance: ______)

**Did the results match your expectations?**

<br><br>

### Q6. What happens when you mask a negation word like "not"? (e.g., "The movie was not good")

**Answer:**

<br><br><br>

### Q7. Why might function words like "the" and "and" have low importance scores?

**Answer:**

<br><br><br>

### Q8. How could word saliency be useful for debugging a text classifier in a real-world application?

**Answer:**

<br><br><br><br>

---

## Module 2: Image Saliency with MobileNetV2

### Q9. For the dog image, which parts of the image had the highest saliency? Does this make sense for recognizing a dog?

**Answer:**

<br><br><br>

### Q10. For the handwritten digit, which parts had high saliency? Why would those regions be important?

**Answer:**

<br><br><br>

### Q11. Upload your own image. What object did the model classify it as? What was the top prediction?

**My uploaded image description:** _____________________

**Top prediction:** _______________

**Confidence:** ______%

<br>

### Q12. For your uploaded image, which parts had the highest saliency? Does this reveal how the model recognized the object?

**Answer:**

<br><br><br><br>

### Q13. Looking at the saliency map, can you think of a scenario where a model might focus on the "wrong" features? (Hint: spurious correlations)

**Answer:**

<br><br><br><br>

### Q14. How is image saliency different from the word saliency you explored in Module 1? What's similar?

**Answer:**

**Different:**

<br><br>

**Similar:**

<br><br><br>

---

## Module 3: Tabular Saliency via Feature Perturbation

### Q15. Which feature had the highest importance for predicting exam success? Does this align with intuition?

**Answer:**

<br><br><br>

### Q16. Try perturbing different features by ±1 standard deviation. Which perturbation changed the prediction the most?

**Answer:**

<br><br><br>

### Q17. If "zip_code" had high importance, why might this be problematic for a real education system?

**Answer:**

<br><br><br><br>

### Q18. Name a feature that might be predictive but ethically problematic to use in a real-world model (hiring, lending, admissions).

**Answer:**

<br><br><br>

---

## Module 4: Ethics & Explainability in Practice

### Q19. Think of a specific high-stakes AI application (medical, legal, financial, etc.). Why would explainability be critical in that context? What could go wrong without it?

**Application:** _____________________

**Why explainability is critical:**

<br><br><br>

**What could go wrong without it:**

<br><br><br>

### Q20. In a medical diagnosis system, would you choose a 75% accurate explainable model or an 85% accurate black-box model? Explain your reasoning.

**My choice:** (Circle one)  **75% Explainable** / **85% Black-box**

**Reasoning:**

<br><br><br><br>

### Q21. Imagine you're deploying a resume screening AI for hiring. How would you use saliency to audit for bias before deployment?

**Answer:**

<br><br><br><br><br>

### Q22. Why do you think regulations like GDPR require "right to explanation" for automated decisions? What problem is this trying to solve?

**Answer:**

<br><br><br><br>

### Q23. Given the limitations of saliency, what else would you want to know about a model before deploying it in a high-stakes application?

**Answer:**

<br><br><br><br><br>

### Q24. Reflecting on the entire lab (Modules 0-4): What is the most important takeaway about explainability and responsible AI? Why does this matter for your future work with AI systems?

**Most important takeaway:**

<br><br><br><br>

**Why it matters for future work:**

<br><br><br><br>

---

## Reflection (Optional)

### What was the most surprising thing you learned about saliency and explainability?

**Answer:**

<br><br><br><br>

### How might saliency methods be used to detect and prevent discrimination in automated decision systems?

**Answer:**

<br><br><br><br>

### What connections do you see between explainability concepts from Lab 6 and the model concepts from Labs 1-5?

**Answer:**

<br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (0, 1, 2, 3, 4)
- [ ] Answered all 24 questions (Q1-Q24)
- [ ] Included specific values where requested (word importance, predictions, confidence scores)
- [ ] Recorded your custom sentence experiment (Q5)
- [ ] Recorded your image upload experiment (Q11-Q12)
- [ ] Recorded feature perturbation results (Q16)
- [ ] Run all code cells and viewed visualizations
- [ ] Tried prediction questions BEFORE running code
- [ ] Written thoughtful, complete answers
- [ ] (Optional) Completed reflection questions
- [ ] Reflected on ethical implications

---

## Submission Instructions

Submit this completed answer sheet according to your instructor's guidelines (Canvas upload, hardcopy, PDF, etc.).

**Congratulations on completing Lab 6!** You now understand how to build **transparent, accountable, and trustworthy AI systems**—a critical skill for responsible AI deployment.
