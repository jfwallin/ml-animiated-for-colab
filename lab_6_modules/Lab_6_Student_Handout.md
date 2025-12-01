# Lab 6: Model Interpretability & Saliency Maps

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In today's lab, you will explore **explainability in AI** through saliency maps—visualizations that show which parts of an input drive a model's predictions. You'll learn to:

- Understand what saliency maps are and why explainability matters
- Compute **word importance** in text classification using masking
- Visualize **pixel importance** in image classification using gradients
- Measure **feature importance** in tabular data using perturbation
- Connect explainability to **AI ethics** and responsible deployment
- Recognize when models focus on the wrong features (spurious correlations)

### Lab Structure

This lab consists of **5 modules** that you'll complete in order:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | What Is Saliency? (Conceptual Introduction) | ~5 min | Conceptual |
| 1 | Text Saliency with Word Masking | ~15 min | Colab |
| 2 | Image Saliency with MobileNetV2 | ~20 min | Colab |
| 3 | Tabular Saliency via Feature Perturbation | ~10 min | Colab |
| 4 | Ethics & Explainability in Practice | ~10 min | Conceptual |

**Total Time:** ~60 minutes

### Working in Groups

- Work in **small groups** (2-4 people)
- One group member runs Colab and shares their screen
- Everyone participates in discussion and predictions
- All group members can use their own notebooks

### Key Concepts

> **SALIENCY MAP**: A visualization showing which parts of an input were most important to a model's prediction. Like highlighting the evidence that drove a decision.
>
> **EXPLAINABILITY**: The degree to which humans can understand how an AI system makes decisions. Critical for trust, debugging, and ethical deployment.
>
> **PERTURBATION**: Changing parts of the input (removing words, masking pixels, altering features) to measure their importance based on prediction changes.
>
> **FEATURE IMPORTANCE**: The impact each feature has on a model's predictions. High importance means changing that feature significantly affects the output.
>
> **SPURIOUS CORRELATION**: When a model learns to rely on features that happen to correlate with the outcome but don't actually cause it (e.g., hospital logos instead of anatomy).

### Connection to Previous Labs

**Lab 1:** You learned about models, parameters, and predictions
- **Lab 6:** You'll learn **which inputs affect those predictions most**

**Lab 2:** You learned about gradient descent
- **Lab 6:** You'll use **gradients to compute saliency**

**Lab 3:** You learned about activation functions
- **Lab 6:** You'll see **which transformed features matter**

**Lab 4:** You learned that hidden layers create representations
- **Lab 6:** You'll understand **what those representations capture**

**Lab 5:** You learned that embeddings encode meaning
- **Lab 6:** You'll see **which parts of meaning drive predictions**

---

## Module 0: What Is Saliency?

**Time:** ~5 minutes
**Type:** Conceptual (markdown only)

### Learning Objectives
- Understand what saliency means in plain language
- Recognize why explainability matters for AI ethics
- Connect to real-world scenarios (medical diagnosis, loan decisions, etc.)
- Understand limitations of saliency (not causality, not complete)

### What You'll Learn

This module sets the conceptual foundation:
- What is a saliency map and why do we need it?
- How do saliency methods work (intuition, no heavy math)?
- When is explainability critical vs. nice-to-have?
- What are the limitations of saliency maps?

### Key Insight

> **When a doctor makes a diagnosis, they can explain their reasoning. When a bank denies a loan, they must provide justification. But when an AI model makes a decision, how do we know what it's "thinking"?**
>
> Saliency maps let us **see inside the black box** and understand what features the model relies on.

### Questions

**Q1.** In your own words, what is a saliency map and why would we want one?

**Q2.** Give an example of a real-world application where model explainability is ethically important.

**Q3.** Why might a saliency map not tell the whole story about how a model makes decisions?

---

## Module 1: Text Saliency with Word Masking

**Time:** ~15 minutes
**Type:** Colab notebook

### Learning Objectives
- Train a simple sentiment classifier on a tiny corpus
- Compute word importance by measuring prediction change when words are removed
- Visualize word saliency as bar charts
- Experiment with custom sentences interactively
- Understand which words drive positive vs. negative predictions

### What You'll Do

1. Train a **logistic regression sentiment classifier** on 12 movie reviews (6 positive, 6 negative)
2. Learn the **word masking** technique: remove words one at a time and measure prediction change
3. Compute and visualize which words are most important (positive/negative contributions)
4. Test on examples: clearly positive reviews, negative reviews, and tricky negations
5. **Try your own sentences** in the interactive section

### The Method: Word Masking

```python
# Simple idea:
baseline = model.predict("The movie was excellent")  → 95% positive

# Remove each word:
remove "The"      → 94% positive (change: -1%)
remove "movie"    → 92% positive (change: -3%)
remove "was"      → 93% positive (change: -2%)
remove "excellent" → 50% positive (change: -45%)

# Importance ranking: "excellent" >> "movie" > "was" > "The"
```

**Key insight:** Large prediction change when removed = high importance

### Key Patterns You'll Discover

- **Content words** (excellent, terrible, disappointing) have **high importance**
- **Function words** (the, and, was) have **low importance**
- **Negations** (not, never) are **crucial** for flipping sentiment
- **Context matters**: "good" is positive, but "not good" is negative

### Questions

**Q4.** In the example sentence, which word had the highest importance score? Does this match your intuition?

**Q5.** Try your own sentence in the interactive section. Which words were most important? Record your sentence and the top 3 words.

**Q6.** What happens when you mask a negation word like "not"? (e.g., "The movie was not good")

**Q7.** Why might function words like "the" and "and" have low importance scores?

**Q8.** How could word saliency be useful for debugging a text classifier in a real-world application?

---

## Module 2: Image Saliency with MobileNetV2

**Time:** ~20 minutes
**Type:** Colab notebook

### Learning Objectives
- Load a pre-trained image classification model (MobileNetV2)
- Compute **gradient-based saliency** using TensorFlow GradientTape
- Visualize heatmaps overlaid on original images
- Understand what pixels drive object recognition
- Upload your own images and see what the model focuses on

### What You'll Do

1. Load **MobileNetV2** (14MB, recognizes 1000 object classes)
2. Classify example images (dog, handwritten digit, objects)
3. Compute **gradient × input saliency** (which pixels affect the prediction most?)
4. Visualize as **heatmaps**: red/bright = important pixels, blue/dark = unimportant
5. **Upload your own images** and see what the model "looks at"

### The Method: Gradient-Based Saliency

Remember from Lab 2 that gradients measure sensitivity:
- For training: gradients show how to change **weights**
- For saliency: gradients show which **pixels** affect output most

```python
# High gradient → small pixel change → big prediction change → important pixel
# Low gradient → small pixel change → no prediction change → unimportant pixel
```

### Three Visualizations

1. **Original image**
2. **Saliency heatmap** (standalone)
3. **Overlay** (heatmap on top of original)

### Expected Patterns

**For a dog image:**
- **High saliency:** Dog's face (ears, snout, eyes), fur texture
- **Low saliency:** Background (grass, sky)

**For a handwritten digit:**
- **High saliency:** Loops, curves, edges that define the shape
- **Low saliency:** White space, background

### Why This Matters

**Medical imaging example:**
- **Good:** AI focuses on lung tissue in X-rays
- **Bad:** AI focuses on hospital logo in corner (spurious correlation!)
- **Saliency reveals** which case you're in

### Questions

**Q9.** For the dog image, which parts of the image had the highest saliency? Does this make sense for recognizing a dog?

**Q10.** For the handwritten digit, which parts had high saliency? Why would those regions be important?

**Q11.** Upload your own image. What object did the model classify it as? What was the top prediction?

**Q12.** For your uploaded image, which parts had the highest saliency? Does this reveal how the model recognized the object?

**Q13.** Looking at the saliency map, can you think of a scenario where a model might focus on the "wrong" features? (Hint: spurious correlations)

**Q14.** How is image saliency different from the word saliency you explored in Module 1? What's similar?

---

## Module 3: Tabular Saliency via Feature Perturbation

**Time:** ~10 minutes
**Type:** Colab notebook

### Learning Objectives
- Understand saliency for structured/tabular data
- Compute feature importance via perturbation (setting features to mean value)
- Connect to fairness and bias discussions
- Recognize when features might be problematic proxies

### What You'll Do

1. Create a synthetic student exam dataset with 4 features:
   - `hours_studied` (0-10)
   - `attendance_rate` (0-1)
   - `homework_completion` (0-1)
   - `previous_gpa` (2.0-4.0)
   - **Target:** `passed_exam` (yes/no)
2. Train a **decision tree** classifier (instant training)
3. Compute **feature importance** via perturbation (set each feature to mean, measure change)
4. Visualize as bar charts showing which features drive pass/fail predictions
5. Explore **ethical issues**: What if "zip_code" was included?

### The Method: Feature Perturbation

```python
# Baseline:
student = {hours: 8, attendance: 0.9, homework: 0.95, GPA: 3.5}
prediction = 90% pass

# Perturb each feature to mean:
Set hours to 5 (mean)       → 60% pass (importance: 30%)
Set attendance to 0.7 (mean) → 85% pass (importance: 5%)
Set GPA to 3.0 (mean)       → 80% pass (importance: 10%)

# Ranking: hours_studied > GPA > attendance
```

### Ethical Issue: Problematic Features

**What if the model used:**
- **zip_code** → proxy for socioeconomic status, redlining
- **name** → proxy for race, ethnicity, gender
- **age** → ageism, discrimination

**Saliency reveals** when models rely on features they shouldn't!

### Key Insight

> Saliency isn't just for debugging—it's crucial for **detecting and preventing discrimination** in automated decision systems.

### Questions

**Q15.** Which feature had the highest importance for predicting exam success? Does this align with intuition?

**Q16.** Try perturbing different features by ±1 standard deviation. Which perturbation changed the prediction the most?

**Q17.** If "zip_code" had high importance, why might this be problematic for a real education system?

**Q18.** Name a feature that might be predictive but ethically problematic to use in a real-world model (hiring, lending, admissions).

---

## Module 4: Ethics & Explainability in Practice

**Time:** ~10 minutes
**Type:** Conceptual (markdown only)

### Learning Objectives
- Understand when explainability is critical vs. nice-to-have
- Recognize the interpretability-performance tradeoff
- Learn real-world examples of saliency revealing problems
- Understand limitations of saliency methods
- Explore best practices for responsible AI deployment

### What You'll Explore

1. **When explainability matters:** High-stakes decisions (medical, legal, financial) vs. low-stakes (recommendations)
2. **The tradeoff:** Simple explainable models (75% accuracy) vs. complex black-box models (85% accuracy)
3. **Real-world failures:** Medical AI focusing on watermarks, loan models using zip code proxies
4. **Regulations:** GDPR (right to explanation), FCRA (adverse action notices), FDA AI guidance
5. **Limitations:** Saliency ≠ causality, can be misleading, doesn't capture feature interactions
6. **Best practices:** Audit early and often, match explainability to stakes, enable human oversight

### Key Questions to Ponder

- Would you choose a 75% accurate explainable model or an 85% accurate black box for medical diagnosis?
- Why do regulations require "right to explanation" for automated decisions?
- How would you audit a hiring AI for bias using saliency?
- What are the dangers of relying too heavily on saliency explanations?

### The Big Picture

**Lab sequence summary:**
- **Labs 1-2:** How models learn (parameters, gradients)
- **Labs 3-4:** How models represent data (activations, hidden layers)
- **Lab 5:** How models encode meaning (embeddings)
- **Lab 6:** How to understand what models do and whether to trust them

### Questions

**Q19.** Think of a specific high-stakes AI application (medical, legal, financial, etc.). Why would explainability be critical in that context? What could go wrong without it?

**Q20.** In a medical diagnosis system, would you choose a 75% accurate explainable model or an 85% accurate black-box model? Explain your reasoning.

**Q21.** Imagine you're deploying a resume screening AI for hiring. How would you use saliency to audit for bias before deployment?

**Q22.** Why do you think regulations like GDPR require "right to explanation" for automated decisions? What problem is this trying to solve?

**Q23.** Given the limitations of saliency, what else would you want to know about a model before deploying it in a high-stakes application?

**Q24.** Reflecting on the entire lab (Modules 0-4): What is the most important takeaway about explainability and responsible AI? Why does this matter for your future work with AI systems?

---

## Key Takeaways

By the end of this lab, you should understand:

- **Saliency transforms black boxes into interpretable decisions** by showing which input parts matter most
- **Different modalities require different methods:** masking for text, gradients for images, perturbation for tabular data
- **Explainability is critical for high-stakes AI:** medical diagnosis, criminal justice, lending, hiring, education
- **Saliency reveals failure modes:** spurious correlations, problematic proxies, background focus, negation failures
- **The interpretability-performance tradeoff** often forces difficult choices between accuracy and explainability
- **Regulations increasingly mandate explainability:** GDPR, FCRA, FDA guidance, emerging AI acts
- **Saliency has important limitations:** not causality, can be misleading, doesn't capture interactions
- **Responsible deployment requires multiple safeguards:** saliency + fairness metrics + human oversight + documentation

---

## Connection to Real-World AI

### Applications Where Saliency Matters

| Application | How Saliency Helps |
|-------------|-------------------|
| **Medical Diagnosis** | Verify AI focuses on right anatomical regions, not artifacts |
| **Loan Decisions** | Detect reliance on discriminatory proxies (zip code, demographics) |
| **Content Moderation** | Understand which words/images trigger flags, reduce false positives |
| **Hiring Systems** | Audit for bias against protected groups (name, university, age) |
| **Autonomous Vehicles** | Verify attention to relevant objects (pedestrians, signs, not billboards) |
| **Criminal Justice** | Ensure risk assessment doesn't encode systemic bias |
| **Educational Assessment** | Verify grading AI focuses on relevant features, not spurious cues |

### Real-World Failures Saliency Could Have Caught

- **Husky vs. Wolf classifier:** Focused on background snow (wolves) vs. grass (huskies), not animals
- **Pneumonia detector:** Focused on hospital logos and watermarks, not lung pathology
- **Hiring AI:** Amazon's system penalized resumes with "women's" (e.g., "women's chess club")
- **Loan approval:** Models using zip code as proxy for race, perpetuating redlining
- **Face recognition:** Focused on hats, glasses, backgrounds rather than facial structure

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (0, 1, 2, 3, 4)
- [ ] Run all code cells and viewed visualizations
- [ ] Answered all 24 questions in the answer sheet
- [ ] Tried your own text sentence in Module 1
- [ ] Uploaded your own image in Module 2
- [ ] Experimented with feature perturbations in Module 3
- [ ] Reflected on ethical implications in Module 4
- [ ] Understood connections to Labs 1-5
- [ ] Recognized when saliency reveals problems
- [ ] Understood limitations of saliency methods

---

## Submission Instructions

Submit your completed answer sheet according to your instructor's guidelines (Canvas upload, PDF, etc.).

**Congratulations!** You've completed Lab 6 and learned how to build **transparent, accountable, and trustworthy AI systems**!

---

## Additional Resources

### Tools for Explainability

- **SHAP (SHapley Additive exPlanations):** Game theory approach to feature importance
- **LIME (Local Interpretable Model-Agnostic Explanations):** Local approximations
- **Integrated Gradients:** More stable gradient-based saliency
- **Captum (PyTorch):** Model interpretability library
- **What-If Tool (Google):** Interactive model probing

### Fairness and Bias Detection

- **Fairlearn (Microsoft):** Bias detection and mitigation toolkit
- **AI Fairness 360 (IBM):** Comprehensive fairness library
- **Aequitas:** Bias and fairness audit toolkit

### Responsible AI Frameworks

- **Google Model Cards:** Documentation standard for ML models
- **Microsoft Responsible AI Standard:** Company-wide principles
- **Partnership on AI:** Industry consortium for responsible AI

### Reading and Learning

- **"Interpretable Machine Learning" by Christoph Molnar:** Free online book covering all major explainability methods
- **"Fairness and Machine Learning" by Barocas, Hardt, Narayanan:** Comprehensive textbook on bias and fairness
- **ACM FAccT Conference:** Top academic venue for fairness, accountability, and transparency research

### Regulations and Policy

- **EU AI Act:** Risk-based regulation framework for AI systems
- **NIST AI Risk Management Framework:** US government guidance for responsible AI
- **GDPR Article 22:** Right to explanation for automated decisions
- **Equal Credit Opportunity Act:** Requires explanation for adverse credit actions

---

**Questions or feedback?** Reach out to your instructor or TA!
