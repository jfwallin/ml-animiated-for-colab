# Lab 4: Real-World Machine Learning with TensorFlow/Keras
## Student Handout

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In this lab, you'll build your understanding from basic concepts to real-world machine learning. You'll manually build neural networks, understand how they train, and then apply industry-standard tools to actual medical and botanical datasets:

1. **Dimension lifting** – Why adding dimensions makes hard problems solvable
2. **Network anatomy** – Understanding parameters and architecture
3. **Gradient descent** – How networks learn automatically
4. **Real-world ML** – Applying neural networks to actual datasets
5. **Stochastic ML** – Understanding variability and statistical reporting
6. **Medical ethics** – High-stakes predictions in healthcare

### Lab Structure

This lab consists of **5 modules** that you'll complete in order:

| Module | Title | Topic | Time |
|--------|-------|-------|------|
| 0 | Lifting Dimensions | Why XOR needs 3D space | ~12-15 min |
| 1 | Anatomy of a Tiny Neural Network | Understanding a 2-2-1 network | ~20 min |
| 2 | Training a Neural Network | Gradient descent automation | ~15-20 min |
| 3 | Iris Flower Classification | Real botanical data (4 features, 3 species) | ~30 min |
| 4 | Breast Cancer Diagnosis | Real medical data (30 features, binary) | ~30-35 min |

**Total Time:** ~110-120 minutes (approximately 2 hours)

### Working in Groups

- Work in **small groups** (2-4 people) or individually
- One person shares their screen running the notebooks
- Everyone participates in discussion
- All group members can use their own notebooks on Google Colab

### Key Concepts

> **TENSORFLOW/KERAS**: Industry-standard machine learning framework
> - TensorFlow: Low-level numerical computation library
> - Keras: High-level neural network API (built into TensorFlow)
> - Powers ML at Google, Uber, Airbnb, and thousands of companies
>
> **STOCHASTIC ML**: Random initialization causes result variability
> - Same code, different results each run
> - Professional practice: run multiple times, report mean ± std
> - Example: "Accuracy: 94.5% ± 1.2%" not just "Accuracy: 95%"
>
> **CONFUSION MATRIX**: Shows types of errors the model makes
> - Critical for medical applications
> - False Negative: Missing cancer (dangerous!)
> - False Positive: False alarm (stressful, leads to unnecessary procedures)

### Connection to Previous Labs

**Lab 1:** You learned about models, parameters, and optimization basics

**Lab 2:** You learned about gradient descent for automatic parameter updates

**Lab 3:** You learned about activation functions and why hidden layers matter

**Lab 4:** You'll understand WHY neural networks work (dimension lifting), HOW they're structured (anatomy), how they LEARN (gradient descent), and apply them to REAL problems!

---

## Module 0: Lifting Dimensions

**Time:** ~12-15 minutes
**Type:** Conceptual foundation

### What You'll Do

1. Revisit the XOR problem that's impossible in 2D
2. Manually add a third dimension (x₃ = x₁ × x₂)
3. Discover that XOR becomes perfectly separable in 3D
4. See how a flat plane in 3D creates a curved boundary in 2D

### Learning Objectives

- Understand why XOR cannot be solved in 2D
- See that adding dimensions can make impossible problems solvable
- Connect dimension-lifting to what hidden layers do
- Build intuition for why neural networks need multiple layers

### The Core Insight

**XOR in 2D:** Impossible to separate with any straight line

**XOR in 3D (with x₃ = x₁ × x₂):**
- Blue points (opposite corners): Both coordinates same sign → x₃ positive
- Red points (opposite corners): Coordinates opposite signs → x₃ negative
- Simple rule: "If x₃ > 0, predict Blue" → Perfect separation!

This is EXACTLY what hidden layers do automatically!

### Key Discovery

You'll experiment with 5 different features to add as x₃:
1. **x₃ = x₁ × x₂** (Product) ✅ Works perfectly!
2. **x₃ = x₁ + x₂** (Sum) ❌ Doesn't help
3. **x₃ = x₁² + x₂²** (Sum of squares) ❌ All points equidistant
4. **x₃ = |x₁| + |x₂|** (Manhattan) ❌ Doesn't help
5. **x₃ = max(x₁, x₂)** (Maximum) ❌ Doesn't help

Only the **product** captures the interaction that defines XOR!

### Interactive 3D Visualization

- Rotate the 3D plot to see separation
- Switch between different features
- See which plane separates the classes
- Understand why **x₁ × x₂** is special

### Questions

**Q1.** In 2D, can you draw a straight line that separates the XOR pattern? Why or why not?

**Q2.** After adding the third dimension (x₃ = x₁ × x₂), describe what you observed in the 3D plot. Could you see how a flat plane separates the two classes?

**Q3.** When you view the 3D separating plane from directly above (bird's eye view), what shape does the decision boundary have in 2D? Is it a straight line?

**Q4.** How is adding a third dimension similar to what activation functions did in Lab 3? (Hint: Think about "transforming" or "warping" space)

---

## Module 1: Anatomy of a Tiny Neural Network

**Time:** ~20 minutes
**Type:** Understanding network structure

### What You'll Do

1. Examine a 2-2-1 neural network architecture
2. Count all parameters (weights and biases)
3. Understand what each layer does
4. See how 9 parameters solve XOR
5. Manually adjust weights to understand their roles

### Learning Objectives

- Understand the structure of a simple neural network
- Count and identify all network parameters (weights and biases)
- See how hidden layers create new dimensions
- Connect to Module 0's dimension lifting concept
- Prepare for automatic training in Module 2

### Network Architecture: 2-2-1

```
Input Layer (2 neurons): x₁, x₂
   ↓ (4 weights + 2 biases = 6 parameters)
Hidden Layer (2 neurons): h₁, h₂
   ↓ (2 weights + 1 bias = 3 parameters)
Output Layer (1 neuron): y
```

**Total:** 9 parameters to learn

### Key Concepts

**Hidden Neurons Create New Dimensions:**
- h₁ = activation(w₁·x₁ + w₂·x₂ + b₁)
- h₂ = activation(w₃·x₁ + w₄·x₂ + b₂)
- These are like the x₃ you created in Module 0!
- But the network learns the best weights automatically

**Connection to Module 0:**
- You manually created x₃ = x₁ × x₂
- Hidden neurons create similar features, but learned!
- Two hidden neurons = two new dimensions
- More flexibility than just one x₃

### Interactive Exploration

- Adjust individual weights with sliders
- See how each weight affects the decision boundary
- Understand the role of biases
- Discover which weight patterns solve XOR

### Questions

**Q5.** How many total parameters (weights + biases) does a 2-2-1 network have?

**Q6.** What do the two hidden neurons (h₁ and h₂) represent? How do they relate to the dimension-lifting you did in Module 0?

**Q7.** When you adjust the weights connecting inputs to hidden layer, what changes about the decision boundary?

**Q8.** Why do we need TWO hidden neurons instead of just one? What would happen with only one hidden neuron?

---

## Module 2: Training a Neural Network

**Time:** ~15-20 minutes
**Type:** Understanding automatic learning

### What You'll Do

1. See gradient descent find weights automatically
2. Understand the loss function (what we're minimizing)
3. Watch training curves (loss decreasing over epochs)
4. Experiment with learning rates
5. See momentum speed up convergence
6. Try multiple random starting points (multi-start)

### Learning Objectives

- Understand how gradient descent automatically finds good weights
- See the loss function decrease during training
- Learn about learning rate and momentum hyperparameters
- Understand that initialization matters (stochastic learning)
- Connect to the manual weight adjustment from Module 1

### The Big Transition

**Module 1:** You manually adjusted 9 weights → Tedious, slow, hard to optimize

**Module 2:** Gradient descent adjusts 9 weights automatically → Fast, optimal, scalable

### Key Concepts

**Loss Function:**
- Measures how wrong the network's predictions are
- Goal: Minimize loss by adjusting weights
- Gradient tells us which direction to move weights

**Gradient Descent:**
1. Compute how wrong the network is (loss)
2. Calculate gradients (how to adjust each weight)
3. Update weights in the direction that reduces loss
4. Repeat for many epochs

**Momentum:**
- Like a ball rolling downhill, builds up speed
- Helps escape local minima
- Speeds up convergence

**Multi-Start:**
- Try different random starting weights
- Pick the one that achieves lowest loss
- Important because of stochasticity!

### Interactive Experiments

- Adjust learning rate (0.01, 0.1, 1.0, 10.0)
- Toggle momentum on/off
- Watch loss curves
- Compare different random initializations

### What You'll Discover

- Learning rate too small → Slow convergence
- Learning rate too large → Unstable, diverges
- Momentum → Faster, smoother convergence
- Random initialization → Different results each time!

### Questions

**Q9.** What is the "loss function" and why do we want to minimize it?

**Q10.** Describe what happens when the learning rate is too small vs. too large.

**Q11.** How does momentum help gradient descent converge faster?

**Q12.** After running multi-start with 5 different random initializations, did all runs converge to the same final loss? What does this tell you about the stochastic nature of training?

---

## Module 3: Iris Flower Classification

**Time:** ~30 minutes
**Dataset:** Iris flowers (150 samples, 4 features, 3 species)
**Type:** Multiclass classification

### What You'll Do

1. Load a real botanical dataset from scikit-learn
2. Visualize the data to see if classes are separable
3. Train a **baseline linear model** (no hidden layers)
4. Train a **hidden layer model** and compare performance
5. **Run multiple experiments** to understand ML variability
6. Report results with statistical rigor (mean ± std)

### Learning Objectives

- Apply neural networks to real data (not toy XOR patterns)
- Understand that 95% accuracy is excellent (not 100%)
- See how Keras automates the gradient descent from Lab 2
- Learn that ML is stochastic - results vary between runs
- Practice professional ML reporting with statistics

### The Iris Dataset

**History:** Collected by botanist Edgar Anderson in 1936, analyzed by statistician Ronald Fisher. One of the most famous datasets in machine learning!

**Features (4 measurements per flower):**
1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

**Target Classes (3 species):**
- Setosa (class 0) - 50 samples
- Versicolor (class 1) - 50 samples
- Virginica (class 2) - 50 samples

**The Task:** Given a flower's measurements, predict which species it is.

### Connection to Earlier Labs

**Remember XOR from Lab 1?**
- 2 inputs → binary output (0 or 1)
- Hand-adjusted 9 weights to get ~100% accuracy
- Needed hidden layer to solve it

**Iris is similar but real:**
- 4 inputs → 3 classes (0, 1, or 2)
- Keras automatically adjusts hundreds of weights
- Hidden layers help, but linear model works surprisingly well!

### What You'll Discover

1. **Real data isn't perfect:**
   - 95-97% accuracy is excellent
   - 100% accuracy would be suspicious (overfitting!)
   - Some overlap between Versicolor and Virginica

2. **Linear models can be powerful:**
   - If data is mostly linearly separable, simple models work
   - Not every problem needs deep networks

3. **Hidden layers add flexibility:**
   - More neurons = more curved boundaries
   - But diminishing returns - 32 units isn't much better than 8

4. **ML is stochastic:**
   - Same model, same data, different results
   - Random weight initialization matters
   - Must run multiple times and report statistics

### Key Activities

**Section 5:** Baseline Linear Model
- Build a model with NO hidden layers
- See if straight boundaries work
- Compare to XOR (which needed hidden layer)

**Section 6:** Hidden Layer Model
- Add one hidden layer with adjustable units
- Experiment with 2, 4, 8, 16, 32, 64 units
- Find the sweet spot

**Section 8:** Multiple Runs Experiment (NEW!)
- Run linear model 5 times with fresh initialization
- Run hidden layer model 5 times
- See box plots showing result distributions
- Learn to report: "Accuracy: 94.2% ± 1.5%"

### Questions

**Q13.** Did the linear model (no hidden layer) achieve high accuracy on Iris? Why or why not?

**Q14.** How much did adding a hidden layer improve accuracy? Was the improvement large or small?

**Q15.** Experiment with different values of `hidden_units` (2, 4, 8, 16, 32, 64). At what point do you see diminishing returns?

**Q16.** Compare this to Module 2 XOR training:
- What's similar? (Hint: gradient descent, weight updates)
- What's different? (Hint: manual vs. automatic, dataset)

**Q17.** Why is 95% accuracy considered "excellent" while 100% might be suspicious?

**Q18.** Looking at the petal scatter plot, why do you think Versicolor and Virginica are harder to separate than Setosa?

**Q19.** (Multiple Runs Section) After running the linear model 5 times, what was the mean accuracy ± standard deviation? Was the model consistent?

**Q20.** Did the hidden layer model show more or less variability than the linear model? Why might this be?

**Q21.** Looking at the box plots, do the two models' accuracy distributions overlap significantly? What does this tell you about whether hidden layers truly help?

---

## Module 4: Breast Cancer Classification

**Time:** ~30-35 minutes
**Dataset:** Wisconsin Breast Cancer (569 samples, 30 features, binary)
**Type:** Binary classification (medical diagnosis)

### What You'll Do

1. Load a real medical dataset from scikit-learn
2. Build a baseline linear model
3. Experiment with custom architectures (layers and units)
4. Analyze **confusion matrices** - which errors does the model make?
5. **Run multiple experiments** and track false negatives (missed cancers!)
6. Compare models statistically

### Learning Objectives

- Apply ML to high-stakes medical diagnosis
- Understand binary classification with many features (30 vs. 4 in Iris)
- Learn that simpler models can be more stable
- Interpret confusion matrices for medical decisions
- Understand medical ethics: false positive vs false negative trade-offs

### The Breast Cancer Dataset

**Source:** University of Wisconsin Hospitals
**Created:** Dr. William H. Wolberg, 1995
**Real-world use:** Helped develop computer-aided diagnosis systems

**Features (30 measurements from cell nucleus images):**
- Radius, texture, perimeter, area, smoothness, compactness, concavity, etc.
- For each: mean, standard error, and "worst" (largest) values
- All computed from digitized images of fine needle aspirate (FNA) of breast mass

**Target Classes:**
- **Malignant (0):** Cancerous - 212 samples (37%)
- **Benign (1):** Not cancerous - 357 samples (63%)

**The Task:** Given cell measurements, predict whether tumor is benign or malignant.

### Why This Matters: Medical Ethics

**False Positive (predicting cancer when there isn't any):**
- Patient stress and anxiety
- Additional tests (biopsy, imaging)
- Healthcare costs
- But NOT life-threatening

**False Negative (missing actual cancer):**
- Delayed treatment
- Disease progression
- Potentially life-threatening
- **THIS IS THE MOST CRITICAL ERROR**

In medical ML, we often care more about **which** errors the model makes than just overall accuracy!

### Connection to XOR

**XOR (Lab 1):**
- 2 inputs → binary output
- Couldn't solve with linear model
- Needed hidden layer

**Breast Cancer:**
- 30 inputs → binary output
- **CAN** solve with linear model! (Surprisingly!)
- Why? In 30-dimensional space, even straight boundaries separate classes well
- Hidden layers help slightly, but not dramatically

### What You'll Discover

1. **High-dimensional data is different:**
   - 30 features give model lots of information
   - Linear model achieves 94-97% accuracy
   - Simpler than expected!

2. **Simpler models can be better:**
   - Baseline linear model may be more stable
   - Complex models don't always help
   - Occam's Razor: prefer simpler explanations

3. **Medical ML requires statistical rigor:**
   - If false negatives vary from 2-4 depending on initialization, that's a problem!
   - Need ensemble methods, extensive validation
   - Clinical deployment requires FDA approval, multi-hospital trials

4. **Accuracy isn't everything:**
   - 97% sounds great, but 3 errors per 100 patients!
   - WHERE errors happen matters (confusion matrix)
   - Context determines acceptable trade-offs

### Key Activities

**Section 5:** Baseline Linear Model
- No hidden layers, just input → output
- Check confusion matrix - how many false negatives?
- Surprisingly high accuracy (~95-97%)

**Section 6:** Experiment with Architecture
- Adjustable `num_hidden_layers` (0, 1, 2)
- Adjustable `units_per_layer` (8, 16, 32, 64)
- Fill in results table manually

**Section 7:** Multiple Runs (NEW!)
- Run baseline 5 times, track false negatives
- Run custom model 5 times
- **Critical:** Does false negative count vary? By how much?
- Statistical comparison with box plots

### Key Visualizations

**Confusion Matrix Heatmap:**
```
                Predicted
            Malignant  Benign
Actual Mal     40        3     ← 3 False Negatives (MISSED CANCER!)
       Ben      2       69     ← 2 False Positives (false alarm)
```

**Box Plots:**
- Show distribution of accuracies across 5 runs
- Show distribution of false negatives across 5 runs
- Visual comparison: is custom model RELIABLY better?

### Questions

**Q22.** How did the baseline linear model perform on breast cancer data? Were you surprised? Why or why not?

**Q23.** Did adding hidden layers significantly improve accuracy? At what architecture did you see diminishing returns?

**Q24.** Looking at your confusion matrices, did you reduce false negatives (missed cancers) with more complex models? Is there a trade-off with false positives?

**Q25.** In medical diagnosis, which error is more concerning: false positive (predicting cancer when there isn't any) or false negative (missing actual cancer)? Explain your reasoning.

**Q26.** Compare Module 3 (Iris) and Module 4 (Breast Cancer):
- Which dataset benefited more from hidden layers?
- Why might this be? (Think about feature count and class separability)

**Q27.** Reflect on your journey from Module 0 to now:
- What's the connection between manually lifting XOR to 3D (Module 0) and hidden layers in Keras?
- How does `.fit()` relate to the gradient descent you saw in Module 2?
- What's the SAME between 2-feature XOR and 30-feature cancer diagnosis?

**Q28.** Given that a simple linear model achieves ~95% accuracy, why might doctors still want a more complex model? Why might they prefer the simpler one?

**Q29.** (Multiple Runs Section) After running your baseline model 5 times, what was the mean number of false negatives ± std? If this varies from 2-4 missed cancers depending on random initialization, is that acceptable for medical deployment?

**Q30.** Did your custom model reduce false negatives CONSISTENTLY across all 5 runs, or was the improvement inconsistent? What does the box plot tell you?

**Q31.** If the accuracy box plots for baseline and custom models overlap significantly, what does this mean about the reliability of any "improvement" you measured?

---

## Key Takeaways from Lab 4

### 1. Real-World ML Works!
- Neural networks classify real data (flowers, tumors) with high accuracy
- The principles from toy XOR apply to actual problems
- 95% accuracy is excellent - 100% is suspicious

### 2. TensorFlow/Keras Automates Everything
- `.fit()` does gradient descent automatically (Lab 2)
- Adam optimizer is like momentum, but smarter
- Hidden layers do dimension lifting automatically (Lab 1, Module 0)
- All the math you learned still applies - it's just hidden!

### 3. Simpler Models Can Be Sufficient
- Not every problem needs deep networks
- Iris: Linear model works well (mostly linearly separable)
- Breast Cancer: Linear model achieves 95%+ (high-dimensional blessing)
- Occam's Razor: prefer simple models when they work

### 4. ML is Stochastic - Statistics Are Essential
- Random initialization causes variability
- Professional practice: run multiple times, report mean ± std
- Example: "Accuracy: 94.2% ± 1.5%" not "Accuracy: 95%"
- Low std = stable, reliable model
- High std = sensitive to initialization

### 5. Medical ML Has Ethical Implications
- Accuracy isn't everything - TYPE of error matters
- False negatives (missed cancer) are more critical than false positives
- Variability in life-or-death predictions is unacceptable
- Real medical ML requires:
  - Ensemble methods (combine multiple models)
  - Extensive clinical trials
  - FDA approval
  - Validation on diverse patient populations

### 6. High-Dimensional Data Can Be Easier
- Breast Cancer (30 features) easier than Iris (4 features)
- More features give model more information
- "Blessing of dimensionality" - classes separate better in high dimensions
- But "curse of dimensionality" also exists (need more data)

---

## Professional ML Practices You Learned

1. **Train/Test Split** - Never evaluate on training data
2. **Feature Scaling** - Normalize features to similar ranges
3. **Baseline First** - Always try simple model before complex
4. **Multiple Runs** - Quantify variability with statistics
5. **Confusion Matrix** - Understand which errors happen
6. **Architecture Search** - Experiment systematically with layers/units
7. **Diminishing Returns** - Recognize when complexity doesn't help
8. **Statistical Reporting** - Report mean ± std, not single runs

---

## Connection to Real-World ML

**The principles you learned apply to:**

- **Computer Vision:** Classifying images (ImageNet: 1000 classes, millions of images)
- **Natural Language Processing:** Understanding text (GPT models: billions of parameters)
- **Recommendation Systems:** Netflix, Spotify, Amazon
- **Autonomous Vehicles:** Object detection, path planning
- **Medical Imaging:** X-rays, MRIs, CT scans
- **Financial Modeling:** Fraud detection, stock prediction
- **Speech Recognition:** Siri, Alexa, Google Assistant

**What's the same:**
- Gradient descent optimization
- Hidden layers for nonlinearity
- Train/test split for evaluation
- Statistical reporting

**What's different:**
- Much larger datasets (millions to billions of samples)
- Deeper networks (10s to 100s of layers)
- Specialized architectures (CNNs for images, RNNs for sequences)
- Massive compute (GPUs, TPUs, distributed training)
- Longer training times (hours to weeks)

---

## Before You Submit

Make sure you have:

- [ ] Completed Module 0 (Lifting Dimensions)
- [ ] Completed Module 1 (Anatomy of a Tiny Neural Network)
- [ ] Completed Module 2 (Training a Neural Network)
- [ ] Completed Module 3 (Iris Classification)
- [ ] Completed Module 4 (Breast Cancer Classification)
- [ ] Run the multiple experiments sections in Modules 3 and 4
- [ ] Filled in experiment results tables
- [ ] Answered all 31 questions (Q1-Q31)
- [ ] Experimented with different architectures
- [ ] Observed box plots and understood statistical comparison
- [ ] Written thoughtful, complete answers
- [ ] Discussed medical ethics implications (false pos vs false neg)

---

## Submission Instructions

Submit your completed answer sheet according to your instructor's guidelines (PDF upload, hardcopy, etc.).

**Congratulations!** You've completed your journey from hand-built gradient descent with abstract XOR patterns to real-world ML with industry-standard tools!

---

## Optional: Going Further

If you finish early or want to explore more:

1. **Try other datasets** from sklearn:
   - Wine classification (3 classes, 13 features)
   - Digits (10 classes, 64 features - small images of handwritten digits)

2. **Experiment with epochs:**
   - Does training for 200 epochs instead of 50/100 help?
   - Do you see overfitting (validation accuracy decreases)?

3. **Try different optimizers:**
   - SGD (basic stochastic gradient descent)
   - RMSprop (adaptive learning rate)
   - Compare convergence speed

4. **Learn about ensemble methods:**
   - Train 10 models, average their predictions
   - Does this reduce variability?

5. **Explore real datasets:**
   - Kaggle.com has thousands of ML competitions
   - UCI Machine Learning Repository
   - TensorFlow Datasets (TFDS)
Human: continue