# Lab 3: Activation Functions & Nonlinearity
## Student Handout

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In this lab, you'll explore why simple linear models fail on certain patterns and how activation functions provide the nonlinear ingredient that makes neural networks powerful:

1. **When linear models fail** – Recognizing patterns that can't be separated by straight lines
2. **How activation functions work** – Understanding functions that "bend space"
3. **Building perceptrons** – Combining weighted sums with activation functions
4. **Understanding limitations** – Why single perceptrons can't solve everything
5. **The need for layers** – Why neural networks have multiple layers

### Lab Structure

This lab consists of **5 modules** that you'll complete in order:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | When Straight Lines Fail | ~10 min | In-class |
| 1 | Activation Functions – Bending Space | ~15-20 min | In-class |
| 2 | Activation Functions in Detail | ~15 min | In-class |
| 3 | Building a Perceptron | ~20 min | In-class |
| 4 | Testing the Perceptron's Limits | ~15 min | In-class |

**Total Time:** ~75-80 minutes

### Working in Groups

- Work in **small groups** (2-4 people) or individually
- One person shares their screen running the notebooks
- Everyone participates in discussion
- All group members can use their own notebooks

### Key Concepts

> **PERCEPTRON**: A basic unit that:
> - Takes inputs and computes a weighted sum
> - Applies an activation function
> - Produces an output used for classification
>
> **ACTIVATION FUNCTION**: A nonlinear function that:
> - "Warps" or transforms the input space
> - Allows models to learn complex patterns
> - Is the key ingredient for neural network power

### Connection to Previous Labs

**Lab 1:** You learned about models, parameters, and optimization

**Lab 2:** You learned about gradient descent for automatic parameter updates

**Lab 3:** You'll learn why nonlinearity (activation functions) is crucial for solving complex problems

---

## Module 0: When Straight Lines Fail

**Time:** ~10 minutes
**Type:** In-class

### What You'll Do

Explore three datasets and try to separate them with straight lines:
1. **Dataset 1:** Two separated clouds (linearly separable)
2. **Dataset 2:** XOR pattern - four corners
3. **Dataset 3:** Concentric circles - inner vs outer ring

### Learning Objectives

- Recognize patterns that cannot be separated by straight lines
- Understand fundamental limitations of linear models
- Build motivation for nonlinear methods

### Key Insight

Some patterns (XOR, circles) **cannot** be separated by ANY straight line, no matter how you adjust the slope and intercept!

This motivates the need for:
- **Activation functions** to introduce nonlinearity
- **Multiple layers** to combine decision boundaries

### Interactive Activity

Use sliders to adjust the slope and intercept of a line and try to separate:
- Blue points on one side
- Red points on the other side

You'll discover that:
- ✅ **Dataset 1 works** – A straight line can separate it perfectly
- ❌ **Dataset 2 (XOR) fails** – No line works (best is ~50% accuracy)
- ❌ **Dataset 3 (circles) fails** – No line works (best is ~50-60% accuracy)

### Questions

**Q1.** Which dataset(s) can be separated by a straight line?

**Q2.** For the XOR pattern (Dataset 2), what happens no matter how you adjust the line? Why can't ANY straight line separate it?

**Q3.** Why can't a straight line separate the circular ring pattern (Dataset 3)? What kind of boundary shape would you need instead?

---

## Module 1: Activation Functions – Bending Space

**Time:** ~15-20 minutes
**Type:** In-class

### What You'll Do

Learn about four activation functions and see how they "warp" 2D space:
1. **Sigmoid:** Smooth S-curve, outputs [0, 1]
2. **Tanh:** Smooth S-curve, outputs [-1, 1]
3. **ReLU:** Rectified Linear Unit, outputs [0, ∞)
4. **Step:** Hard jump, outputs {0, 1}

### Learning Objectives

- Describe what activation functions do to numbers
- Visualize how activations warp/bend 2D coordinate space
- Understand that activations are the "nonlinear ingredient"
- Recognize that straight rules after warping create curved boundaries

### The Big Idea

When you apply an activation function to BOTH coordinates (x₁ and x₂), it's like stretching or compressing a rubber sheet with a grid drawn on it.

**Key visualization:**
- **Left plot:** Original grid (square, evenly spaced)
- **Right plot:** Warped grid after activation (curves, compressed)

Straight lines in the original space become **curved** lines after warping!

### Interactive Exploration

1. **Section 2:** Explore individual activation functions
   - See their graphs
   - Test example values
   - Learn their behavior at extremes

2. **Section 4:** Watch grids warp
   - Compare original vs warped space
   - See how sigmoid compresses everything to [0,1] × [0,1]
   - Observe how ReLU collapses negative quadrants

3. **Section 5:** See curved boundaries
   - Simple rule in activated space: `y₁ + y₂ > threshold`
   - This rule is a **straight line** in activated space
   - But looks **curved** in original space!

### Key Insight

**Simple linear rules AFTER activation warping** → **Flexible curved boundaries BEFORE warping**

This is why activation functions are powerful!

### Questions

**Q4.** What happens to very large positive/negative inputs for sigmoid and tanh?

**Q5.** Which activation changes most rapidly near x = 0?

**Q6.** For sigmoid, where do faraway points end up in activated space?

**Q7.** For ReLU, what happens to points where x₁ or x₂ is negative?

**Q8.** Which activation warps the grid the most?

**Q9-Q10.** Describe the boundary appearance in original vs activated space

**Q11.** Explain how activations help build flexible decision rules

---

## Module 2: Activation Functions in Detail

**Time:** ~15 minutes
**Type:** In-class

### What You'll Do

Compare activation functions side-by-side and learn their properties:
- Output ranges
- Smoothness
- Saturation behavior
- When to use each one

### Learning Objectives

- Compare how different activations handle the same input
- Understand key properties: smoothness, range, behavior at extremes
- Choose appropriate activations for different situations
- Connect activation properties to their effects on learning

### Key Properties Comparison

| Function | Range | Smooth? | Centered? | Kills Negatives? |
|----------|-------|---------|-----------|------------------|
| **Sigmoid** | [0, 1] | Yes | No | No (→~0) |
| **Tanh** | [-1, 1] | Yes | Yes | No (→~-1) |
| **ReLU** | [0, ∞) | No (corner) | No | Yes (→0) |
| **Step** | {0, 1} | No (jump) | No | Yes (→0) |

### Important Concepts

**Saturation:**
- When output barely changes despite large input changes
- Happens with Sigmoid/Tanh at extreme values
- Creates "flat regions" where learning is slow
- ReLU doesn't saturate for positive inputs → faster learning!

**When to Use Each:**
- **Hidden layers (modern):** ReLU (most popular)
- **Output layer (probabilities):** Sigmoid (0 to 1 range)
- **Hidden layers (older networks):** Tanh (centered at zero)
- **Theoretical understanding:** Step (simplest, but can't train with gradient descent)

### Interactive Comparison

Use the slider to test the same input on all four activations:
- See how outputs differ
- Understand range differences
- Observe saturation behavior

### Questions

**Q12.** Which activation outputs are between 0 and 1? Why useful for probabilities?

**Q13.** Which activation is centered at zero?

**Q14.** What does "saturation" mean? Which functions saturate?

**Q15.** What advantage does ReLU have for large positive inputs?

**Q16.** Why is Step bad for gradient descent training?

---

## Module 3: Building a Perceptron

**Time:** ~20 minutes
**Type:** In-class

### What You'll Do

Build complete perceptrons by combining weighted sums with activations:
- Test on 5 different linearly separable datasets
- Adjust weights (w₁, w₂) and bias (b)
- See how parameters control the decision boundary
- Achieve 100% accuracy on multiple datasets

### Learning Objectives

- Understand the two-step perceptron process
- Interactively adjust weights and bias to classify data
- See how perceptrons create decision boundaries
- Connect perceptrons to neural networks

### The Two Steps

A perceptron is incredibly simple:

**Step 1: Weighted Sum**
```
z = w₁×x₁ + w₂×x₂ + b
```
- w₁, w₂ = weights (how much each input matters)
- b = bias (shifts the decision boundary)
- z = "pre-activation" value

**Step 2: Activation**
```
output = activation(z)
```
- Applies nonlinearity
- Often squashes to useful range (0-1, etc.)

**Complete perceptron:**
```
output = activation(w₁×x₁ + w₂×x₂ + b)
```

### The Five Datasets

You'll practice on datasets with different orientations:
1. **Diagonal (↗):** Classes separated diagonally
2. **Vertical (←→):** Classes separated vertically
3. **Horizontal (↑↓):** Classes separated horizontally
4. **Diagonal (↖):** Opposite diagonal
5. **Tight Clusters:** Requires precise parameters

All are linearly separable – perceptrons can solve them all!

### Understanding Parameters

**Weights (w₁, w₂):**
- Control **orientation** (angle) of decision boundary
- If w₁ ≫ w₂: boundary is mostly vertical (x₁ matters more)
- If w₂ ≫ w₁: boundary is mostly horizontal (x₂ matters more)
- If w₁ ≈ w₂: boundary is diagonal

**Bias (b):**
- Controls **position** of decision boundary
- Shifts the boundary without changing its angle
- Positive bias → shifts toward negative direction
- Negative bias → shifts toward positive direction

**Activation:**
- For linearly separable data, **all activations work equally well!**
- Choice doesn't affect whether you CAN classify correctly
- Only affects output range and smoothness

### Tips for Success

- **Dataset 2 (vertical):** Try large |w₁|, small w₂
- **Dataset 3 (horizontal):** Try large |w₂|, small w₁
- **Diagonal datasets:** Try similar |w₁| and |w₂|
- **Tight clusters:** Requires fine-tuning all parameters

### Connection to Neural Networks

A **neural network** is just many perceptrons in layers:
- Each perceptron has its own weights and bias
- Outputs from one layer become inputs to the next
- More layers + more perceptrons = more complex patterns!

**Single perceptron:** 1 decision boundary
**Multi-layer network:** Many combined boundaries → complex regions

### Questions

**Q17.** Write the two perceptron steps in your own words

**Q18.** What do weights control about the boundary?

**Q19.** What does bias control about the boundary?

**Q20.** Compare vertical vs horizontal datasets - which weight needed to be larger for each?

**Q21.** Did activation choice affect classification success? Why/why not?

**Q22.** Why might multiple layers solve problems a single perceptron can't?

---

## Module 4: Testing the Perceptron's Limits

**Time:** ~15 minutes
**Type:** In-class

### What You'll Do

Test perceptrons on the "impossible" problems from Module 0:
- XOR pattern (four corners)
- Concentric circles (inner vs outer)

Discover that **no single perceptron can solve these**, no matter what parameters or activation you use!

### Learning Objectives

- Test perceptrons on non-linearly separable problems
- Understand why single perceptrons fail
- Connect this limitation to multi-layer networks
- Recognize that activations alone don't solve everything

### The Challenge

Try to find perceptron parameters (w₁, w₂, b, activation) that work for XOR and circles.

**Prediction:** You won't be able to get much better than 50% accuracy (random guessing)!

**Why?**
- A perceptron creates **ONE** straight decision boundary
- XOR needs at least **TWO** boundaries
- Circles need a **CIRCULAR** boundary
- Even with activation functions, one perceptron = one boundary

### Why Perceptrons Fail

**For XOR:**
- 4 clusters in 4 corners
- Bottom-left & top-right = Class 0
- Top-left & bottom-right = Class 1
- Need at least 2 lines to separate 4 regions
- One line can only separate 2 regions max

**For Circles:**
- Inner circle = Class 0
- Outer ring = Class 1
- Need circular boundary
- Straight line always has both classes on both sides

### The Critical Insight

Even though activation functions **warp space**, a single perceptron still creates only **one decision boundary**.

This is a **fundamental mathematical limitation**, proven by Minsky & Papert in 1969.

### The Solution: Multi-Layer Networks

**Single perceptron:**
```
Input → [Perceptron] → Output
Result: 1 boundary → Can't solve XOR or circles
```

**Multi-layer network:**
```
Input → [Hidden Layer: 2-3 perceptrons] → [Output Perceptron] → Output
Result: Multiple boundaries combined → Can solve XOR and circles!
```

**How it works:**
1. Hidden layer perceptrons each create one boundary
2. Output perceptron combines these boundaries
3. Result: Complex, nonlinear decision regions!

**Example for XOR:**
- Hidden perceptron 1: Separates left from right
- Hidden perceptron 2: Separates top from bottom
- Output perceptron: Combines these to identify diagonal pattern

### Historical Context

This limitation caused the first "AI winter" in the 1970s. People thought neural networks were fundamentally too weak.

The solution (multi-layer networks + backpropagation) was developed in the 1980s, leading to today's deep learning revolution!

### Questions

**Q23.** What was your best accuracy on XOR? Close to 100%?

**Q24.** What was your best accuracy on circles? Close to 100%?

**Q25.** Why can't one straight line separate XOR's four clusters?

**Q26.** Why does a straight line always have both classes on both sides for circles?

**Q27.** How many hidden layer perceptrons would you need for XOR? Why?

---

## Key Takeaways

By completing this lab, you should understand:

### 1. Linear Model Limitations
- Some patterns (XOR, circles) cannot be separated by straight lines
- This is a fundamental mathematical limitation
- Recognizing these patterns is crucial for choosing models

### 2. Activation Functions
- Transform inputs nonlinearly ("warp space")
- Four main types: Sigmoid, Tanh, ReLU, Step
- Each has different properties (range, smoothness, saturation)
- Modern networks primarily use ReLU for hidden layers

### 3. Space Warping
- Activations create curved boundaries from straight rules
- Straight line in activated space → curved line in original space
- This is the "nonlinear ingredient" for neural networks

### 4. Perceptron Architecture
- Two simple steps: weighted sum → activation
- Parameters (w₁, w₂, b) control decision boundary
- Can perfectly classify linearly separable data
- **Cannot** solve XOR or circles (single boundary limitation)

### 5. Need for Layers
- Single perceptron = one decision boundary
- Multi-layer networks = multiple combined boundaries
- Hidden layers enable complex, nonlinear decision regions
- This is why neural networks are powerful!

---

## Connecting to AI and Machine Learning

Everything you did in this lab connects directly to real neural networks:

| Lab Activity | Real Neural Network Equivalent |
|--------------|-------------------------------|
| Testing activation functions | Choosing activation for network layers |
| Adjusting perceptron weights | Training weights with backpropagation |
| Seeing decision boundaries | Understanding what neurons learn |
| Single perceptron limitations | Why we need deep architectures |
| Multi-layer solution concept | Deep learning / hidden layers |
| XOR problem | Classic benchmark for neural networks |

### Real-World Applications

**Single perceptron models:**
- Simple binary classification (spam vs not spam with linear features)
- Logistic regression (perceptron with sigmoid activation)

**Multi-layer neural networks (what you now understand the need for):**
- Image recognition (CNNs)
- Natural language processing (Transformers)
- Game playing (AlphaGo)
- Generative models (ChatGPT, DALL-E)

### Next Steps in Your AI Journey

- **Next labs:** Learn about multi-layer networks and backpropagation
- **Real networks:** Millions of perceptrons, dozens of layers
- **Modern architectures:** Clever combinations (residual connections, attention mechanisms)
- **Training:** Gradient descent updates ALL weights simultaneously
- **Applications:** Solve problems far more complex than XOR!

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules
- [ ] Answered all 27 questions (Q1-Q27)
- [ ] Tested perceptrons on all datasets in Module 3
- [ ] Attempted to classify XOR and circles in Module 4
- [ ] Understood why single perceptrons have limitations
- [ ] Grasped why neural networks need multiple layers

---

## Additional Resources

### If You Want to Learn More

**Activation Functions:**
- Why ReLU works so well: "dying ReLU" problem and variants (Leaky ReLU, ELU)
- Modern alternatives: GELU, Swish, Mish

**Historical Papers:**
- Minsky & Papert (1969): "Perceptrons" - proving the XOR limitation
- Rumelhart et al. (1986): Backpropagation learning

**Mathematical Depth:**
- Universal Approximation Theorem: Multi-layer networks can approximate any function
- Why depth matters: Deeper networks are more efficient than wider networks

**Modern Deep Learning:**
- Convolutional Neural Networks (CNNs) for images
- Recurrent Neural Networks (RNNs) for sequences
- Transformers for language (attention mechanism)

---

**Questions or Issues?**
- Check the LMS discussion board
- Ask your instructor or TA
- Review the lab modules for additional context
- Work through examples with your group

Great work! 🎉 You now understand the fundamental building blocks of neural networks!
