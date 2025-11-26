# Lab 4 Module 1: Anatomy of a Tiny Neural Network

## Design Plan (Post-Crash Recovery)

### Core Insight from Crash Analysis

**The Problem:** A 2-2-1 network **can** solve XOR, but the clean separation happens in *hidden space* (h₁, h₂), not input space (x₁, x₂).

**The Solution:** Reframe the entire module to make hidden space transformation the primary goal, not input-space decision boundaries.

---

## Module Structure (15 minutes)

### 1. Network Architecture Explanation (2 min)

**Purpose:** Introduce the 2-2-1 architecture and connect to Module 0

**Content:**
- Diagram showing: Input(2) → Hidden(2) → Output(1)
- **9 total parameters:**
  - H1: w₁₁, w₁₂, b₁ (3 parameters)
  - H2: w₂₁, w₂₂, b₂ (3 parameters)
  - Output: w_out1, w_out2, b_out (3 parameters)
- Connection to Module 0: "Instead of manually adding x₃ = x₁×x₂, hidden neurons create h₁ and h₂ automatically"

### 2. Setup: XOR Data + Network Definition (2 min)

**Code cells:**
- Load XOR dataset (reuse from Module 0)
- Define 2-2-1 network class with forward pass
- Use sigmoid activation for hidden layer, sigmoid for output

### 3. Parameter Counting Exercise (2 min)

**Interactive question:**
- "How many total parameters does this network have?"
- Show breakdown by layer
- **Answer: 9** (leads to Q5)

### 4. Interactive Network Builder (7-8 min) ⭐ CORE ACTIVITY

**This is the main pedagogical component.**

#### 4.1 Four-Panel Visualization

**CRITICAL: Correct emphasis on panels**

```
┌─────────────────────┬─────────────────────┐
│  Top-Left:          │  Top-Right:         │
│  INPUT SPACE        │  HIDDEN SPACE ⭐    │
│  (x₁, x₂)           │  (h₁, h₂)          │
│  [Complex boundary] │  [LINEAR separator] │
│                     │  THE KEY PLOT!      │
├─────────────────────┼─────────────────────┤
│  Bottom-Left:       │  Bottom-Right:      │
│  Hidden Neuron 1    │  Hidden Neuron 2    │
│  Individual boundary│  Individual boundary│
└─────────────────────┴─────────────────────┘
```

**Panel Details:**

**Top-Left (Input Space):**
- Shows XOR data in (x₁, x₂)
- Shows final decision boundary (will be complex/curved)
- **Label:** "Final Boundary in Input Space (complex is OK!)"
- De-emphasize this - it's for illustration only

**Top-Right (Hidden Space):** ⭐ PRIMARY FOCUS
- Shows transformed XOR data in (h₁, h₂) coordinates
- Each point (x₁, x₂) becomes (h₁(x₁,x₂), h₂(x₁,x₂))
- Shows output layer's linear decision boundary (green line)
- **Label:** "Hidden Space Transformation - Your Main Goal!"
- **Visual Success:** Two clusters separated by green line

**Bottom-Left:**
- Shows H1's individual decision boundary in input space
- Color-codes regions by H1's activation
- **Label:** "What does Hidden Neuron 1 separate?"

**Bottom-Right:**
- Shows H2's individual decision boundary in input space
- Color-codes regions by H2's activation
- **Label:** "What does Hidden Neuron 2 separate?"

#### 4.2 Interactive Sliders (9 total)

**Layout:**
```
Hidden Layer 1 (H1):
  w₁₁: [-5, 5]
  w₁₂: [-5, 5]
  b₁:  [-5, 5]

Hidden Layer 2 (H2):
  w₂₁: [-5, 5]
  w₂₂: [-5, 5]
  b₂:  [-5, 5]

Output Layer:
  w_out1: [-5, 5]
  w_out2: [-5, 5]
  b_out:  [-5, 5]
```

All sliders update plots in **real-time**.

#### 4.3 Accuracy Display

Show current accuracy at top:
```
Current Accuracy: 73.5%
Status: Getting there! Try adjusting the output layer.
```

#### 4.4 Strategic Guidance System

**Adaptive hints based on accuracy:**

**99-100% Accuracy:**
```
🎉 AMAZING! You solved XOR!

Look at the top-right panel (h₁, h₂):
- XOR has been transformed into two linearly separable clusters
- The green line is a simple straight boundary
- This is exactly what hidden layers do!
```

**90-99% Accuracy:**
```
🎯 Very close! You're almost there!

Strategy:
- Your hidden neurons are working well
- Fine-tune the output layer (w_out1, w_out2, b_out)
- Watch the green line in the top-right panel
```

**75-90% Accuracy:**
```
📈 Good progress!

Strategy:
- Check the bottom panels: Are H1 and H2 creating useful splits?
- Goal: H1 should separate one way, H2 another way
- Their combination in hidden space should separate XOR
```

**<75% Accuracy:**
```
💡 Strategy Guide:

Step 1: Set H1 to separate left/right
  - Try: w₁₁=5, w₁₂=0, b₁=0

Step 2: Set H2 to separate top/bottom
  - Try: w₂₁=0, w₂₂=5, b₂=0

Step 3: Tune output layer to combine them
  - Watch the top-right panel for linear separation!
```

#### 4.5 Load Example Solution Button

**Available from the start** (not at the end!)

Button: "📖 Load Example Solution"

When clicked:
- Sets parameters to a working solution
- Shows explanation of what each neuron does
- Students can then tweak to explore

**Example Solution:**
```python
# One possible solution:
H1_params = {'w11': 5, 'w12': 0, 'b1': 0}    # Separates left/right
H2_params = {'w21': 0, 'w22': 5, 'b2': 0}    # Separates top/bottom
Out_params = {'w_out1': 5, 'w_out2': 5, 'b_out': -4}  # XOR logic
```

**Why provide this early?**
1. Reduces student frustration in 9D parameter space
2. Gives concrete target to aim for
3. Helps them understand the **pattern** of what works
4. Allows exploration by tweaking a working solution

#### 4.6 Visual Target Guidance

Show students what they're aiming for:

```markdown
### 🎯 Target Pattern for Hidden Space

You want the top-right panel to look something like this:

    h₂
    ↑
  + │  red    blue
    │   x       .
    │   x       .
  0 ├──────────────→ h₁
    │   .       x
    │   .       x
  - │ blue     red
```

**Key insight:** When XOR forms diagonal clusters in (h₁, h₂) space,
a simple straight line can separate them!

### 5. Understanding What Happened (2 min)

**Explanation section:**

```markdown
## What Did You Just Do?

### The Manual Approach (You):
1. Adjusted 9 sliders by hand
2. Tried to make hidden space linearly separable
3. Saw how two perceptrons can combine to solve XOR

### Why This Was Hard:
- 9-dimensional parameter space
- Non-obvious which direction to adjust
- Trial and error with visual feedback

### Coming Next (Module 2):
**Automatic training does this for you!**
- Gradient descent finds good parameters automatically
- Scales to millions of parameters
- You just saw WHY hidden layers work - next you'll see HOW they learn
```

### 6. Example Solution Walkthrough (1 min)

**Even if students loaded it earlier, explain it:**

```markdown
## One Working Solution Explained

### Hidden Neuron 1 (H1):
- w₁₁=5, w₁₂=0, b₁=0
- **What it does:** Creates a vertical line at x₁=0
- **Effect:** Separates left points from right points

### Hidden Neuron 2 (H2):
- w₂₁=0, w₂₂=5, b₂=0
- **What it does:** Creates a horizontal line at x₂=0
- **Effect:** Separates bottom points from top points

### Output Neuron:
- w_out1=5, w_out2=5, b_out=-4
- **What it does:** Computes `sigmoid(5h₁ + 5h₂ - 4)`
- **Logic:** Predicts 1 when BOTH h₁ and h₂ are active (XOR pattern!)

### The Hidden Space Magic:
Look at the top-right panel:
- Bottom-left corner (x₁<0, x₂<0): h₁≈0, h₂≈0 → output≈0 (blue)
- Top-right corner (x₁>0, x₂>0): h₁≈1, h₂≈1 → output≈0 (blue)
- Top-left corner (x₁<0, x₂>0): h₁≈0, h₂≈1 → output≈1 (red)
- Bottom-right corner (x₁>0, x₂<0): h₁≈1, h₂≈0 → output≈1 (red)

**This is XOR logic implemented with perceptrons!**
```

### 7. Key Takeaways (1 min)

```markdown
## Key Takeaways from Module 1

### 1. Hidden Layers Create New Dimensions
- Just like you manually added x₃=x₁×x₂ in Module 0
- Hidden neurons create h₁, h₂ automatically during training
- These new dimensions make the problem solvable

### 2. Separation Happens in Hidden Space
- **Not in input space!** The input boundary can be complex
- The key transformation is (x₁, x₂) → (h₁, h₂)
- Linear separation in hidden space = complex boundary in input space

### 3. Each Hidden Neuron is a Perceptron (from Lab 3)
- H1 and H2 are both perceptrons with sigmoid activation
- Each creates one boundary/transformation
- The output layer combines their outputs

### 4. Manual Tuning Doesn't Scale
- 9 parameters was already hard
- Modern networks have millions or billions of parameters
- **Next:** See how gradient descent learns these automatically!
```

---

## Questions (Q5-Q7)

**Q5:** How many total parameters does the 2-2-1 network have? Break down by layer.
- **Answer:** 9 total (H1: 3, H2: 3, Output: 3)
- **Purpose:** Reinforces architecture understanding

**Q6:** Describe what each hidden neuron (H1 and H2) separated in your solution. What patterns did they detect?
- **Example answer:** "H1 separated left from right, H2 separated top from bottom"
- **Purpose:** Tests observation and understanding of individual neuron roles

**Q7:** Look at the hidden space plot (top-right panel). Explain how the XOR data was transformed in (h₁, h₂) space and why this made it easier to separate.
- **Key concept:** "XOR formed linearly separable clusters in hidden space"
- **Purpose:** Tests core understanding of hidden layer transformation

---

## Implementation Notes

### Critical Design Principles

1. **Hidden space is the hero**
   - Make top-right panel visually prominent (larger? highlighted border?)
   - Frequent references in text
   - Success criteria focuses on hidden space

2. **Early example solution**
   - Button available from start of activity
   - Not a "cheat" - it's a learning scaffold
   - Reduces frustration, enables exploration

3. **Clear visual targets**
   - Show cartoon of desired hidden-space pattern
   - Use contrasting colors for separation
   - Make green decision line prominent

4. **Adaptive guidance**
   - Accuracy-based hints
   - Strategic direction, not just "try again"
   - Progressive reveal of full strategy

5. **Explicit mental model**
   - State clearly: "separation happens AFTER transformation"
   - Repeated emphasis: hidden space → simple, input space → complex
   - Connect back to Module 0's manual feature engineering

### Technical Implementation

**Plotting Requirements:**
- Real-time updates (ipywidgets + matplotlib or plotly)
- Color consistency across panels
- Clear legends and labels
- Grid layout for four panels

**Network Architecture:**
```python
class TinyNetwork:
    def __init__(self):
        # Hidden layer 1
        self.w11, self.w12, self.b1 = 0, 0, 0
        # Hidden layer 2
        self.w21, self.w22, self.b2 = 0, 0, 0
        # Output layer
        self.w_out1, self.w_out2, self.b_out = 0, 0, 0

    def forward(self, x1, x2):
        # Hidden activations
        h1 = sigmoid(self.w11 * x1 + self.w12 * x2 + self.b1)
        h2 = sigmoid(self.w21 * x1 + self.w22 * x2 + self.b2)
        # Output
        out = sigmoid(self.w_out1 * h1 + self.w_out2 * h2 + self.b_out)
        return out, h1, h2
```

**Accuracy Calculation:**
```python
def calculate_accuracy(network, X, y):
    predictions = []
    for x1, x2 in X:
        out, _, _ = network.forward(x1, x2)
        predictions.append(1 if out > 0.5 else 0)
    return np.mean(np.array(predictions) == y) * 100
```

---

## Connection to Module 2

**Module 1 ends with:**
"You just manually tuned 9 parameters to solve XOR. This took you several minutes of trial and error. Imagine if you had 1000 parameters... or 1 million!

**In Module 2:** You'll see gradient descent automatically find these parameters through training. The network will learn the same hidden-space transformation you just discovered - but completely on its own!"

---

## Testing Checklist

Before deployment:

- [ ] Four-panel layout renders correctly
- [ ] All 9 sliders update plots in real-time
- [ ] Hidden space plot shows (h₁, h₂) transformation correctly
- [ ] Accuracy display updates
- [ ] Guidance messages adapt to accuracy level
- [ ] Example solution button loads working parameters
- [ ] Students can achieve 99%+ accuracy
- [ ] Text cells clearly emphasize hidden space
- [ ] Questions Q5-Q7 are clearly stated
- [ ] Connection to Module 0 is explicit
- [ ] Setup for Module 2 is clear

---

## Success Metrics

**Students should leave Module 1 able to:**
1. ✅ Count parameters in a simple network
2. ✅ Understand that hidden neurons = individual perceptrons
3. ✅ See that XOR becomes separable in hidden space
4. ✅ Explain how transformation (x₁,x₂)→(h₁,h₂) enables separation
5. ✅ Appreciate why automatic training is necessary
6. ✅ Connect to Module 0's manual feature engineering
7. ✅ Ready for Module 2's gradient descent training

**The "aha moment":**
"Oh! The hidden layer creates a NEW space where XOR is easy to separate - just like when we manually added x₃ in Module 0, but now it happens automatically!"
