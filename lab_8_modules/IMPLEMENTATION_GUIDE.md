# Lab 8: Diffusion Models - Implementation Guide

**Course:** DATA 1010 – Artificial Intelligence in Action
**Target:** First-year students with no math/CS background
**Duration:** 60-75 minutes total
**Status:** Modules 0-3 complete, Module 4 + documentation remaining

---

## 🎯 Project Vision

Create a hands-on lab teaching diffusion models (the technology behind DALL-E, Midjourney, Stable Diffusion) through progressively complex demonstrations:

1. **Conceptual foundation** (Module 0)
2. **Forward diffusion demo** (Module 1)
3. **Train toy denoiser** (Module 2)
4. **Generate from noise** (Module 3)
5. **Professional pre-trained model** (Module 4) - **TO DO**

---

## ✅ Completed Modules

### **Module 0: What Is a Diffusion Model?** ✓ COMPLETE
**File:** `lab_8_module_0_what_is_diffusion.ipynb`
**Type:** Markdown-only, conceptual
**Time:** 5-8 minutes

**Key Features:**
- Opening analogy: "Sculpting in reverse"
- Two-phase process explanation (forward/reverse)
- Multiple analogies (scrambled eggs, fog, detective work)
- Comparison table: Lab 7 (CNNs) vs Lab 8 (Diffusion)
- Real-world applications (DALL-E, Midjourney, Stable Diffusion)
- Questions Q1-Q4 (predict-first design)

**Pedagogical Patterns Used:**
- Heavy use of analogies for intuition building
- Connection to previous labs (especially Lab 7)
- No code, pure conceptual understanding
- Questions encourage prediction before seeing implementation

---

### **Module 1: Forward Diffusion Demo** ✓ COMPLETE
**File:** `lab_8_module_1_forward_diffusion.ipynb`
**Type:** Colab notebook with code
**Time:** 8-12 minutes

**Key Features:**
- Implements forward diffusion on single MNIST digit
- Linear beta schedule: 0.0001 to 0.02, 200 timesteps
- **Three-tier math presentation** (symbols → plain English → pseudocode)
- Progressive noise addition visualization (t=0,25,50,75,100,125,150,175,199)
- SNR analysis (both linear and dB scales with annotations)
- Multiple noise realizations experiment (shows stochasticity)
- Questions Q5-Q9

**Technical Implementation:**
```python
# Forward diffusion formula
noisy = sqrt(alpha_bar_t) * original + sqrt(1 - alpha_bar_t) * noise
```

**Visualizations:**
- Noise schedule plots (alpha_bar and noise weight over time)
- Progressive noising grid with color-coded borders (green/orange/red)
- SNR curves with annotations marking key regions
- 5 different noise realizations at t=100

**Educational Impact:**
- Students see information loss is gradual, not sudden
- Understand noise schedules control destruction rate
- Learn that forward process is deterministic given image + timestep

---

### **Module 2: Training a Toy Denoiser** ✓ COMPLETE
**File:** `lab_8_module_2_toy_denoiser.ipynb`
**Type:** Colab notebook with training
**Time:** 18-22 minutes
**Training Time:** 2-3 minutes on T4 GPU

**Key Features:**
- **User-configurable NUM_CLASSES** (2, 4, 6, or 10 digits) - default 4
- Dataset: MNIST digits 0-3, downsampled to 16×16
- Simplified U-Net architecture with timestep embeddings
- Custom training loop with on-the-fly noise addition
- Training progress tracking and visualization
- Denoising performance at multiple timesteps (t=25,50,100,150)
- Learned filter visualization

**Architecture Details:**
```
Simplified U-Net:
- Input: 16×16 noisy image + timestep embedding (32-dim sinusoidal)
- Encoder: Conv(32)→Pool→Conv(64)→Pool
- Bottleneck: Conv(128)
- Decoder: Upsample→Concat→Conv(64)→Upsample→Concat→Conv(32)
- Output: Predicted noise (16×16)
Total parameters: ~100,000
```

**Training Configuration:**
- Loss: MSE (mean squared error)
- Optimizer: Adam (lr=1e-3)
- Epochs: 8
- Batch size: 128
- Target loss: ~0.05

**Visualizations:**
- Training loss curve with target line
- Denoising examples at 4 timesteps (shows: original → noisy → actual noise → predicted noise → denoised)
- Learned filters from first conv layer (32 filters, 3×3)

**Questions:** Q10-Q15 (includes prediction before training)

**Connection to Lab 7:** Comparison table showing similarities/differences with CNN training

---

### **Module 3: Multi-Step Denoising (Reverse Diffusion)** ✓ COMPLETE
**File:** `lab_8_module_3_iterative_denoising.ipynb`
**Type:** Colab notebook with generation
**Time:** 12-15 minutes
**Generation Time:** ~20-30 seconds per image (200 steps)

**Key Features:**
- **Smart model handling:** Checks if model exists from Module 2, offers quick training if not
- Complete DDPM reverse diffusion implementation
- Trajectory visualization (10 snapshots from noise → digit)
- Multiple sample generation (12 different digits)
- Step count comparison (50 vs 100 vs 200 steps)
- Bridge to Module 4 with comparison table

**Reverse Diffusion Algorithm:**
```python
# Simplified DDPM formula
x_{t-1} = (x_t - noise_removal_amount) / signal_scale + small_random_noise

# Full implementation includes:
# - Noise prediction from model
# - Scaled noise removal
# - Stochastic noise addition (except at t=0)
```

**Three-Tier Math Presentation:**
- Symbols: Full DDPM equation
- Plain English: "Next step = (current - scaled predicted noise) / signal scale + tiny random noise"
- Pseudocode: Clear algorithmic steps

**Visualizations:**
- Generation trajectory with color-coded borders (red/orange/green)
- 12 different generated samples showing variety
- Quality comparison across step counts

**Questions:** Q16-Q20 (observations and critical thinking about quality/limitations)

**Educational Highlights:**
- Students see their own trained model generate images!
- Imperfect results are educational (authentic limitations)
- Clear explanation of why professional models need billions of parameters
- Emphasis: core algorithm is SAME as DALL-E, just different scale

**Critical Design Decision: Model Portability**
- Module checks if model exists from Module 2
- If not found, provides self-contained quick training option
- This handles Colab session resets gracefully

---

## 🚧 Remaining Work

### **Module 4: Pre-Trained Diffusion Model** - TO DO
**File:** `lab_8_module_4_pretrained_sampling.ipynb` (needs creation)
**Type:** Colab notebook with PyTorch
**Time:** 12-15 minutes

**Requirements:**
- Use Hugging Face `diffusers` library
- Model: `google/ddpm-cifar10-32` (~200 MB)
- Generate 32×32 RGB images (10 object classes)
- Step comparison: 25 vs 50 vs 100 inference steps
- Questions Q21-Q25

**Framework Switch - PyTorch:**
- Modules 0-3 use Keras/TensorFlow (consistent with course)
- Module 4 uses PyTorch (easier with diffusers library, no conflicts)
- **Important:** Explain to students: "PyTorch is an alternative to TensorFlow"
- Emphasize: same concepts, different syntax
- Students don't need to understand PyTorch details

**Key Implementation:**
```python
# Install diffusers
!pip install -q diffusers

# Load model
from diffusers import DDPMPipeline
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
pipeline = pipeline.to("cuda")  # Use GPU

# Generate
images = pipeline(batch_size=16, num_inference_steps=50).images
```

**Visualizations Needed:**
- Grid of generated images (4×4 or similar)
- Step count comparison (25 vs 50 vs 100 steps)
- Quality comparison to Module 3 toy model

**Educational Goals:**
- Show dramatic quality improvement with scale
- Bridge to text-to-image (DALL-E, Stable Diffusion)
- Explain how text conditioning works (conceptually)
- Connect unconditional (CIFAR-10) to conditional (text-guided) generation

**Connection Section:**
- Table: Toy model vs DALL-E/Stable Diffusion
- Explain scaling challenges (compute, data, training time)
- Text-to-image conditioning mechanism
- Brief mention: video, medical imaging, molecule design

---

### **Student Handout** - TO DO
**File:** `Lab_8_Student_Handout.md` (needs creation)
**Reference:** `Lab_7_Student_Handout.md` as template

**Required Sections:**
1. **Overview**
   - What You'll Learn Today
   - Lab Structure (5 modules with time estimates)
   - Working in Groups
   - Key Concepts (definitions)
   - Connection to Previous Labs

2. **Module 0-4 Descriptions**
   - Each module gets its own section
   - Learning objectives
   - What you'll do
   - Key insights
   - Questions (Q1-Q25)

3. **Key Takeaways**
   - Summary of main concepts
   - What students should understand by the end

4. **Connection to Real-World AI**
   - **"From MNIST Digits to DALL-E"** section
   - Table comparing toy model vs professional models
   - Text-to-image explanation
   - Applications: DALL-E, Midjourney, Stable Diffusion
   - Brief mentions: video, medical imaging, audio (future lab?)

5. **Before You Submit** (checklist)

6. **Submission Instructions**

7. **Additional Resources** (optional reading)

**Pedagogical Elements:**
- Predict-first questions (Q10, Q16, Q20)
- Connection tables to previous labs
- Real-world application examples
- Emphasis on conceptual understanding over math

---

### **Answer Sheet** - TO DO
**File:** `Lab_8_Answer_Sheet.md` (needs creation)
**Reference:** `Lab_7_Answer_Sheet.md` as template

**Format:**
```markdown
# Lab 8: Diffusion Models - Answer Sheet

**Name:** _________________
**Date:** _________________

---

## Module 0: What Is a Diffusion Model?

**Q1.** In your own words, what is forward diffusion? What is reverse diffusion?

**Answer:**



---

**Q2.** Why is it useful to train a model to reverse the noise process?

**Answer:**



---

[Continue for all 25 questions...]
```

**Include:**
- All 25 questions (5 per module)
- Space for written answers
- Hints where appropriate (italicized)
- Reflection section at the end (optional)
- Submission checklist

---

## 🔧 Technical Implementation Notes

### **Colab vs Local File System Handling**

**Current State:**
- Modules 0-3 assume files are available in current session
- Module 2 → Module 3 transition relies on model in memory

**Recommended Enhancement for Module 4:**
Add environment detection at the beginning:

```python
import os

# Detect environment
IN_COLAB = 'google.colab' in str(get_ipython())

if IN_COLAB:
    print("Running in Google Colab")
    # Check if running from Google Drive
    if os.path.exists('/content/drive'):
        print("Google Drive mounted")
        BASE_PATH = '/content/drive/MyDrive/lab_8_modules'
    else:
        print("Using Colab temporary storage")
        BASE_PATH = '/content/lab_8_modules'
else:
    print("Running locally")
    BASE_PATH = os.getcwd()
```

### **Model Persistence Options**

**Option A: Keep Current Design (Recommended)**
- Modules are self-contained
- Module 3 offers quick training if needed
- Students can run modules independently
- **Pros:** Flexibility, no file management complexity
- **Cons:** Re-training if session resets

**Option B: Add Model Saving/Loading**
```python
# At end of Module 2
model.save('toy_denoiser.h5')
print("Model saved! You can load it in Module 3.")

# At start of Module 3
try:
    model = keras.models.load_model('toy_denoiser.h5')
    print("Model loaded from Module 2!")
except:
    print("Model not found, running quick training...")
```

**Option C: Provide Pre-Trained Weights**
- Upload pre-trained weights to Google Drive or GitHub
- Students download and load
- Backup if training fails
- **Implementation:** Add cell in Module 3:
```python
# Option to load pre-trained weights
USE_PRETRAINED = False  # Set to True to skip training

if USE_PRETRAINED:
    !wget https://example.com/toy_denoiser.h5
    model = keras.models.load_model('toy_denoiser.h5')
```

**Recommendation:** Keep Option A (current design) for Module 3. It's most robust for Colab environment.

---

## 📊 Module Statistics

| Module | File | Status | Time | Questions | Code Cells | Key Viz |
|--------|------|--------|------|-----------|------------|---------|
| 0 | `lab_8_module_0_what_is_diffusion.ipynb` | ✅ Done | 5-8 min | Q1-Q4 | 0 | 0 (markdown only) |
| 1 | `lab_8_module_1_forward_diffusion.ipynb` | ✅ Done | 8-12 min | Q5-Q9 | 6 | 4 |
| 2 | `lab_8_module_2_toy_denoiser.ipynb` | ✅ Done | 18-22 min | Q10-Q15 | 11 | 5 |
| 3 | `lab_8_module_3_iterative_denoising.ipynb` | ✅ Done | 12-15 min | Q16-Q20 | 8 | 3 |
| 4 | `lab_8_module_4_pretrained_sampling.ipynb` | ⏳ TODO | 12-15 min | Q21-Q25 | ~6 | ~3 |

**Total Time:** 55-72 minutes (target: 60-75 minutes) ✅
**Total Questions:** 25 (5 per module) ✅

---

## 🎨 Design Patterns Established

### **Pedagogical Framework**

1. **Three-Tier Math Presentation** (CRITICAL - use in Module 4!)
   - Symbols: Mathematical formula with Greek letters
   - Plain English: One-sentence explanation
   - Pseudocode: Algorithm in code-like syntax
   - **Example from Module 1:**
     ```
     Symbols: x_t = √(α̅_t) × x_0 + √(1 - α̅_t) × ε
     English: "Noisy image = weighted mix of original + random noise"
     Pseudocode: noisy = (signal_weight * original) + (noise_weight * noise)
     ```

2. **Predict-First Questions**
   - Q10: Predict training time BEFORE training
   - Q16: Predict when digit becomes recognizable BEFORE seeing trajectory
   - Q20 (suggested): Predict CIFAR-10 quality BEFORE running Module 4

3. **Connection Tables**
   - Always compare to previous labs (especially Lab 7)
   - Show similarities and differences
   - Help students build mental models

4. **Authentic Limitations**
   - Don't hide imperfections in toy model
   - Use limitations as teaching moments
   - Bridge to professional models explicitly

5. **Visual Hierarchy**
   - Color-coded borders (green/orange/red) for progress indicators
   - Consistent visualization style (matplotlib, no seaborn/plotly)
   - Clear titles and annotations

### **Code Style**

- **Functions are documented:** Docstrings with Args and Returns
- **Progress indicators:** Print statements showing what's happening
- **Checkmarks and emojis:** ✅ for success, ⚠️ for warnings, 💡 for tips
- **Error handling:** Try/except where needed (e.g., model availability check)
- **Reproducibility:** Set random seeds (np.random.seed(42), tf.random.set_seed(42))

### **Markdown Style**

- **Headers:** Clear hierarchy (##, ###, ####)
- **Bold for key terms:** **forward diffusion**, **U-Net**, **DDPM**
- **Code blocks:** Use triple backticks with syntax highlighting
- **Horizontal rules:** `---` to separate major sections
- **Blockquotes:** For key insights and takeaways

---

## 🔗 Cross-References

### **Connections to Previous Labs**

**Lab 2 (Gradient Descent):**
- Training process is identical (gradient descent + backprop)
- Loss curves show convergence
- Referenced in Module 2

**Lab 4 (Real-World ML):**
- Similar training workflow
- Comparison table in Module 2
- Stochastic ML concept (multiple runs, variability)

**Lab 6 (Saliency Maps):**
- Saliency shows WHERE model looks
- Feature maps show WHAT model extracts
- Diffusion shows HOW model creates

**Lab 7 (CNNs):**
- **Critical connection** - most important!
- CNNs analyze (image → class)
- Diffusion synthesizes (noise → image)
- Both use convolution
- U-Net = encoder (like CNN) + decoder
- Comparison table appears in Modules 0, 2, and 3

### **Within Lab 8**

- Module 0 sets conceptual foundation for all
- Module 1 implements forward process used in Module 2 training
- Module 2 trains model used in Module 3 generation
- Module 3 shows limitations that Module 4 addresses

---

## ⚠️ Common Pitfalls to Avoid

1. **Don't assume model persistence between sessions**
   - Colab sessions reset
   - Provide self-contained solutions (quick training option)

2. **Don't make math too complex**
   - This is for first-year students with NO math background
   - Three-tier presentation is mandatory
   - Use analogies liberally

3. **Don't skip the "why"**
   - Explain why we predict noise (not images directly)
   - Explain why we add noise back during generation
   - Explain why more steps help quality

4. **Don't hide imperfections**
   - Toy model results should be blurry/imperfect
   - This teaches students about scale and complexity
   - Leads naturally to Module 4 comparison

5. **Don't forget GPU reminders**
   - Remind students to enable T4 GPU in Colab
   - Print GPU availability status
   - Show timing comparisons (with/without GPU)

6. **Don't overload with hyperparameters**
   - Keep it simple: one user-configurable parameter (NUM_CLASSES in Module 2)
   - Other params should have good defaults with brief explanations

---

## 📚 File Locations

```
lab_8_modules/
├── lab_8_module_0_what_is_diffusion.ipynb          ✅ COMPLETE
├── lab_8_module_1_forward_diffusion.ipynb          ✅ COMPLETE
├── lab_8_module_2_toy_denoiser.ipynb               ✅ COMPLETE
├── lab_8_module_3_iterative_denoising.ipynb        ✅ COMPLETE
├── lab_8_module_4_pretrained_sampling.ipynb        ⏳ TODO
├── Lab_8_Student_Handout.md                         ⏳ TODO
├── Lab_8_Answer_Sheet.md                            ⏳ TODO
├── IMPLEMENTATION_GUIDE.md                          ✅ YOU ARE HERE
└── README.md                                        💡 Consider adding
```

---

## 🎯 Next Steps for Implementation

### **Priority 1: Module 4 Notebook**

1. **Setup section:**
   ```python
   # Install diffusers
   !pip install -q diffusers transformers accelerate

   # Imports
   import torch
   from diffusers import DDPMPipeline
   import matplotlib.pyplot as plt
   import numpy as np

   # Check GPU
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")
   ```

2. **Load model section:**
   - Explain: "PyTorch is an alternative to TensorFlow"
   - Download `google/ddpm-cifar10-32` (~200 MB)
   - Show model summary/info

3. **Generate images section:**
   - Generate 16 images (4×4 grid)
   - Show variety
   - Note: unconditional (no text prompts)

4. **Step count comparison:**
   - 25 steps vs 50 steps vs 100 steps
   - Side-by-side comparison
   - Discuss speed vs quality trade-off

5. **Comparison to Module 3:**
   - Show Module 3 toy model output next to Module 4 output
   - Dramatic quality difference
   - Explain why (scale, data, training time)

6. **Bridge to DALL-E:**
   - Table: CIFAR-10 unconditional vs Stable Diffusion text-conditional
   - Explain text conditioning mechanism (conceptually)
   - Show example text-to-image workflow diagram
   - Mention: Stable Diffusion = same algorithm + text encoder

7. **Questions Q21-Q25:**
   - Q21: How many object categories can you identify?
   - Q22: Quality difference at different step counts?
   - Q23: How does this compare to toy model?
   - Q24: How would DALL-E differ from CIFAR-10 model?
   - Q25: **Synthesis question:** Explain complete diffusion pipeline in own words

### **Priority 2: Student Handout**

1. Use `Lab_7_Student_Handout.md` as direct template
2. Follow exact same structure and formatting
3. Include all 5 modules
4. Add "From MNIST Digits to DALL-E" section
5. Connection to Real-World AI section with applications table
6. Before You Submit checklist
7. Optional: Additional Resources section

### **Priority 3: Answer Sheet**

1. Use `Lab_7_Answer_Sheet.md` as direct template
2. List all 25 questions
3. Provide answer spaces
4. Add hints where helpful (italicized)
5. Include submission checklist

---

## 💡 Quality Checklist

Before considering Lab 8 complete, verify:

### **Content Quality**
- [ ] All math uses three-tier presentation (symbols, English, pseudocode)
- [ ] Every module connects to previous labs (especially Lab 7)
- [ ] Analogies are consistent across modules
- [ ] Questions follow predict-first pattern where appropriate
- [ ] Code has clear comments and docstrings
- [ ] Visualizations are consistent style (matplotlib, no gradio)

### **Technical Quality**
- [ ] All notebooks run in Colab without errors
- [ ] GPU usage is checked and reported
- [ ] Training completes in stated time (2-3 min for Module 2)
- [ ] Generation works correctly (Module 3 and 4)
- [ ] Random seeds are set for reproducibility
- [ ] Model availability is handled gracefully (Module 3)

### **Educational Quality**
- [ ] Total time is 60-75 minutes
- [ ] Questions total 25 (5 per module)
- [ ] Difficulty progression feels natural
- [ ] Authentic limitations are acknowledged
- [ ] Real-world applications are clear
- [ ] Students understand WHY not just HOW

### **Documentation Quality**
- [ ] Student Handout is complete and clear
- [ ] Answer Sheet has all 25 questions
- [ ] File names follow convention
- [ ] README exists (optional but helpful)
- [ ] This IMPLEMENTATION_GUIDE is updated if changes made

---

## 🚀 Estimated Time to Complete Remaining Work

- **Module 4 notebook:** 2-3 hours (includes testing, visualization tuning)
- **Student Handout:** 1-2 hours (adapt from Lab 7 template)
- **Answer Sheet:** 30 minutes (straightforward formatting)
- **Testing/Polish:** 1 hour (run all modules end-to-end)

**Total:** ~5-7 hours of focused work

---

## 📝 Notes for Future Developer

**You're inheriting a well-designed lab!** The foundation is solid:

1. **Modules 0-3 are complete and tested** - follow their patterns
2. **Pedagogical framework is established** - don't deviate unnecessarily
3. **Three-tier math presentation is mandatory** - students need it
4. **Connection to Lab 7 is critical** - emphasize throughout
5. **Authentic limitations are features, not bugs** - embrace them

**Module 4 is the victory lap:** Show students the power of scale. The toy model taught them the algorithm; Module 4 shows them what's possible with proper resources.

**The Student Handout ties it all together:** This is where students see the full narrative. Take time to make connections explicit.

**You got this!** The hard part (algorithm implementation) is done. Now show students why it matters. 🎨

---

## 🔍 References

- **Plan document:** `C:\Users\jwallin\.claude\plans\gleaming-herding-bentley.md`
- **Lab 7 templates:** `lab_7_modules/Lab_7_Student_Handout.md`, `Lab_7_Answer_Sheet.md`
- **Lab 7 training notebook:** `lab_7_modules/lab_7_module_4_mnist_training.ipynb`
- **Diffusers docs:** https://huggingface.co/docs/diffusers/
- **DDPM paper:** "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

---

**Good luck and happy implementing!** 🚀

*Last updated: December 2, 2025*
*Modules 0-3 complete, Module 4 + documentation remaining*
