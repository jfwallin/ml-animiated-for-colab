# Lab 7: Convolutions — How Computers See Images

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In today's lab, you will explore **convolutions and convolutional neural networks (CNNs)**—the fundamental architecture behind computer vision systems. You'll learn to:

- Understand what convolution operations are and why they work for images
- Apply classic image filters (blur, sharpen, edge detection) to real photos
- Visualize feature maps from a pretrained CNN to see what it detects
- Understand hierarchical feature extraction (edges → textures → shapes → objects)
- Train your own CNN from scratch on handwritten digit recognition
- Connect convolutions to previous labs (saliency, hidden layers, embeddings)

### Lab Structure

This lab consists of **5 modules** that you'll complete in order:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | What Is a Convolution? | ~10 min | Conceptual + code |
| 1 | Applying Filters to Real Images | ~15 min | Colab |
| 2 | Visualizing CNN Feature Maps | ~20 min | Colab |
| 3 | Hierarchical Feature Extraction | ~10 min | Conceptual |
| 4 | Training a CNN on MNIST | ~15-20 min | Colab |

**Total Time:** ~70-75 minutes

### Working in Groups

- Work in **small groups** (2-4 people)
- One group member runs Colab and shares their screen
- Everyone participates in discussion and predictions
- All group members can use their own notebooks

### Key Concepts

> **CONVOLUTION**: A sliding window operation that applies a small filter across an image, detecting patterns through multiply-and-add operations.
>
> **FILTER/KERNEL**: A small matrix (typically 3×3) that detects specific patterns like edges, corners, or textures when convolved with an image.
>
> **FEATURE MAP**: The output of a convolutional layer, showing where in the image a particular pattern was detected.
>
> **HIERARCHICAL LEARNING**: Building complex patterns from simple ones—edges combine into shapes, shapes combine into objects.
>
> **TRANSLATION INVARIANCE**: The same filter detects the same pattern anywhere in the image, making CNNs efficient and robust.

### Connection to Previous Labs

**Lab 3:** You learned about activation functions
- **Lab 7:** CNNs use ReLU after convolution to add nonlinearity

**Lab 4:** You learned that hidden layers create new representations
- **Lab 7:** Convolutional layers are spatially-structured hidden layers

**Lab 5:** You learned how embeddings encode meaning as vectors
- **Lab 7:** CNN final layers create image embeddings

**Lab 6:** You learned saliency shows **WHERE** a model looks
- **Lab 7:** Feature maps show **WHAT** a model extracts

---

## Module 0: What Is a Convolution?

**Time:** ~10 minutes
**Type:** Conceptual + hands-on code

### Learning Objectives
- Understand convolution as sliding window + multiply-and-add
- Recognize why convolution works for images (local patterns, translation invariance, parameter sharing)
- Apply convolution to tiny arrays with different filters
- Connect to Lab 4's hidden layers and Lab 6's saliency

### What You'll Do

1. Learn the intuition: Convolution = Pattern matching with a sliding stamp
2. See the math (simple version, no calculus!)
3. Explore common filters: edge detectors, blur, sharpen, identity
4. Run hands-on demo with a 5×5 image and 3×3 filters
5. Understand why convolution works (local patterns, translation invariance, efficiency)

### Key Insight

> **Convolution is the fundamental operation that lets computers "see" patterns in images. It's efficient, powerful, and the basis for all modern computer vision systems.**

### Questions

**Q1.** In your own words, what does a convolution operation do?

**Q2.** Look at the vertical edge filter `[[-1,0,1],[-1,0,1],[-1,0,1]]`. Why would this highlight vertical edges?

**Q3.** What happens when you convolve an image with the identity filter `[[0,0,0],[0,1,0],[0,0,0]]`?

**Q4.** How is convolution different from the dimension-lifting you saw in Lab 4 Module 0? What's similar?

---

## Module 1: Applying Filters to Real Images

**Time:** ~15 minutes
**Type:** Colab notebook

### Learning Objectives
- Apply classic computer vision filters to real photographs
- Visualize effects of blur, sharpen, and edge detection
- Design and test custom 3×3 filters
- Understand difference between hand-designed and learned filters
- Connect to Lab 3's activation functions

### What You'll Do

1. Load two sample images (edges demo and textures demo)
2. Apply 6 different filters:
   - Original (no filter, for comparison)
   - Blur (3×3 averaging)
   - Sharpen
   - Sobel vertical edge detector
   - Sobel horizontal edge detector
   - Laplacian (all-edges detector)
3. Observe how each filter transforms the images
4. Design your own custom 3×3 filter and see what it does
5. (Optional) Upload your own image and apply filters to it

### The Filters

**Blur:** Averages nearby pixels, reduces noise
```
[1  1  1]
[1  1  1]  ÷ 9
[1  1  1]
```

**Sharpen:** Emphasizes differences between pixels
```
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]
```

**Sobel Vertical:** Detects vertical edges
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

### Key Insight

> **CNNs automatically learn filters through training—starting with edge detectors in early layers (similar to Sobel) and building up to complex pattern detectors in deeper layers. No human design required!**

### Questions

**Q5.** After applying the blur filter, what changed about the image? Why might blur be useful in image processing?

**Q6.** What happened when you applied the Sobel vertical edge detector? Which parts of the image were highlighted?

**Q7.** Compare the sharpened image to the original. What features became more pronounced?

**Q8.** Design your own 3×3 filter in the interactive section. What effect did it have? Record your filter values and describe the result.

**Q9.** Why do you think edge detection is important for object recognition?

---

## Module 2: Visualizing CNN Feature Maps

**Time:** ~20 minutes
**Type:** Colab notebook

### Learning Objectives
- Load a pretrained CNN (MobileNetV2)
- Extract and visualize feature maps from different depths
- Observe hierarchical pattern detection (simple → complex)
- Compare early layers (edges) vs. deep layers (object parts)
- Connect to Lab 6's saliency maps

### What You'll Do

1. Load MobileNetV2 (14MB, recognizes 1000 object categories)
2. Create a feature extraction model that outputs intermediate layers
3. Extract feature maps from 3 layers:
   - Layer 1 (early): `block_1_conv1`
   - Layer 3 (middle): `block_3_conv1`
   - Layer 6 (deep): `block_6_conv1`
4. Visualize 16 feature maps from each layer
5. Observe the progression from simple to complex patterns
6. (Optional) Upload your own image and see what features the CNN extracts

### Expected Observations

**Layer 1 (Early):**
- Strong responses to edges (vertical, horizontal, diagonal)
- High contrast boundaries
- Simple patterns

**Layer 3 (Middle):**
- Corners and curves
- Textures (stripes, dots)
- Small shapes

**Layer 6 (Deep):**
- Complex patterns
- Object parts (if real photo: ears, wheels, windows)
- Contextual features

### Connection to Lab 6

**Lab 6 saliency** showed which pixels matter for classification.

**Lab 7 feature maps** show what patterns the network extracts from those pixels.

**Together:** They reveal how CNNs work!

### Questions

**Q10.** Before viewing the feature maps, predict: What will Layer 1 (early layer) detect?

**Q11.** Looking at Layer 1 feature maps, which filters activated strongly? What patterns did they detect?

**Q12.** Compare Layer 1 and Layer 6 feature maps. How are they different? What does this tell you about hierarchical learning?

**Q13.** Find a feature map in Layer 3 that activated strongly for one part of the image. What pattern was it detecting?

**Q14.** Why do deeper layers show more abstract/complex patterns than early layers?

**Q15.** How does this connect to the saliency maps from Lab 6? (Hint: Saliency shows importance; feature maps show what's extracted)

---

## Module 3: Hierarchical Feature Extraction

**Time:** ~10 minutes
**Type:** Conceptual (markdown only)

### Learning Objectives
- Understand the hierarchical structure of CNNs
- Learn why hierarchical learning works (compositionality, reusability, efficiency)
- Compare CNN layers to human visual cortex
- Connect to all previous labs (3, 4, 5, 6)
- Recognize real-world applications

### The Hierarchy

```
Input Image (Raw Pixels)
    ↓
Layer 1-2: Edges & Gradients
    ↓
Layer 3-4: Corners & Textures
    ↓
Layer 5-6: Object Parts
    ↓
Layer 7+: Whole Objects
    ↓
Output: Class Predictions
```

### Comparison to Human Vision

| Brain Region | CNN Layer | What It Detects |
|--------------|-----------|-----------------|
| V1 (Primary Visual Cortex) | Layer 1-2 | Edges, orientation |
| V2 (Secondary) | Layer 3-4 | Corners, textures |
| V4 (Intermediate) | Layer 5-6 | Shapes, object parts |
| IT (Inferotemporal) | Layer 7+ | Whole objects, faces |

### Why Hierarchical Learning Works

1. **Compositionality:** Complex features built from simple ones (edges → shapes → objects)
2. **Reusability:** Low-level features are reused across many objects
3. **Translation Invariance:** Same filter detects pattern anywhere in image
4. **Parameter Efficiency:** ~20,000x more efficient than fully-connected networks for images

### Questions

**Q16.** Why does hierarchical feature extraction make sense for object recognition?

**Q17.** How is a CNN's Layer 1 similar to the human visual cortex area V1?

**Q18.** What advantage does parameter sharing give CNNs?

**Q19.** Thinking back to Lab 4: How are convolutional layers similar to hidden layers? How are they different?

---

## Module 4: Training a CNN on MNIST

**Time:** ~15-20 minutes
**Type:** Colab notebook

### Learning Objectives
- Build a simple CNN architecture from scratch
- Train on MNIST handwritten digits (60,000 training images)
- Achieve >97% test accuracy in just 3 epochs (~2-3 minutes)
- Visualize learned filters (automatic edge detectors!)
- Analyze confusion matrix to see common mistakes
- Compare to Lab 4's dense network training

### What You'll Do

1. Load MNIST dataset (28×28 grayscale images, 10 classes)
2. Explore sample images from each digit (0-9)
3. Preprocess: Reshape and normalize
4. Build CNN architecture:
   ```
   Conv2D (32 filters) → MaxPool → Conv2D (64 filters) → MaxPool → Flatten → Dense (128) → Dense (10)
   ```
5. Compile with Adam optimizer and sparse categorical cross-entropy loss
6. Train for 3 epochs (~2-3 minutes)
7. Evaluate on test set
8. Visualize training curves (accuracy and loss)
9. Generate confusion matrix
10. Visualize learned filters from first conv layer
11. Examine misclassified examples

### Architecture Details

**Input:** 28×28×1 (grayscale images)

**Layer 1:** Conv2D (32 filters, 3×3) + ReLU → MaxPooling (2×2)

**Layer 2:** Conv2D (64 filters, 3×3) + ReLU → MaxPooling (2×2)

**Classifier:** Flatten → Dense (128, ReLU) → Dense (10, Softmax)

**Total parameters:** ~100,000 (compare to millions for fully-connected!)

### Connection to Lab 4

| Aspect | Lab 4 (Breast Cancer) | Lab 7 (MNIST) |
|--------|-----------------------|----------------|
| Data type | Tabular (30 features) | Images (28×28 pixels) |
| Architecture | Dense → Dense → Output | Conv → Pool → Conv → Pool → Dense → Output |
| Training method | Gradient descent | Same! |
| Optimizer | Adam | Adam |
| Loss | Binary cross-entropy | Sparse categorical cross-entropy |

### Key Insight

> **CNNs use the same training process as dense networks from Lab 4—gradient descent + backpropagation—just with a different architecture optimized for spatial data!**

### Questions

**Q20.** Before training, predict: What accuracy do you expect on MNIST? (10% random guessing, 50%, 90%, 99%?)

**Q21.** After training for 3 epochs, what test accuracy did you achieve? Was this higher or lower than your prediction?

**Q22.** Looking at the confusion matrix, which digits are most commonly confused with each other? Why might this be?

**Q23.** Examine the learned filters from the first convolutional layer. Do they look like edge detectors (similar to Sobel filters from Module 1)?

**Q24.** Compare this CNN training to the Breast Cancer classifier from Lab 4. What's similar? What's different?

**Q25.** Find 2-3 misclassified examples. Can you understand why the model got them wrong? (Are they ambiguous even to you?)

---

## Key Takeaways

By the end of this lab, you should understand:

- **Convolution** = sliding window + multiply-and-add operation that detects patterns
- **Filters** are small matrices (3×3) that detect specific patterns like edges, textures, shapes
- **Feature maps** show where in an image a particular pattern was detected
- **Hierarchical learning** builds complex features from simple ones (edges → textures → shapes → objects)
- **CNNs automatically learn filters** through gradient descent—no human design required
- **Parameter sharing** makes CNNs ~20,000x more efficient than dense networks for images
- **Training process** is identical to Lab 4 (gradient descent + backprop), just different architecture
- **Real-world CNNs** power face recognition, medical imaging, autonomous vehicles, and more

---

## Connection to Real-World AI

### Applications Where CNNs Excel

| Application | How CNNs Help |
|-------------|---------------|
| **Face Recognition** | Early layers: edges; Mid layers: facial features; Deep layers: unique face patterns |
| **Medical Imaging** | Detect tumors, lesions, and pathology patterns in X-rays, CT scans, MRIs |
| **Autonomous Vehicles** | Recognize lane markings, pedestrians, vehicles, traffic signs in real-time |
| **Content Moderation** | Detect unsafe or violating content in images and videos |
| **Agriculture** | Identify crop diseases, pests, and growth patterns from drone imagery |
| **Manufacturing** | Quality control—detect defects in products on assembly lines |

### The CNN Revolution (2012-Present)

**2012: ImageNet Competition**
- AlexNet (CNN) achieved 84.7% accuracy vs. 73.8% for traditional methods
- Sparked the "deep learning revolution"

**2015: Superhuman Performance**
- ResNet achieved 96.4% on ImageNet (surpassing human-level ~94%)

**Today: Everywhere**
- Your phone's camera (face detection, scene recognition)
- Social media (automatic photo tagging, content moderation)
- Healthcare (radiology AI assistants)
- Autonomous vehicles (Tesla, Waymo, Cruise)

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (0, 1, 2, 3, 4)
- [ ] Run all code cells and viewed visualizations
- [ ] Answered all 25 questions in the answer sheet
- [ ] Tried custom filter design in Module 1
- [ ] Made predictions before viewing results (Q10, Q20)
- [ ] Uploaded your own image in Module 2 (optional but recommended)
- [ ] Understood the hierarchical structure (edges → textures → shapes → objects)
- [ ] Seen your trained CNN achieve >97% accuracy on MNIST
- [ ] Examined confusion matrix and misclassified examples
- [ ] Connected concepts to Labs 3, 4, 5, and 6

---

## Submission Instructions

Submit your completed answer sheet according to your instructor's guidelines (Canvas upload, PDF, etc.).

**Congratulations!** You've completed Lab 7 and now understand how CNNs enable computers to "see" images—the foundation of modern computer vision!

---

## Bridge to Lab 8: From Analysis to Synthesis

### What You've Learned in Lab 7:
- CNNs **extract features** from images
- **Analysis task:** Given an image, identify what's in it
- Hierarchical architecture: edges → textures → shapes → objects

### What's Coming in Lab 8: Diffusion Models
- **Synthesis task:** Given a description (or noise), **generate** an image
- Diffusion models **create images from noise**
- Powers DALL-E, Stable Diffusion, Midjourney
- Similar architecture (U-Net uses convolution!), opposite direction

### The Connection:
```
Lab 7 (CNNs):      Image → Features → Classification
                   (Analysis: "What is this?")

Lab 8 (Diffusion): Noise → Features → Image
                   (Synthesis: "Create this!")
```

**Both use convolutional architectures—one for understanding, one for creating!**

---

## Additional Resources

### Further Reading

- **Original CNN Paper:** LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"
- **AlexNet Paper:** Krizhevsky et al. (2012) - "ImageNet Classification with Deep Convolutional Neural Networks"
- **Interactive CNN Visualizer:** https://poloclub.github.io/cnn-explainer/

### Tools and Libraries

- **TensorFlow/Keras:** Python deep learning framework (used in this lab)
- **PyTorch:** Alternative deep learning framework (popular in research)
- **OpenCV:** Computer vision library for image processing

### Datasets to Explore

- **ImageNet:** 1.4M images, 1000 categories (used to train most CNNs)
- **CIFAR-10/100:** 60,000 tiny images, 10 or 100 categories
- **MS COCO:** Images with detailed annotations for object detection

---

**Questions or feedback?** Reach out to your instructor or TA!
