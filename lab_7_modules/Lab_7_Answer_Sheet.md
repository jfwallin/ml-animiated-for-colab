# Lab 7: Convolutions — How Computers See Images
## Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Student Name:** _________________________________

**Date:** _________________________________

---

## Module 0: What Is a Convolution?

### Q1. In your own words, what does a convolution operation do?

*(Hint: Think about the sliding window + multiply-and-add operation)*

**Answer:**

<br><br><br>

### Q2. Look at the vertical edge filter `[[-1,0,1],[-1,0,1],[-1,0,1]]`. Why would this highlight vertical edges?

*(Hint: What happens when you multiply this filter with a patch that has dark pixels on the left and bright pixels on the right?)*

**Answer:**

<br><br><br>

### Q3. What happens when you convolve an image with the identity filter `[[0,0,0],[0,1,0],[0,0,0]]`?

*(Hint: Look at the visualization in Module 0. Why does the identity filter preserve the image?)*

**Answer:**

<br><br><br>

### Q4. How is convolution different from the dimension-lifting you saw in Lab 4 Module 0? What's similar?

*(Hint: Both create new representations. But one looks at the whole input, while the other looks at local patches.)*

**Answer:**

**Different:**

<br><br>

**Similar:**

<br><br><br>

---

## Module 1: Applying Filters to Real Images

### Q5. After applying the blur filter, what changed about the image? Why might blur be useful in image processing?

*(Hint: Look at the edges—did they become sharper or softer? Think about noise reduction.)*

**Answer:**

<br><br><br>

### Q6. What happened when you applied the Sobel vertical edge detector? Which parts of the image were highlighted?

*(Hint: Compare vertical vs. horizontal edges. Why does the vertical edge detector respond strongly to vertical stripes but not horizontal ones?)*

**Answer:**

<br><br><br>

### Q7. Compare the sharpened image to the original. What features became more pronounced?

*(Hint: Look at boundaries and transitions. How does sharpen differ from edge detection?)*

**Answer:**

<br><br><br>

### Q8. Design your own 3×3 filter in the interactive section. What effect did it have? Record your filter values and describe the result.

**Your custom filter:**
```
[ __ __ __ ]
[ __ __ __ ]
[ __ __ __ ]
```

**Effect observed:**

<br><br><br>

### Q9. Why do you think edge detection is important for object recognition?

*(Hint: Think about what defines shapes. How do edges relate to object boundaries?)*

**Answer:**

<br><br><br>

---

## Module 2: Visualizing CNN Feature Maps

### Q10. Before viewing the feature maps, predict: What will Layer 1 (early layer) detect?

*(Make your prediction BEFORE continuing to the next cell!)*

**Prediction:**

<br><br><br>

### Q11. Looking at Layer 1 feature maps, which filters activated strongly? What patterns did they detect?

*(Hint: Look for bright (yellow/white) regions. Which filters show strong activation? Do they correspond to edges, corners, or other patterns?)*

**Answer:**

<br><br><br>

### Q12. Compare Layer 1 and Layer 6 feature maps. How are they different? What does this tell you about hierarchical learning?

*(Hint: Compare the patterns you see. Are Layer 1 patterns simple or complex? What about Layer 6?)*

**Answer:**

**Layer 1:**

<br><br>

**Layer 6:**

<br><br>

**What this tells us about hierarchical learning:**

<br><br><br>

### Q13. Find a feature map in Layer 3 that activated strongly for one part of the image. What pattern was it detecting?

*(Hint: Look for filters that "light up" on specific features. What visual pattern caused that activation?)*

**Answer:**

<br><br><br>

### Q14. Why do deeper layers show more abstract/complex patterns than early layers?

*(Hint: Each layer builds on the previous one. If Layer 1 detects edges, Layer 3 can combine edges into shapes. What can Layer 6 do with Layer 3's shapes?)*

**Answer:**

<br><br><br>

### Q15. How does this connect to the saliency maps from Lab 6?

*(Hint: Saliency shows importance; feature maps show what's extracted. How do they work together?)*

**Answer:**

<br><br><br><br>

---

## Module 3: Hierarchical Feature Extraction

### Q16. Why does hierarchical feature extraction make sense for object recognition?

*(Hint: Objects are made of parts, parts are made of shapes, shapes are made of edges. How does this match the CNN hierarchy?)*

**Answer:**

<br><br><br><br>

### Q17. How is a CNN's Layer 1 similar to the human visual cortex area V1?

*(Hint: Look at the comparison table. What do both detect? Why would evolution and machine learning arrive at similar solutions?)*

**Answer:**

<br><br><br>

### Q18. What advantage does parameter sharing give CNNs?

*(Hint: Compare the number of parameters for fully-connected vs. convolutional layers. What does this enable?)*

**Answer:**

<br><br><br><br>

### Q19. Thinking back to Lab 4: How are convolutional layers similar to hidden layers? How are they different?

*(Hint: Both create new representations. But one looks at the entire input globally, while the other focuses on local spatial patterns.)*

**Answer:**

**Similar:**

<br><br>

**Different:**

<br><br><br>

---

## Module 4: Training a CNN on MNIST

### Q20. Before training, predict: What accuracy do you expect on MNIST?

*(Consider: 10% random guessing, 50%, 90%, 99%?)*

**My prediction:** ______%

**Reasoning:**

<br><br><br>

### Q21. After training for 3 epochs, what test accuracy did you achieve? Was this higher or lower than your prediction?

**Test accuracy achieved:** ______%

**Higher or lower than prediction?**

<br>

**Were you surprised? Why or why not?**

<br><br><br>

### Q22. Looking at the confusion matrix, which digits are most commonly confused with each other? Why might this be?

*(Hint: Look at the off-diagonal cells with the highest values. Do the confused digits look similar?)*

**Most common confusions:**

1. Digit __ confused with digit __
2. Digit __ confused with digit __
3. Digit __ confused with digit __

**Why this might happen:**

<br><br><br>

### Q23. Examine the learned filters from the first convolutional layer. Do they look like edge detectors (similar to Sobel filters from Module 1)?

**Answer:**

<br><br><br>

### Q24. Compare this CNN training to the Breast Cancer classifier from Lab 4. What's similar? What's different?

*(Think about: training process, architecture, data type, accuracy, time to train)*

**Similar:**

<br><br><br>

**Different:**

<br><br><br>

### Q25. Find 2-3 misclassified examples. Can you understand why the model got them wrong? Are they ambiguous even to you?

**Example 1:**
- True label: __
- Predicted label: __
- Why the model made the mistake:

<br><br>

**Example 2:**
- True label: __
- Predicted label: __
- Why the model made the mistake:

<br><br>

**Example 3:**
- True label: __
- Predicted label: __
- Why the model made the mistake:

<br><br>

**Were they ambiguous to you as well?**

<br><br>

---

## Reflection (Optional)

### What was the most surprising thing you learned about convolutions and CNNs?

**Answer:**

<br><br><br><br>

### How does understanding CNNs change your perspective on computer vision and AI systems?

**Answer:**

<br><br><br><br>

### What connections do you see between Lab 7 (convolutions) and previous labs (Labs 3, 4, 5, 6)?

**Answer:**

<br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (0, 1, 2, 3, 4)
- [ ] Answered all 25 questions (Q1-Q25)
- [ ] Included specific observations where requested
- [ ] Recorded your custom filter experiment (Q8)
- [ ] Made predictions BEFORE viewing results (Q10, Q20)
- [ ] Recorded confusion matrix observations (Q22)
- [ ] Analyzed misclassified examples (Q25)
- [ ] Run all code cells and viewed visualizations
- [ ] Written thoughtful, complete answers
- [ ] (Optional) Completed reflection questions
- [ ] Connected concepts to previous labs where relevant

---

## Submission Instructions

Submit this completed answer sheet according to your instructor's guidelines (Canvas upload, hardcopy, PDF, etc.).

**Congratulations on completing Lab 7!** You now understand how **CNNs enable computers to see**—the foundation of modern computer vision systems from face recognition to autonomous vehicles!

---

## Next Up: Lab 8 - Diffusion Models

In Lab 8, you'll learn how to **generate images from text descriptions** using diffusion models—the technology behind DALL-E, Stable Diffusion, and Midjourney. You'll see how the convolutional architectures you learned in Lab 7 are used in reverse to create images rather than analyze them!
