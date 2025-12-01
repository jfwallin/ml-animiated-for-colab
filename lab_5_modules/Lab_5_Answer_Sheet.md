# Lab 5: Embeddings and the Geometry of Meaning
## Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Student Name:** _________________________________

**Date:** _________________________________

---

## Module 0: Introduction to Embeddings

### Q1. Before running the code, predict: Will "cats" and "dogs" have similar embedding vectors? Why or why not?

**Answer:**

<br><br><br>

### Q2. Looking at the cosine similarities in the output, which word pair is most similar? Does this match your intuition?

**Answer:**

<br><br><br>

### Q3. Why do "cats" and "galaxies" have a similarity of 0.000? What does this tell you about their co-occurrence in the corpus?

**Answer:**

<br><br><br>

### Q4. In the PCA visualization, which words cluster together? Why do you think they form these groups?

**Answer:**

<br><br><br>

### Q5. The original embedding space has 92 dimensions (one per word). PCA reduces this to 2D. What information might be lost in this reduction?

**Answer:**

<br><br><br><br>

### Q6. How is this co-occurrence approach similar to what you learned about hidden layers in Lab 4? (Hint: both create new representations)

**Answer:**

<br><br><br><br>

---

## Module 1: Word Embeddings & Vector Arithmetic

### Q7. How many dimensions does each GloVe word vector have? How does this compare to Module 0's co-occurrence embeddings?

**Answer:**

<br><br><br>

### Q8. Looking at the dimension plots for "galaxy", "person", "table", and "atom", what do you notice about parameter 30?

**Answer:**

<br><br><br>

### Q9. Why do science words have lower values at parameter 33 compared to non-science words? What does this dimension seem to capture?

**Answer:**

<br><br><br><br>

### Q10. Predict: What word should complete this analogy: **paris - france + italy = ?**

**Prediction:**

<br>

**Actual result after running code:**

<br><br>

### Q11. After running the analogy code, was your prediction correct? What was the top result?

**Answer:**

<br><br><br>

### Q12. Try your own analogy in the interactive section. Record your input (A, B, C) and the top result. Did it work as expected?

**My analogy:**
- A = _______________
- B = _______________
- C = _______________

**Top result:** _______________

**Did it work as expected?**

<br><br>

### Q13. Which type of analogy works best in this model: capital-country, comparative adjectives, verb tenses, or family relationships?

**Answer:**

<br><br><br>

### Q14. Why do you think some analogies (like animal → sound) don't work well in word embeddings?

**Answer:**

<br><br><br><br>

### Q15. How does vector arithmetic (like king - man + woman = queen) demonstrate that embeddings capture meaning relationships rather than just word similarity?

**Answer:**

<br><br><br><br>

---

## Module 2: Sentence Embeddings & Semantic Search

### Q16. Before seeing the PCA plot, predict: Which two sentences from the list of 5 will be closest together? Why?

**Prediction:**

<br><br>

**After viewing the plot, was your prediction correct?**

<br>

### Q17. After viewing the PCA visualization, which sentences clustered together? Was your prediction from Q16 correct?

**Answer:**

<br><br><br>

### Q18. Why is "Neural networks learn patterns" isolated from the other sentences in the PCA plot?

**Answer:**

<br><br><br>

### Q19. Use the dropdown tool to compare "The cat sat on the mat." with "Cats are great pets." What is their cosine similarity? Are they similar or different?

**Cosine similarity:** _______________

**Are they similar or different?**

<br><br>

### Q20. Now compare "Stars fuse hydrogen into helium." with "Galaxies contain billions of stars." What is their cosine similarity? Is it higher or lower than the cat sentences? Why?

**Cosine similarity:** _______________

**Higher or lower than cat sentences?**

<br>

**Why?**

<br><br><br>

### Q21. In the semantic search activity, what query did you test? What were the top 3 results?

**My query:** _______________________________________________

**Top 3 results:**

1. _________________________________________________________________

2. _________________________________________________________________

3. _________________________________________________________________

<br>

### Q22. Do the top 3 results make sense for your query? Would keyword search have found these sentences?

**Answer:**

<br><br><br><br>

### Q23. How is semantic search different from searching with Ctrl+F (keyword search)? Give a specific example from your results.

**Answer:**

<br><br><br><br>

### Q24. Reflecting on all three modules: How do embeddings allow AI systems like ChatGPT to "understand" the meaning of text rather than just matching keywords?

**Answer:**

<br><br><br><br><br>

---

## Reflection (Optional)

### What was the most surprising thing you learned about embeddings?

**Answer:**

<br><br><br><br>

### How might embeddings be used in real-world applications like RAG systems or search engines?

**Answer:**

<br><br><br><br>

### What connections do you see between embeddings and the hidden layer concepts from Lab 4?

**Answer:**

<br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 3 core modules (0, 1, 2)
- [ ] Answered all 24 questions (Q1-Q24)
- [ ] Included specific values where requested (cosine similarities, query results)
- [ ] Recorded your custom analogy attempt (Q12)
- [ ] Recorded your semantic search query and results (Q21)
- [ ] Ran all code cells and viewed visualizations
- [ ] Tried prediction questions BEFORE running code
- [ ] Written thoughtful, complete answers
- [ ] (Optional) Completed reflection questions
- [ ] (Optional) Explored Module 3 RAG activity if assigned

---

## Submission Instructions

Submit this completed answer sheet according to your instructor's guidelines (Canvas upload, hardcopy, PDF, etc.).

**Congratulations on completing Lab 5!** You now understand how AI systems represent meaning as geometry—the foundation of modern natural language processing!
