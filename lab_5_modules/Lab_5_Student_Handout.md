# Lab 5: Embeddings and the Geometry of Meaning

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In today's lab, you will explore how AI systems represent meaning as geometry. You'll build embeddings from scratch, explore professional pre-trained models, and create a semantic search engine:

- **Build a tiny embedding system** from 27 sentences to understand the core principles
- **Explore professional GloVe embeddings** trained on 6 billion words
- **Use vector arithmetic** to solve analogies (king - man + woman ≈ queen)
- **Embed entire sentences** and measure semantic similarity
- **Build a semantic search engine** that finds documents by meaning, not keywords

### Lab Structure

This lab consists of **3 core modules** that you'll complete in order:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | Introduction to Embeddings | ~20 min | Colab |
| 1 | Word Embeddings & Vector Arithmetic | ~25 min | Colab |
| 2 | Sentence Embeddings & Semantic Search | ~25 min | Colab |

**Total Time:** ~70 minutes

**Optional Extension:**
| Module | Title | Time | Type |
|--------|-------|------|------|
| 3 (Optional) | RAG in Practice | ~20-30 min | lmnotebook |

*Note: Module 3 may be assigned separately as an external assignment.*

### Working in Groups

- Work in **small groups** (2-4 people)
- One group member runs Colab and shares their screen
- Everyone participates in discussion and predictions
- All group members can use their own notebooks

### Key Concepts

> **EMBEDDINGS**: Numerical representations of words or sentences as vectors in a high-dimensional space, where similar meanings point in similar directions.
>
> **COSINE SIMILARITY**: A measure of similarity between two vectors, ranging from 0 (unrelated) to 1 (identical meaning), based on the angle between them.
>
> **PCA (Principal Component Analysis)**: A dimensionality reduction technique that projects high-dimensional data onto 2D or 3D for visualization.
>
> **SEMANTIC SEARCH**: Finding information by meaning rather than keyword matching. The foundation of modern search engines and RAG systems.
>
> **RAG (Retrieval-Augmented Generation)**: Systems that use semantic search to find relevant context before generating responses. Used by ChatGPT, Claude, and other AI assistants.

### Connection to Previous Labs

**Lab 1:** You learned about models, parameters, and optimization basics

**Lab 2:** You learned about gradient descent for automatic parameter updates

**Lab 3:** You learned about activation functions and how they transform space

**Lab 4:** You learned that hidden layers create new representations that make problems solvable

**Lab 5:** You'll learn that **embeddings are the representation space where meaning lives** — the same idea as hidden layers, but for language!

---

## Module 0: Introduction to Embeddings

**Time:** ~20 minutes
**Type:** Colab notebook

### Learning Objectives

- Understand how words become vectors based on co-occurrence patterns
- Build a simple embedding system from scratch
- Visualize word clusters using PCA
- Connect embeddings to hidden layer concepts from Lab 4
- Recognize that embeddings encode meaning as measurable geometry

### What You'll Do

1. Learn how embedding models capture meaning from word co-occurrence
2. Build a tiny embedding system from 27 sentences
3. Compute cosine similarity between word pairs
4. Visualize the embedding space with PCA
5. Observe how related words cluster together

### Key Insight

> **Words that appear in similar contexts get similar vector representations.**
>
> "Cats" and "dogs" both appear near "pets" → similar vectors
> "Stars" and "galaxies" both appear in astronomy contexts → similar vectors
> "Cats" and "galaxies" never appear together → completely different vectors

This is the same core idea behind word2vec, GloVe, BERT, and GPT!

### Questions

**Q1.** Before running the code, predict: Will "cats" and "dogs" have similar embedding vectors? Why or why not?

**Q2.** Looking at the cosine similarities in the output, which word pair is most similar? Does this match your intuition?

**Q3.** Why do "cats" and "galaxies" have a similarity of 0.000? What does this tell you about their co-occurrence in the corpus?

**Q4.** In the PCA visualization, which words cluster together? Why do you think they form these groups?

**Q5.** The original embedding space has 92 dimensions (one per word). PCA reduces this to 2D. What information might be lost in this reduction?

**Q6.** How is this co-occurrence approach similar to what you learned about hidden layers in Lab 4? (Hint: both create new representations)

---

## Module 1: Word Embeddings & Vector Arithmetic

**Time:** ~25 minutes
**Type:** Colab notebook

### Learning Objectives

- Explore professional GloVe embeddings trained on 6 billion words
- Understand that individual dimensions capture abstract concepts
- Discover how vector arithmetic solves analogies
- Experiment with custom analogies to test the model
- Recognize which types of relationships work well in embeddings

### What You'll Do

1. Load pre-trained GloVe word vectors (400,000 words, 50 dimensions)
2. Examine individual word embeddings across dimensions
3. Investigate "parameter 33" which correlates with "science-ness"
4. Run vector arithmetic to solve analogies (paris - france + italy ≈ rome)
5. Try your own custom analogies interactively

### The Big Idea

> **Relationships between words are preserved as directional patterns in embedding space.**
>
> The direction from "france" to "paris" (country → capital) should be similar to the direction from "italy" to "rome".
>
> This means we can use vector arithmetic: **paris - france + italy ≈ rome**

### Analogy Types That Work Well

- Country ↔ Capital (paris - france + italy ≈ rome)
- Comparatives (smaller - small + big ≈ bigger)
- Verb tenses (walked - walk + swim ≈ swam)
- Pluralization (children - child + person ≈ people)
- Family roles (aunt - uncle + brother ≈ sister)

### Questions

**Q7.** How many dimensions does each GloVe word vector have? How does this compare to Module 0's co-occurrence embeddings?

**Q8.** Looking at the dimension plots for "galaxy", "person", "table", and "atom", what do you notice about parameter 30?

**Q9.** Why do science words have lower values at parameter 33 compared to non-science words? What does this dimension seem to capture?

**Q10.** Predict: What word should complete this analogy: **paris - france + italy = ?**

**Q11.** After running the analogy code, was your prediction correct? What was the top result?

**Q12.** Try your own analogy in the interactive section. Record your input (A, B, C) and the top result. Did it work as expected?

**Q13.** Which type of analogy works best in this model: capital-country, comparative adjectives, verb tenses, or family relationships?

**Q14.** Why do you think some analogies (like animal → sound) don't work well in word embeddings?

**Q15.** How does vector arithmetic (like king - man + woman = queen) demonstrate that embeddings capture meaning relationships rather than just word similarity?

---

## Module 2: Sentence Embeddings & Semantic Search

**Time:** ~25 minutes
**Type:** Colab notebook

### Learning Objectives

- Understand how entire sentences are embedded (not just words)
- Measure semantic similarity between sentences
- Build a semantic search engine that finds documents by meaning
- Connect embeddings to real-world applications (RAG, Google Search)
- Recognize the power of meaning-based retrieval over keyword matching

### What You'll Do

1. Load a SentenceTransformer model (all-MiniLM-L6-v2)
2. Embed 5 example sentences and visualize with PCA
3. Use interactive dropdowns to compare sentence similarities
4. Build a semantic search engine on a 20-sentence corpus
5. Enter natural language queries and get top 3 results by meaning

### The Model: all-MiniLM-L6-v2

- Trained specifically for sentence embeddings
- Produces **384-dimensional vectors** (vs GloVe's 50 dimensions)
- Understands entire sentences as unified concepts
- Used in production systems for document retrieval

### Why This Matters

**Semantic search is the foundation of:**
- Modern search engines (Google, Bing)
- RAG (Retrieval-Augmented Generation) systems
- ChatGPT and Claude's document understanding
- Recommendation systems (Netflix, Spotify, Amazon)
- Question answering from knowledge bases

### Key Insight

> **Traditional search:** Find documents that contain specific keywords
>
> **Semantic search:** Find documents that match the *meaning* of your query
>
> Query: "animals people keep at home"
> → Finds: "Cats are popular pets that like to nap." (no exact word match!)

### Questions

**Q16.** Before seeing the PCA plot, predict: Which two sentences from the list of 5 will be closest together? Why?

**Q17.** After viewing the PCA visualization, which sentences clustered together? Was your prediction correct?

**Q18.** Why is "Neural networks learn patterns" isolated from the other sentences in the PCA plot?

**Q19.** Use the dropdown tool to compare "The cat sat on the mat." with "Cats are great pets." What is their cosine similarity? Are they similar or different?

**Q20.** Now compare "Stars fuse hydrogen into helium." with "Galaxies contain billions of stars." What is their cosine similarity? Is it higher or lower than the cat sentences? Why?

**Q21.** In the semantic search activity, what query did you test? What were the top 3 results?

**Q22.** Do the top 3 results make sense for your query? Would keyword search have found these sentences?

**Q23.** How is semantic search different from searching with Ctrl+F (keyword search)? Give a specific example from your results.

**Q24.** Reflecting on all three modules: How do embeddings allow AI systems like ChatGPT to "understand" the meaning of text rather than just matching keywords?

---

## Key Takeaways

By the end of this lab, you should understand:

- **Embeddings transform language into geometry** where similar meanings point in similar directions
- **Co-occurrence patterns** from billions of words teach models what belongs together
- **Individual dimensions** capture abstract concepts (though not perfectly interpretable)
- **Vector arithmetic works** because relationships are preserved as directional patterns
- **Sentence embeddings** represent entire phrases as unified concepts, not bags of words
- **Semantic search** finds information by meaning, enabling modern AI retrieval
- **PCA lets us visualize** high-dimensional embedding spaces in 2D
- **This is the same core idea as hidden layers** from Lab 4, scaled up for language

---

## Connection to Real-World AI

### Applications of Embeddings

| Application | How Embeddings Are Used |
|-------------|-------------------------|
| **Search Engines** | Google uses embeddings to understand query intent and find relevant pages |
| **RAG Systems** | ChatGPT/Claude embed documents and queries to find relevant context |
| **Recommendations** | Netflix/Spotify embed content and user preferences to suggest similar items |
| **Translation** | Map words from different languages to a shared embedding space |
| **Sentiment Analysis** | Classify text by embedding and measuring distance to sentiment anchors |
| **Document Clustering** | Group similar articles, detect duplicates, organize knowledge bases |
| **Question Answering** | Embed questions and knowledge base to find relevant answers |

### RAG (Retrieval-Augmented Generation)

Modern AI systems like ChatGPT use a two-step process:

1. **Retrieval:** Use semantic search to find relevant documents
2. **Generation:** Use the LLM to generate a response based on retrieved context

This is why they can answer questions about specific documents without retraining—embeddings power the retrieval step!

---

## Optional: Module 3 — RAG in Practice

**Time:** ~20-30 minutes
**Type:** lmnotebook (external assignment)

If your instructor assigns Module 3, you'll explore how semantic search integrates with LLMs to create Retrieval-Augmented Generation systems. You'll use lmnotebook to:

- Build a document collection and embed it
- Ask natural language questions
- See how the system retrieves relevant context
- Observe how the LLM uses that context to generate answers

This is the architecture behind modern AI assistants!

---

## Before You Submit

Make sure you have:

- [ ] Completed all 3 core modules (0, 1, 2)
- [ ] Run all code cells and viewed visualizations
- [ ] Answered all 24 questions in the answer sheet
- [ ] Tried the prediction questions BEFORE running code
- [ ] Experimented with custom analogies in Module 1
- [ ] Tested your own queries in Module 2 semantic search
- [ ] Understood connections to Labs 1-4 (especially Lab 4's hidden layers)
- [ ] Recognized how embeddings power real-world AI systems
- [ ] (Optional) Completed Module 3 RAG activity if assigned

---

## Submission Instructions

Submit your completed answer sheet according to your instructor's guidelines (Canvas upload, PDF, etc.).

**Congratulations!** You've completed your journey from tiny co-occurrence matrices to understanding the geometric representation of meaning that powers modern AI systems!

---

## Additional Resources

Want to learn more about embeddings?

**Foundational Papers:**
- **word2vec:** Mikolov et al. (2013) - "Efficient Estimation of Word Representations"
- **GloVe:** Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"
- **Sentence-BERT:** Reimers & Gurevych (2019) - Sentence embeddings using Siamese networks

**Tools & Libraries:**
- **Gensim:** Python library for word embeddings
- **Sentence-Transformers:** Library for sentence embeddings
- **Hugging Face:** Hub for pre-trained embedding models
- **FAISS:** Facebook's library for efficient similarity search at scale

**Applications:**
- **Pinecone, Weaviate, Qdrant:** Vector databases for production semantic search
- **LangChain:** Framework for building RAG applications
- **LlamaIndex:** Data framework for LLM applications

**Tutorials:**
- Jay Alammar's "The Illustrated Word2vec"
- "Sentence Embeddings: A Primer" on TowardsDataScience
- Hugging Face Sentence Transformers documentation

---

**Questions or feedback?** Reach out to your instructor or TA!
