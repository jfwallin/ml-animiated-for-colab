# **Lab 5: Embeddings, Prompting, and the Geometry of Meaning**  
**DATA 1010 – Artificial Intelligence in Action**

---

## **Overview**

### **What You’ll Learn Today**
In today’s lab, you will explore how large large language models (LLMs) represent meaning and why embeddings—high-dimensional vectors—are essential to how these systems work. You will:

- Interact with an LLM to see how different prompts change its behavior  
- Generate embeddings for sentences and visualize them in 2D  
- Build a mini “semantic search engine” using embeddings  
- Return to an LLM to connect these ideas back to foundational model behavior  

### **Lab Structure**

| Module | Title | Time | Type |
|--------|----------------------------------------|--------|---------|
| 0 | Setup & Why LLMs Need Embeddings | ~5 min | Colab |
| 1 | Prompting Foundations | ~15 min | LLM-based |
| 2 | Word & Sentence Embeddings | ~20 min | Colab |
| 3 | Semantic Search | ~15 min | Colab |
| 4 | Foundations Reflection with LLMs | ~10 min | LLM-based |

**Total Time:** ~75 minutes

### **Working in Groups**
- Work in **small groups** of 2–4  
- One group member runs Colab and shares their screen  
- Everyone participates in discussion and prediction questions  
- All group members enter the **same group code**

---

# **Module 0 — Setup & Why LLMs Need Embeddings**

**Time:** ~5 minutes  
**Type:** Colab

### **Learning Objectives**
- Understand that LLMs do not store words—they store numerical vectors  
- Connect embeddings to “hidden space” ideas from Lab 4  
- Prepare your notebook environment for later modules

### **What You'll Do**
1. Open the Lab 5 Colab notebook  
2. Run the initial setup cell  
3. Enter your **group code** (an integer your group chooses)  
4. Generate a sample embedding (e.g., for the word “astronomy”) and see its dimensionality

### **Key Idea**
LLMs operate in a **high-dimensional meaning space**.  
Words or sentences with similar meaning have **vectors pointing in similar directions**.

> This is the same principle you explored in Lab 4: hidden layers create new representations.

### **Q0.**  
*What surprised you about the embedding vector (size, shape, values, etc.)?*  
<br><br>

---

# **Module 1 — Prompting Foundations (LLM Interaction #1)**

**Time:** ~15 minutes  
**Type:** ChatGPT/Claude interaction

### **Learning Objectives**
- Understand how prompts influence the model’s behavior  
- Compare zero-shot and few-shot prompting  
- Observe how chain-of-thought changes reasoning quality  
- Identify cases where LLMs hallucinate or produce ambiguous answers

---

## **Activity 1: Zero-Shot vs Few-Shot Prompting**
Ask the LLM to summarize a short paragraph.

Then provide an **example summary** and ask again.

**Questions to consider:**
- What details change?
- Does the tone match your example?
- Does structure change?

---

## **Activity 2: Chain-of-Thought Prompting**
Ask a reasoning question:

> “If Taylor has 7 apples and gives away 3, how many are left?”

Then ask:

> “Explain your reasoning step by step.”

Compare the responses.

---

## **Activity 3: Ambiguous Prompt & Hallucination**
Ask:

> “Write two sentences about the scientist who discovered element X.”

Choose a rare/obscure element (e.g., *Rhenium* or *Promethium*).

Note whether the model invents details.

---

### **Reflection Questions**

**Q1.** What changed when you added an example (few-shot prompting)?  
<br><br>

**Q2.** Did chain-of-thought improve the reasoning clarity? Explain.  
<br><br>

**Q3.** Describe an example of hallucination or ambiguity you observed. What caused it?  
<br><br>

---

# **Module 2 — Word & Sentence Embeddings**

**Time:** ~20 minutes  
**Type:** Colab

### **Learning Objectives**
- Generate embeddings for sentences using a real model  
- Visualize meaning in 2D using PCA  
- Interpret clusters and distances between sentences  
- Predict similarity using intuition before computing it

---

## **Activity 1: Generate and Visualize Embeddings**
You will embed the following sentences:

1. “The cat sat on the mat.”  
2. “Cats are great pets.”  
3. “Stars fuse hydrogen into helium.”  
4. “Galaxies contain billions of stars.”  
5. “Neural networks learn patterns.”

Run the Colab cells to:

- Generate embeddings  
- Reduce to 2D with PCA  
- Plot the points with labels  

### **Think About**
- Why do astronomy sentences cluster together?  
- Why do cat sentences cluster?  
- Why is “Neural networks learn patterns” isolated?

---

## **Activity 2: Cosine Similarity**
Use the provided dropdown to select **two sentences** and compute:

- Cosine similarity  
- Cosine distance  
- Their positions on the PCA plot  

### **Prediction Questions**

**Q2A.** Which pair do you predict will be the most similar? Why?  
<br><br>

**Q2B.** Which pair do you predict will be the least similar? Why?  
<br><br>

### **After computing**

**Q2C.** Were your predictions correct? Why or why not?  
<br><br>

---

# **Module 3 — Semantic Search Mini-Application**

**Time:** ~15 minutes  
**Type:** Colab

### **Learning Objectives**
- Understand meaning-based search  
- Embed a query and compare it to an embedded corpus  
- Retrieve the most semantically similar items  
- Visualize query placement in PCA space  

---

### **Activity**
You will be given a small corpus (15–20 mixed-topic sentences).

You will:

1. Embed the corpus  
2. Embed your **query**  
3. Compute cosine similarity  
4. Retrieve the **top 3 most similar sentences**  
5. Visualize the query point in PCA  

### **Suggested queries**
- “objects that orbit the sun”  
- “examples of nuclear fusion”  
- “animals people keep at home”  
- “how machines learn patterns”  

---

### **Reflection Questions**

**Q3.** What query did you test?  
<br><br>

**Q4.** Do the top 3 results make sense? Why or why not?  
<br><br>

**Q5.** How is meaning-based search different from keyword-based search?  
<br><br>

---

# **Module 4 — Returning to an LLM: Connecting Embeddings to Foundational Models**

**Time:** ~10 minutes  
**Type:** LLM-based (ChatGPT/Claude)

### **Learning Objectives**
- Connect embeddings to how foundational models operate  
- Understand why prompting shifts behavior  
- Reflect on how embeddings drive text generation and retrieval  

---

### **Activity**
Ask the LLM:

**1.** “Explain how sentence embeddings work using a simple example.”  
Compare the explanation to your PCA plot.

**2.** “If I change one word in a sentence, how does that change the embedding?”  
Discuss small vs large meaning shifts.

**3.** “How do embeddings help LLMs generate coherent responses?”  
Look for ideas about:
- next-token prediction  
- context windows  
- similarity in high-dimensional space  

(Optional)  
Show the LLM your query and top-3 semantic search results:  
> “Do these results make sense? Why?”

---

### **Reflection Questions**

**Q6.** How do embeddings allow models to compare meanings instead of keywords?  
<br><br>

**Q7.** What does it mean to say that “prompting moves you around in embedding space”?  
<br><br>

**Q8.** After this lab, how do you think LLMs “understand” text?  
<br><br>

---

# **Key Takeaways**

By the end of this lab, you should understand:

- Prompting influences how LLMs navigate meaning space  
- Embeddings represent text as high-dimensional vectors  
- Similar meanings → similar directions  
- PCA visualizes tiny slices of these huge spaces  
- Semantic search retrieves items by meaning, not keywords  
- LLMs use embeddings internally for reasoning and generation  
- Foundation models build on the same geometric principles you explored today  

---
