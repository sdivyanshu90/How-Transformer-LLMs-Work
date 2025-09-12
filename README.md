# How Transformer LLMs Work

This repository is a **student-friendly reference guide** for the course **"How Transformer LLMs Work"** by [DeepLearning.AI](https://www.deeplearning.ai/) in collaboration with **Jay Alammar** and **Maarten Grootendorst**.

It covers the fundamentals of **Language Models (LMs), Transformers, Tokenization, Self-Attention, and Mixture of Experts (MoE)** in a clear and practical way.

---

## 🏁 Introduction

Large Language Models (LLMs) like **GPT**, **BERT**, and **LLaMA** are built on the **Transformer architecture**. But how do they actually work under the hood?

This guide breaks down the **core concepts** step-by-step:

* From **Bag-of-Words models** → to **Word Embeddings** → to **Attention & Transformers**.
* Explains how **tokenizers process text** for models.
* Shows the inner workings of **Transformer blocks** and **Self-Attention**.
* Introduces **Mixture of Experts (MoE)** as an advanced scaling method.

Think of this README as a **friendly textbook + cheat sheet** for learning or revising Transformers.

---

## 📚 Course Topics and Explanations

### 1. Understanding Language Models: Language as a Bag-of-Words

* **Idea**: Early models treated text as a collection of words without order (like a shopping bag of tokens).
* Each word → represented as an index in a **vocabulary**.
* Problems:

  * Ignores **word order** ("dog bites man" vs "man bites dog").
  * Ignores **context/meaning** (polysemy, synonyms).
* **Takeaway**: Bag-of-Words is simple but too limited for understanding language.

---

### 2. Understanding Language Models: (Word) Embeddings

* **Embeddings** map words to **dense vectors** in continuous space.
* Similar words → close in vector space.
* Example: `king - man + woman ≈ queen`.
* Techniques: **Word2Vec, GloVe, FastText**.
* Solves:

  * Captures **semantic similarity**.
  * Provides compact numeric representation of words.

---

### 3. Encoding and Decoding Context with Attention

* Attention answers: **“Which words should I focus on when interpreting this word?”**
* Example: In *“The animal didn’t cross the street because it was too tired”* →

  * "it" should attend more to "animal", not "street".
* Mechanism:

  * Computes **weights** for each word relative to others.
  * Focuses on **important context**, ignores irrelevant.
* This enables **contextual embeddings** instead of static ones.

---

### 4. Transformers

* Proposed in **“Attention Is All You Need” (2017)**.
* Replace **RNNs/CNNs** with **Self-Attention + Feedforward blocks**.
* Key features:

  * **Parallelizable** (faster training).
  * Captures **long-range dependencies**.
* Transformer = **Encoder + Decoder stacks**.

  * Encoder → processes input sequence.
  * Decoder → generates output sequence.
* Foundation of **BERT, GPT, T5, LLaMA, etc.**

---

### 5. Tokenizers

* LLMs don’t read raw text → they work with **tokens**.
* Types:

  * **Word-level**: simple but large vocab.
  * **Subword-level (BPE, WordPiece)**: balances vocab size + flexibility.
  * **Character-level**: robust but long sequences.
* Example:

  * “unbelievable” → `["un", "believ", "able"]`.
* Tokenization = **bridge between text & model inputs**.

---

### 6. The Transformer Block

* Core building unit of Transformer.
* Components:

  1. **Self-Attention layer** → finds relationships between words.
  2. **Feedforward neural network** → processes attended info.
  3. **Residual connections + Layer normalization** → stabilize training.
* Stacking multiple blocks → deeper understanding of text.

---

### 7. Self-Attention

* Most important mechanism in Transformers.
* Steps:

  1. Convert words into **Queries (Q), Keys (K), Values (V)**.
  2. Compute similarity: `Attention(Q, K, V) = softmax(QKᵀ / √d) V`.
  3. Each word gets a **weighted mix** of all others.
* Example:

  * "The cat sat on the mat" →

    * "cat" attends to "sat" more than "mat".
* **Multi-Head Attention**: multiple attention heads → capture different relationships.

---

### 8. Mixture of Experts (MoE)

* Scaling technique: instead of using **all parameters** for every input, use only a **subset of experts**.
* Structure:

  * Multiple "expert" networks (specialists).
  * A **gating network** decides which experts to activate for each token.
* Benefits:

  * Efficient scaling → larger models without huge compute for every input.
  * Specialization → different experts learn different aspects of language.
* Used in: **Switch Transformers, GLaM, Mixtral**.

---

## 🙏 Acknowledgements

Special thanks to:

* [**DeepLearning.AI**](https://www.deeplearning.ai/) – for providing this excellent course.
* **Jay Alammar** – for his brilliant visualizations and teaching.
* **Maarten Grootendorst** – for insightful contributions to NLP education.

---

## 💡 Tips for Fellow Learners

1. Don’t rush – read each section twice and try to **explain it in your own words**.
2. Use visuals – draw diagrams of attention and transformer blocks to understand better.
3. Practice – implement mini versions of **tokenizers, attention, and transformer layers** in Python/PyTorch.
4. Learn by analogy – think of attention like **spotlights** focusing on important words.
5. Stay updated – follow new LLM research papers to see how these core ideas evolve.

---

🚀 Happy Learning! May this guide help you **understand the magic inside Transformers & LLMs**.
