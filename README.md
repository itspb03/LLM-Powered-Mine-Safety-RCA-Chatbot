# LLM-Powered Mine Safety RCA Chatbot

This project builds an end-to-end, **on‑premise root cause analysis (RCA) assistant** for mine safety incidents.  
It combines a **fine‑tuned Llama 3.2‑3B model** with a **Retrieval‑Augmented Generation (RAG)** pipeline over DGMS regulations, and exposes the system as an **interactive Gradio chatbot**.

***

## Problem Statement

Manual RCA of MSHA mine fatality reports is:

- Time‑consuming and resource‑intensive  
- Inconsistent across investigators  
- Hard to keep aligned with evolving **DGMS regulations**  

The goal is to **automate, standardize, and ground** RCA generation while keeping all data **local and privacy‑preserving**.

***

## Key Features

- **Fine‑Tuned LLM (Llama 3.2‑3B + LoRA)**
  - Trained on 349 curated MSHA fatality reports (Description, Investigation, Discussion, RCA)
  - Parameter‑efficient QLoRA fine‑tuning (4‑bit) for single‑GPU (T4‑class) deployment

- **Regulation‑Aware RAG Pipeline**
  - Ingests 3 PDFs: *Mines Act 1952*, *DGMS MMR 2017*, *DGMS CMR 2017*
  - Recursive text chunking (1000 chars, 200‑char overlap)
  - 384‑dim embeddings using **Nomic‑Embed‑Text**
  - Local **Chroma** VectorDB with HNSW indexing

- **Smart Query Processing**
  - Multi‑query expansion to bridge operational language (“roof fall”) and legal language (“roof support system failure”)
  - Top‑K semantic retrieval (K≈3–5) with source + section metadata

- **Augmented RCA Generation**
  - System prompt includes:
    - DGMS regulatory context (retrieved chunks)
    - MSHA incident narrative
    - Task instructions (classification, RCA, recommendations, citations)
  - Outputs: explanation, immediate & underlying causes, DGMS‑linked recommendations

- **Gradio Chatbot UI**
  - Web interface for:
    - Incident text input / upload
    - RCA report preview & copy
    - Visible regulation snippets and citations

- **On‑Premise & Privacy‑First**
  - Runs locally via **Ollama / HF Transformers + Unsloth**
  - No external APIs; suitable for sensitive safety data

***

## Architecture Overview

1. **Fine‑Tuning Stage**
   - Load `meta-llama/Llama-3.2-3B-Instruct`
   - Apply LoRA (rank 16) to Q, K, V, O, Gate, Up, Down
   - Train on MSHA dataset with:
     - LR = 2e‑4, AdamW, 2 epochs, batch size 4 (grad accumulation)
   - Save base + LoRA weights (`.safetensors`)

2. **RAG Stage**
   - Load Mines Act, MMR, CMR PDFs via LangChain `UnstructuredPDFLoader`
   - Chunk with `RecursiveCharacterTextSplitter`:
     - `chunk_size=1000`, `chunk_overlap=200`
   - Embed with `Nomic-Embed-Text` (384‑dim)
   - Store in local **Chroma** collection (`local-rag`)

3. **Inference + Chatbot**
   - User enters incident description in Gradio
   - MultiQueryRetriever generates 3–5 semantic variants
   - Retrieve top‑K chunks from Chroma
   - Build augmented system prompt with:
     - DGMS context
     - Incident details
     - Task steps (classification, RCA, recommendations, citations)
   - Run generation on fine‑tuned Llama 3.2‑3B
   - Render structured RCA + citations in Gradio UI

 


https://github.com/user-attachments/assets/b46b1917-d223-4ce2-b690-0894f40fd8a9


***

## Evaluation

Evaluation is done on 35 held‑out MSHA incidents using three metrics:

- **ROUGE‑L** – structural overlap via Longest Common Subsequence  
- **BERTScore (F1)** – token‑level semantic similarity using contextual embeddings  
- **Embedding Cosine Similarity** – sentence‑level semantic alignment using SentenceTransformers  

Empirically, the **fine‑tuned + RAG model** produces RCAs that are **close to human‑written analyses** and **consistently cite correct regulations**.

***

## Tech Stack

- **Model & Training**
  - Llama 3.2‑3B Instruct (Meta)
  - Unsloth (QLoRA fine‑tuning)
  - PyTorch, Transformers, TRL
- **RAG & Data**
  - LangChain, Unstructured
  - Nomic‑Embed‑Text
  - Chroma VectorDB
- **Serving**
  - Gradio (chatbot UI)
  - Ollama / HF pipeline for local inference
- **Evaluation**
  - `rouge-score`
  - `bert-score`
  - `sentence-transformers`

***



