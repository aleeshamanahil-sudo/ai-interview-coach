# 🏆 AI Interview Coach

> Built during GenAI Cohort 2 Hackathon — Deployed on Hugging Face Spaces

An intelligent interview preparation system powered by RAG, LLMs, and Voice AI.

## ✨ Features
- 🎯 Personalized questions based on job role + resume
- 🗣️ Voice input via Groq Whisper real-time transcription
- 🧠 RAG-powered evaluation using FAISS vector search
- 📊 Emotion detection + filler word analysis
- 🏆 XP system with 11 unlockable badges + leaderboard
- 📄 Downloadable PDF session report

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq LLaMA 3.1 8B |
| Speech | Groq Whisper Large v3 |
| RAG | FAISS + MiniLM-L6-v2 |
| Emotion AI | HuggingFace DistilRoBERTa |
| Frontend | Gradio + Custom CSS |
| Memory | SQLite session memory |
| Deploy | Hugging Face Spaces |

## 🚀 Live Demo
👉 [Try it on Hugging Face](https://huggingface.co/spaces/arfaahm23/AI-Interview-Coach-PROJ)

## 🏗️ How It Works
1. Enter your name, job role, experience level and target company
2. LLaMA 3.1 generates a personalized interview question
3. Answer by voice (Whisper transcribes) or typing
4. FAISS RAG retrieves best coaching knowledge
5. AI evaluates your answer with score + model answer
6. Earn XP, badges and export your PDF report

## 👩‍💻 Built By
**Aleesha Manahil** — Data Science Student, GenAI Cohort 2
- 🤗 [Hugging Face](https://huggingface.co/Aleesha29)
- 💼 [LinkedIn](https://www.linkedin.com/in/aleesha-manahil-406633376/)
