import os
import re
import json
import time
import sqlite3
import fitz  # PyMuPDF
import gradio as gr
import numpy as np
from groq import Groq
from datetime import datetime, date
import random
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# ============================================================
# LAZY IMPORTS — loaded only when first used
# ============================================================
_sentence_transformer = None
_faiss = None
_emotion_model = None

def _get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_transformer

def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss

# ============================================================
# INITIALIZE CLIENT
# ============================================================
client = Groq(api_key=os.environ.get("API_KEY"))

DB_PATH       = "/tmp/interview_coach.db"
FAISS_PATH    = "/tmp/interview_coach.faiss"
CHUNKS_PATH   = "/tmp/interview_chunks.json"

# ============================================================
# KNOWLEDGE BASE
# ============================================================
KNOWLEDGE_BASE = """
==============================
TELL ME ABOUT YOURSELF STRUCTURE
==============================
Strong answer structure:
1. Present: Current education or role.
2. Past: Relevant experience or projects.
3. Skills: Technical and soft skills relevant to job.
4. Future: Career goals aligned with company.
Common mistakes:
* Giving personal life details.
* Talking too long (over 2 minutes).
* No structure or alignment with job role.
Strong answers are concise, structured, and role-focused.
---
==============================
STAR METHOD FOR BEHAVIORAL QUESTIONS
==============================
Situation: Brief background. Keep it short and relevant.
Task: Clearly explain your responsibility.
Action: Explain what YOU specifically did. Avoid "we" without clarifying your own role.
Result: Quantify results if possible (percentage, time saved, cost reduced).
Common mistakes:
* Spending too much time on situation.
* No measurable result.
* Not explaining personal contribution.
Strong STAR answers devote 60% of time to Action and Result.
---
==============================
TECHNICAL INTERVIEW ANSWER STRUCTURE
==============================
Strong technical answers:
1. Clarify the problem.
2. Explain your logic or approach.
3. Provide a concrete example or implementation.
4. Mention edge cases or limitations.
5. Discuss trade-offs.
Common mistakes:
* Jumping directly to code without explanation.
* Not explaining reasoning.
* No clarity in steps.
---
==============================
BEHAVIORAL QUESTION TIPS
==============================
Use the STAR method for every behavioral question.
Focus on your individual actions and decisions.
Quantify your impact wherever possible.
Show self-awareness: mention what you learned.
Avoid vague language like "we did a lot of things."
---
==============================
CONFIDENCE AND COMMUNICATION
==============================
Confident speakers:
* Minimize filler words (um, uh, like, basically).
* Speak at a measured pace.
* Use clear, active sentence construction.
* Back every claim with a specific example.
Filler words reduce perceived credibility. Replace with a 1-second pause.
---
==============================
EVALUATION RUBRIC
==============================
Excellent (9-10):
* Structured response with clear logic.
* Specific, quantified examples.
* Confident, filler-free communication.
Good (7-8):
* Mostly structured with minor clarity issues.
* Some specificity, limited quantification.
Average (5-6):
* Incomplete structure, vague examples, weak results.
Poor (0-4):
* No structure, irrelevant content, no clear impact.
---
==============================
COMMON INTERVIEW MISTAKES
==============================
1. Talking too long without a clear point.
2. Failing to answer the actual question asked.
3. Excessive use of filler words.
4. No concrete examples or metrics.
5. Negative talk about previous employers.
6. No question prepared for the interviewer.
7. Memorised-sounding, robotic delivery.
---
==============================
CLOSING THE INTERVIEW
==============================
Always prepare 2-3 thoughtful questions for the interviewer.
Show curiosity about the team, product, or challenges.
Express genuine enthusiasm for the role.
Reaffirm your key strengths briefly.
Strong close: thank the interviewer, confirm next steps.
"""

# ============================================================
# RAG: CHUNK + EMBED + INDEX
# ============================================================

def _chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words  = text.split()
    chunks = []
    step   = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_vector_index(force_rebuild: bool = False):
    """
    Build (or load) a FAISS flat-L2 index from the knowledge base.
    Persists both the FAISS index and the chunk list to disk.
    Returns (index, chunks).
    """
    faiss = _get_faiss()
    model = _get_sentence_transformer()

    # Load from cache if available
    if not force_rebuild and os.path.exists(FAISS_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            index  = faiss.read_index(FAISS_PATH)
            with open(CHUNKS_PATH, "r") as f:
                chunks = json.load(f)
            print(f"[RAG] Loaded FAISS index ({index.ntotal} vectors, {len(chunks)} chunks)")
            return index, chunks
        except Exception as e:
            print(f"[RAG] Cache load failed ({e}), rebuilding…")

    # Build fresh index
    chunks     = _chunk_text(KNOWLEDGE_BASE)
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    dim        = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)          # Inner-product = cosine sim on normalised vecs
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, FAISS_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)

    print(f"[RAG] Built FAISS index ({index.ntotal} vectors, {len(chunks)} chunks)")
    return index, chunks


# Global RAG state
_rag_index:  object       = None
_rag_chunks: list[str]    = []

def _ensure_rag():
    global _rag_index, _rag_chunks
    if _rag_index is None:
        _rag_index, _rag_chunks = build_vector_index()


def retrieve_context(question: str, answer: str, top_k: int = 3) -> str:
    """
    Semantic search over the knowledge base.
    Returns the top-k most relevant chunks joined as a single string.
    """
    _ensure_rag()
    model = _get_sentence_transformer()
    query = (question + " " + answer).strip()
    qvec  = model.encode([query], normalize_embeddings=True)
    scores, indices = _rag_index.search(np.array(qvec, dtype="float32"), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(_rag_chunks) and score > 0.25:   # similarity threshold
            results.append(_rag_chunks[idx])

    return "\n---\n".join(results) if results else ""


# ============================================================
# DATABASE
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT, role TEXT, question TEXT, answer TEXT,
            overall_score REAL, tech_score INTEGER, clarity_score INTEGER,
            confidence_score REAL, grade TEXT, xp_earned INTEGER,
            badges_earned TEXT, star_score INTEGER, interview_type TEXT,
            rag_context_used INTEGER DEFAULT 0, created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            total_xp INTEGER DEFAULT 0, level TEXT DEFAULT 'Beginner',
            streak_days INTEGER DEFAULT 0, last_practice_date TEXT,
            badges TEXT DEFAULT '[]', total_sessions INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    # Session memory for adaptive context
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT, role TEXT, question TEXT, answer TEXT,
            feedback TEXT, score REAL, created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# SESSION MEMORY (adaptive per-user history)
# ============================================================
def save_session_memory(username: str, role: str, question: str,
                        answer: str, feedback: str, score: float):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "INSERT INTO session_memory (username,role,question,answer,feedback,score,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (username or "guest", role, question, answer, feedback, score,
             datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Memory] Save error: {e}")


def get_session_history(username: str, limit: int = 4) -> str:
    """Return last N Q&A pairs for the user as formatted text."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "SELECT question, answer, feedback, score FROM session_memory "
            "WHERE username=? ORDER BY id DESC LIMIT ?",
            (username or "guest", limit),
        )
        rows = c.fetchall()
        conn.close()
    except Exception:
        return ""

    if not rows:
        return ""

    lines = []
    for q, a, fb, sc in reversed(rows):
        lines.append(f"Q: {q}\nA: {a[:300]}\nScore: {sc}/10\n")
    return "\n".join(lines)


# ============================================================
# XP & LEVELS
# ============================================================
LEVELS = [
    ("Beginner",         0,    ""),
    ("Learner",          200,  ""),
    ("Practitioner",     500,  ""),
    ("Proficient",       1000, ""),
    ("Expert",           2000, ""),
    ("Interview Master", 3500, ""),
]

ALL_BADGES = {
    "first_step":        ("", "First Step",        "Answered your first question"),
    "star_master":       ("", "STAR Master",        "Used complete STAR method"),
    "high_achiever":     ("", "High Achiever",      "Scored 9+ on any question"),
    "on_fire":           ("", "On Fire",            "3 questions in a row >= 7"),
    "diamond_focus":     ("", "Diamond Focus",      "5 questions in a row >= 7"),
    "interview_machine": ("", "Interview Machine",  "Answered 10 questions total"),
    "perfect_score":     ("", "Perfect Score",      "Scored 10/10"),
    "resume_pro":        ("", "Resume Pro",         "Used resume upload"),
    "company_ready":     ("", "Company Ready",      "Used Company Mode"),
    "fluent_speaker":    ("", "Fluent Speaker",     "Zero filler words in an answer"),
    "well_rounded":      ("", "Well-Rounded",       "Scored >= 7 in all 3 categories"),
}

COMPANY_QUESTIONS = {
    "Google": [
        "Tell me about a time you had to solve a technically complex problem with incomplete information.",
        "How would you design Google Maps for a city with no existing mapping data?",
        "Describe a situation where you had to influence a team without direct authority.",
        "What is the most creative solution you have ever implemented? Why was it creative?",
        "How do you handle disagreements with teammates on technical decisions?",
    ],
    "Amazon": [
        "Tell me about a time you delivered results under extreme time pressure. (Customer Obsession)",
        "Describe a situation where you had to make a difficult decision with limited data. (Bias for Action)",
        "Give an example of when you raised the bar for quality on your team. (Insist on the Highest Standards)",
        "Tell me about a time you failed and what you learned from it. (Learn and Be Curious)",
        "How have you used data to drive a major decision? (Are Right, A Lot)",
    ],
    "Microsoft": [
        "Tell me about a project where you had to learn a completely new technology quickly.",
        "How would you improve Microsoft Teams for remote-first companies?",
        "Describe a time you had to collaborate across teams with conflicting priorities.",
        "How do you balance innovation with stability in a large codebase?",
        "Tell me about a time you turned customer feedback into a product improvement.",
    ],
    "Meta": [
        "Move fast and break things — tell me about a time this philosophy helped and hurt you.",
        "How would you redesign the news feed to maximise meaningful social interaction?",
        "Describe a time you had to build something at massive scale.",
        "Tell me about a bold bet you made that did not pan out. What happened?",
        "How do you think about privacy vs personalisation trade-offs in product decisions?",
    ],
    "Startup": [
        "Tell me about a time you wore multiple hats and delivered results under pressure.",
        "How do you prioritise when everything is urgent and the team is small?",
        "Describe how you would validate a product idea in 2 weeks with no budget.",
        "Tell me about a time you built something from scratch with minimal resources.",
        "How do you balance technical debt against shipping speed?",
    ],
}


def get_level(xp: int) -> tuple:
    current = LEVELS[0]
    for lvl in LEVELS:
        if xp >= lvl[1]:
            current = lvl
        else:
            break
    return current


def get_xp_for_answer(score: float, star_score: int,
                       filler_count: int, used_resume: bool) -> int:
    base  = int(score * 10)
    bonus = 0
    if star_score == 4:   bonus += 20
    elif star_score >= 2: bonus += 10
    if score >= 9:        bonus += 30
    elif score >= 8:      bonus += 15
    if filler_count == 0: bonus += 10
    if used_resume:       bonus += 5
    return base + bonus


def check_badges(history, current_entry, star_score,
                 filler_count, used_resume, existing_badges):
    new_badges  = list(existing_badges)
    scores_only = [h["score"] for h in history]

    checks = [
        ("first_step",        len(history) >= 1),
        ("star_master",       star_score == 4),
        ("high_achiever",     current_entry.get("score", 0) >= 9),
        ("perfect_score",     current_entry.get("score", 0) >= 10),
        ("on_fire",           len(scores_only) >= 3 and all(s >= 7 for s in scores_only[-3:])),
        ("diamond_focus",     len(scores_only) >= 5 and all(s >= 7 for s in scores_only[-5:])),
        ("interview_machine", len(history) >= 10),
        ("resume_pro",        used_resume),
        ("fluent_speaker",    filler_count == 0 and len(current_entry.get("answer", "").split()) > 20),
        ("well_rounded",      (current_entry.get("tech", 0) >= 7
                               and current_entry.get("clarity", 0) >= 7
                               and current_entry.get("confidence", 0) >= 7)),
    ]
    for badge_key, condition in checks:
        if condition and badge_key not in new_badges:
            new_badges.append(badge_key)

    return new_badges


# ============================================================
# EMOTION MODEL
# ============================================================
def _load_emotion_model():
    global _emotion_model
    if _emotion_model is not None:
        return _emotion_model
    try:
        from transformers import pipeline
        _emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )
        print("[Emotion] Local model loaded.")
    except Exception as exc:
        print(f"[Emotion] Local model unavailable: {exc}")
        _emotion_model = None
    return _emotion_model

_load_emotion_model()


def detect_emotion(text: str) -> tuple[str, int]:
    model = _emotion_model
    if model is not None:
        try:
            result = model(text[:512])[0]
            return result["label"].capitalize(), round(result["score"] * 100)
        except Exception:
            pass
    # Groq LLM fallback
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": "Emotion classifier. Reply ONE word only: Joy | Sadness | Anger | Fear | Surprise | Disgust | Neutral"},
                {"role": "user",
                 "content": f"Classify dominant emotion: {text[:300]}"},
            ],
            temperature=0.1, max_tokens=5,
        )
        return resp.choices[0].message.content.strip().capitalize(), 0
    except Exception:
        return "Neutral", 0


# ============================================================
# LLM HELPER
# ============================================================
def call_llm(prompt: str, system: str = "You are a helpful assistant.",
             temperature: float = 0.7) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"


# ============================================================
# RESUME PARSER
# ============================================================
def extract_resume_text(pdf_file) -> str:
    if pdf_file is None:
        return ""
    try:
        doc = fitz.open(pdf_file)
        return " ".join(page.get_text() for page in doc)[:3000]
    except Exception as e:
        return f"Resume parse error: {str(e)}"


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================
def transcribe_audio(file_path: str) -> str:
    if not file_path:
        return ""
    try:
        with open(file_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text",
            )
        return (transcription if isinstance(transcription, str)
                else transcription.text).strip()
    except Exception as e:
        return f"Audio transcription failed: {str(e)}"


# ============================================================
# SPEECH ANALYSIS
# ============================================================
def analyze_speech(text: str) -> tuple[float, dict]:
    if not text.strip():
        return 5.0, {}
    words         = text.split()
    filler_words  = ["um", "uh", "like", "you know", "basically",
                     "literally", "actually", "right", "so"]
    filler_bd     = {f: text.lower().split().count(f)
                     for f in filler_words if text.lower().split().count(f) > 0}
    filler_count  = sum(filler_bd.values())
    base_score    = min(10, len(words) / 15)
    conf_score    = round(max(1.0, min(10.0, base_score - filler_count * 0.4)), 1)
    return conf_score, filler_bd


# ============================================================
# STAR METHOD DETECTOR
# ============================================================
def detect_star_method(text: str) -> tuple[dict, int]:
    indicators = {
        "Situation": ["when", "during", "there was", "i was working", "at my",
                      "in my previous", "the project"],
        "Task":      ["needed to", "had to", "responsible for", "my goal was",
                      "i was asked", "my role was"],
        "Action":    ["i did", "i implemented", "i built", "i resolved", "i led",
                      "i created", "i developed", "i fixed"],
        "Result":    ["as a result", "which led to", "achieved", "improved",
                      "reduced", "increased", "we saw", "the outcome"],
    }
    found = {k: any(kw in text.lower() for kw in v) for k, v in indicators.items()}
    return found, sum(found.values())


# ============================================================
# LIVE ANSWER STATS
# ============================================================
def get_live_stats(text: str) -> str:
    if not text:
        return "Words: 0  |  Fillers: 0  |  Quality: Too Short"
    words        = text.split()
    wc           = len(words)
    filler_words = ["um", "uh", "like", "basically", "literally", "actually"]
    fc           = sum(text.lower().split().count(f) for f in filler_words)
    _, star_score = detect_star_method(text)
    if wc < 30:    quality = "Too Short"
    elif wc < 80:  quality = "Developing"
    elif wc < 150: quality = "Good"
    else:          quality = "Comprehensive"
    return (f"**Words:** {wc}  |  **Fillers:** {fc}  |  "
            f"**Quality:** {quality}  |  **STAR:** {star_score}/4")


# ============================================================
# BENCHMARK
# ============================================================
def get_benchmark(score: float) -> str:
    if score >= 8.5: return "**Top 10%** of candidates"
    if score >= 7.5: return "**Top 25%** of candidates"
    if score >= 6.5: return "**Above average**"
    if score >= 5.0: return "**Average performer**"
    return "**Needs improvement**"


def extract_score(text: str, label: str) -> int:
    m = re.search(rf"{label}.*?(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else 5


# ============================================================
# QUESTION GENERATOR
# ============================================================
def generate_question(role, experience, difficulty, interview_type,
                      resume_file, history, company_mode):
    if not role:
        return "Please enter a job role first.", history

    # Company-specific question bank
    if company_mode and company_mode != "None":
        bank      = COMPANY_QUESTIONS.get(company_mode, [])
        asked     = {h.get("question", "") for h in (history or [])}
        remaining = [q for q in bank if q not in asked]
        if remaining and random.random() > 0.3:
            return random.choice(remaining), history

    resume_text  = extract_resume_text(resume_file)
    resume_ctx   = f"\nCandidate Resume:\n{resume_text}" if resume_text else ""
    prev_qa      = ""
    if history:
        lines = [f"Q: {h['question']}\nA: {h.get('answer','')}"
                 for h in history[-3:] if h.get("answer")]
        if lines:
            prev_qa = "\nPrevious Q&A:\n" + "\n".join(lines)
    company_hint = (
        f"\nThis is a {company_mode} interview. Match their culture and values."
        if company_mode and company_mode != "None" else ""
    )
    prompt = (
        f"Generate ONE realistic {difficulty} level {interview_type} interview question "
        f"for a {experience} {role}.{resume_ctx}{prev_qa}{company_hint}\n"
        f"Rules: reference resume if provided; ask a follow-up if prior answers exist. "
        f"Return ONLY the question text."
    )
    return call_llm(prompt, temperature=0.8), history


# ============================================================
# MAIN EVALUATOR  (RAG-enhanced)
# ============================================================
def evaluate_answer(username, role, question, text_answer, audio,
                    interview_type, history, resume_file,
                    company_mode, user_data):

    if not question or question.startswith("Please enter"):
        return ("Please generate a question first.",
                0, 0, 0, 0, "", "", "", history, "", user_data, "")

    # --- Transcribe audio if provided ---
    if audio:
        transcribed = transcribe_audio(audio)
        if transcribed and not transcribed.startswith("Audio transcription failed"):
            text_answer = transcribed

    if not text_answer or not text_answer.strip():
        return ("Please type or record an answer.",
                0, 0, 0, 0, "", "", "", history, "", user_data, "")

    # --- Local analysis ---
    conf_score, filler_bd  = analyze_speech(text_answer)
    star_found, star_score = detect_star_method(text_answer)
    filler_count           = sum(filler_bd.values())
    used_resume            = resume_file is not None
    emotion, emotion_conf  = detect_emotion(text_answer)
    emotion_display        = (f"**Tone:** {emotion}"
                              + (f" ({emotion_conf}%)" if emotion_conf else ""))

    # --- RAG: retrieve relevant knowledge ---
    rag_context     = retrieve_context(question, text_answer, top_k=3)
    rag_context_blk = (
        f"\n\nRelevant Interview Coaching Knowledge:\n{rag_context}"
        if rag_context else ""
    )
    rag_used = bool(rag_context)

    # --- Adaptive session history ---
    history_ctx = get_session_history(username or "guest", limit=4)
    history_blk = (
        f"\n\nCandidate's Recent Practice History:\n{history_ctx}"
        if history_ctx else ""
    )

    # --- STAR note ---
    star_note = ""
    if interview_type == "Behavioral":
        star_status = "  |  ".join(
            f"{'✓' if v else '✗'} {k}" for k, v in star_found.items()
        )
        star_note = f"\n\n**STAR Check:** {star_status}  ({star_score}/4)"

    # --- LLM evaluation with RAG context ---
    system_prompt =(
        """You are a senior hiring manager with 15 years of experience 
      at top-tier tech companies. You evaluate answers with precision and consistency.
      SCORING RUBRIC (follow strictly):
      - 9-10: Specific metrics, perfect STAR structure, zero vagueness, compelling narrative
      - 7-8: Good structure, some specifics, minor clarity issues  
      - 5-6: Vague examples, incomplete structure, missing impact
      - 3-4: Rambling, off-topic, or no concrete examples
      - 1-2: Completely irrelevant or incoherent
      Never give 8+ without a quantified result. Never give 5+ without at least one concrete example."""
    )
    eval_prompt = f"""
      ROLE BEING INTERVIEWED FOR: {role}
      INTERVIEW TYPE: {interview_type}
      DIFFICULTY: Senior
      {rag_context_blk}
      {history_blk}
      QUESTION ASKED: {question}
      CANDIDATE ANSWER: {text_answer}
      WORD COUNT: {len(text_answer.split())} words
      SCORE STRICTLY using the rubric. Deduct points for:
      - No quantified result (-2 points from Technical)  
      - Filler words detected: {filler_count} (-{min(filler_count, 3)} from Communication)
      - Answer under 50 words (-2 from Clarity)
      - No personal "I" contribution (-1 from Technical)
      Return EXACTLY this format with no deviation:
      Technical Knowledge: X/10
      Clarity: X/10
      Communication: X/10
      **Strengths:**
      - [Quote specific words from the answer, explain why they're strong]
      - [Quote specific words from the answer, explain why they're strong]
      **Areas to Improve:**
      - [Cite the specific coaching principle from the knowledge base that applies]
      - [Be specific: what word/phrase should they change and to what]
      **Model Answer (for this exact question and role):**
      [3-5 sentences. Must include: situation context, specific action, quantified result]
      **One-Line Coach Tip:**
      [The single most important thing for this candidate to fix next time]
      """

    feedback      = call_llm(eval_prompt, system=system_prompt, temperature=0.4)
    tech_score    = extract_score(feedback, "Technical Knowledge")
    clarity_score = extract_score(feedback, "Clarity")
    comm_score    = extract_score(feedback, "Communication")
    overall_score = round((tech_score + clarity_score + comm_score + conf_score) / 4, 1)

    if overall_score >= 8.5:   grade = "A+"
    elif overall_score >= 7.5: grade = "A"
    elif overall_score >= 6.5: grade = "B+"
    elif overall_score >= 5.5: grade = "B"
    else:                      grade = "C"

    filler_text = ""
    if filler_bd:
        fl_list     = ", ".join(f"'{k}' ×{v}" for k, v in filler_bd.items())
        filler_text = f"\n\n**Filler Words Detected:** {fl_list}\n💡 Replace with a deliberate 1-second pause."

    full_feedback = feedback + filler_text + star_note

    # --- Update in-memory history ---
    current_entry = {
        "question":   question,
        "answer":     text_answer,
        "score":      overall_score,
        "grade":      grade,
        "tech":       tech_score,
        "clarity":    clarity_score,
        "confidence": conf_score,
        "comm":       comm_score,
    }
    history = history or []
    history.append(current_entry)

    # --- XP & badges ---
    xp_earned             = get_xp_for_answer(overall_score, star_score, filler_count, used_resume)
    user_data             = user_data or {"xp": 0, "badges": [], "sessions": 0}
    user_data["xp"]       = user_data.get("xp", 0) + xp_earned
    user_data["sessions"] = user_data.get("sessions", 0) + 1
    old_badges            = user_data.get("badges", [])
    new_badges            = check_badges(history, current_entry, star_score,
                                         filler_count, used_resume, old_badges)
    newly_earned          = [b for b in new_badges if b not in old_badges]
    user_data["badges"]   = new_badges

    level_name  = get_level(user_data["xp"])[0]
    avg_score   = round(sum(h["score"] for h in history) / len(history), 1)
    session_sum = (
        f"Q{len(history)} complete  |  Avg: **{avg_score}/10**  |  "
        f"XP: **+{xp_earned}**  |  Total: **{user_data['xp']} XP**  |  **{level_name}**"
    )
    badge_notif = ""
    if newly_earned:
        names = [f"**{ALL_BADGES[b][1]}**" for b in newly_earned if b in ALL_BADGES]
        badge_notif = "🏅 New Badge(s): " + ", ".join(names)

    # --- Persist to DB ---
    _persist_session(
        username, role, question, text_answer, overall_score,
        tech_score, clarity_score, conf_score, grade, xp_earned,
        newly_earned, star_score, interview_type, rag_used,
        new_badges, level_name, user_data,
    )

    # --- Save to session memory for future adaptive context ---
    save_session_memory(
        username or "guest", role, question, text_answer,
        feedback, overall_score,
    )

    return (
        full_feedback, tech_score, clarity_score, conf_score, overall_score,
        grade, emotion_display, get_benchmark(overall_score),
        history, session_sum, user_data, badge_notif,
    )


def _persist_session(username, role, question, text_answer, overall_score,
                     tech_score, clarity_score, conf_score, grade, xp_earned,
                     newly_earned, star_score, interview_type, rag_used,
                     new_badges, level_name, user_data):
    if not username:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            """INSERT INTO sessions
               (username,role,question,answer,overall_score,tech_score,clarity_score,
                confidence_score,grade,xp_earned,badges_earned,star_score,
                interview_type,rag_context_used,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (username, role, question, text_answer, overall_score, tech_score,
             clarity_score, conf_score, grade, xp_earned,
             json.dumps(newly_earned), star_score, interview_type,
             int(rag_used), datetime.now().isoformat()),
        )
        c.execute(
            "INSERT OR IGNORE INTO users (username, created_at) VALUES (?,?)",
            (username, datetime.now().isoformat()),
        )
        c.execute(
            """UPDATE users SET total_xp=?,level=?,badges=?,total_sessions=?,
               last_practice_date=? WHERE username=?""",
            (user_data["xp"], level_name, json.dumps(new_badges),
             user_data["sessions"], date.today().isoformat(), username),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Error: {e}")

# ============================================================
# HISTORY TABLE
# ============================================================
def format_history_table(history):
    if not history:
        return "No questions answered yet."
    rows = ["| # | Question | Score | Grade | Tech | Clarity | Confidence |",
            "|---|----------|-------|-------|------|---------|------------|"]
    for i, h in enumerate(history, 1):
        q = h["question"][:50] + "..." if len(h["question"]) > 50 else h["question"]
        rows.append(
            f"| {i} | {q} | **{h['score']}/10** | {h['grade']} | "
            f"{h.get('tech','-')} | {h.get('clarity','-')} | {h.get('confidence','-')} |"
        )
    avg = round(sum(h["score"] for h in history) / len(history), 1)
    rows.append(f"\n**Session Average: {avg}/10**")
    return "\n".join(rows)

# ============================================================
# LEADERBOARD
# ============================================================
def get_leaderboard():
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            SELECT username, total_xp, level, total_sessions,
                   (SELECT AVG(overall_score) FROM sessions s WHERE s.username = u.username) as avg_score
            FROM users u ORDER BY total_xp DESC LIMIT 10
        """)
        rows = c.fetchall()
        conn.close()
        if not rows:
            return "No leaderboard data yet. Be the first!"
        lines  = ["| Rank | Name | Level | Avg Score | XP | Sessions |",
                  "|------|------|-------|-----------|-----|----------|"]
        medals = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, 11)]
        for i, (uname, xp, lvl, sess, avg) in enumerate(rows):
            avg_str = f"{avg:.1f}" if avg else "—"
            lines.append(f"| {medals[i]} | {uname} | {lvl} | {avg_str}/10 | {xp} | {sess} |")
        return "\n".join(lines)
    except Exception as e:
        return f"Error loading leaderboard: {e}"


# ============================================================
# PDF REPORT
# ============================================================
def generate_pdf(username, role, question, feedback, overall, grade, history, user_data):
    # Fixed: no apostrophe in filename, safe fallback for username
    safe_name = (username or "anonymous").replace(" ", "_").replace("'", "")
    file_path = f"/tmp/{safe_name}_Interview_Report.pdf"
    try:
        # Sanitize inputs
        username  = username or "Anonymous"
        role      = role or "Not specified"
        question  = str(question or "N/A").replace("**", "").replace("*", "")
        feedback  = str(feedback or "No feedback available.").replace("**", "").replace("*", "").replace("#", "")
        grade     = str(grade or "N/A").replace("**", "").replace("*", "").strip()

        try:
            overall_float = float(overall) if overall else 0.0
        except (ValueError, TypeError):
            overall_float = 0.0

        doc    = SimpleDocTemplate(
            file_path,
            rightMargin=0.75*inch, leftMargin=0.75*inch,
            topMargin=0.75*inch,   bottomMargin=0.75*inch,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"], fontSize=20,
            textColor=colors.HexColor("#1A1A2E"), alignment=TA_CENTER, spaceAfter=6,
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"], fontSize=13,
            textColor=colors.HexColor("#1A1A2E"),
        )
        sub_style = ParagraphStyle(
            "Sub", parent=styles["Normal"], fontSize=10,
            textColor=colors.grey, alignment=TA_CENTER,
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"], fontSize=10,
            textColor=colors.HexColor("#2E2E42"), leading=16,
        )

        el = []
        el.append(Paragraph("AI Interview Coach — Report", title_style))
        el.append(Paragraph(
            f"Candidate: {username}  |  Role: {role}  |  "
            f"{datetime.now().strftime('%B %d, %Y')}",
            sub_style,
        ))
        el.append(Spacer(1, 0.25*inch))

        ud         = user_data or {}
        xp         = ud.get("xp", 0)
        level_info = get_level(xp)
        benchmark  = get_benchmark(overall_float).replace("**", "").replace("*", "")

        summary_data = [
            ["Metric",        "Value"],
            ["Overall Score", f"{overall_float}/10"],
            ["Grade",         grade],
            ["Level",         level_info[0]],
            ["Total XP",      f"{xp} XP"],
            ["Sessions",      str(ud.get("sessions", 0))],
            ["Benchmark",     benchmark],
        ]
        tbl = Table(summary_data, colWidths=[2.8*inch, 3.7*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1A1A2E")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#F5F4F0"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E8E6E0")),
            ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        el.append(Paragraph("Performance Summary", h2_style))
        el.append(Spacer(1, 0.1*inch))
        el.append(tbl)
        el.append(Spacer(1, 0.3*inch))

        el.append(Paragraph("Interview Question", h2_style))
        el.append(Spacer(1, 0.08*inch))
        el.append(Paragraph(question, body_style))
        el.append(Spacer(1, 0.25*inch))

        el.append(Paragraph("Detailed Feedback", h2_style))
        el.append(Spacer(1, 0.08*inch))
        for line in feedback.split("\n"):
            line = line.strip()
            if line:
                el.append(Paragraph(line, body_style))
                el.append(Spacer(1, 0.04*inch))
        el.append(Spacer(1, 0.25*inch))

        if history:
            el.append(Paragraph("Full Session History", h2_style))
            el.append(Spacer(1, 0.08*inch))
            hist_data = [["#", "Question", "Score", "Grade"]]
            for i, h in enumerate(history, 1):
                q_text = h.get("question", "")[:55] + ("..." if len(h.get("question","")) > 55 else "")
                hist_data.append([
                    str(i), q_text,
                    f"{h.get('score', 0)}/10",
                    str(h.get("grade", "—")),
                ])
            ht = Table(hist_data, colWidths=[0.4*inch, 4.0*inch, 0.9*inch, 0.9*inch])
            ht.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1A1A2E")),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#F5F4F0"), colors.white]),
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#E8E6E0")),
                ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            el.append(ht)

        doc.build(el)
        return file_path

    except Exception as e:
        print(f"PDF generation error: {e}")
        return None
# ============================================================
# DASHBOARD HTML
# ============================================================
def build_dashboard_html(username, user_data, history):
    if not username:
        return """
<div style="text-align:center;padding:70px 20px;font-family:'Outfit',sans-serif;background:#F5F4F0;">
  <div style="font-size:17px;font-weight:700;color:#1A1A2E;margin-bottom:6px;">
    Enter your name in the Practice tab</div>
  <div style="font-size:13px;color:#888880;">
    Your dashboard, XP, and badges will appear once you start a session.</div>
</div>"""

    ud       = user_data or {"xp": 0, "badges": [], "sessions": 0}
    xp       = ud.get("xp", 0)
    badges   = ud.get("badges", [])
    sessions = ud.get("sessions", 0)
    lvl_name, lvl_min, _ = get_level(xp)

    lvl_idx  = next((i for i, l in enumerate(LEVELS) if l[0] == lvl_name), 0)
    next_xp  = LEVELS[lvl_idx + 1][1] if lvl_idx < len(LEVELS) - 1 else lvl_min
    xp_pct   = (
        min(100, int((xp - lvl_min) / max(1, next_xp - lvl_min) * 100))
        if lvl_idx < len(LEVELS) - 1 else 100
    )

    avg_score = round(sum(h["score"] for h in history) / len(history), 1) if history else 0
    weekly    = sessions * 3
    streak    = ud.get("streak_days", 1) if sessions > 0 else 0

    if avg_score >= 8:   ql, qc, qbg = "Excellent", "#059669", "#ECFDF5"
    elif avg_score >= 6.5: ql, qc, qbg = "Good",    "#2B5CE6", "#EFF4FF"
    elif avg_score > 0:  ql, qc, qbg = "Developing","#D97706", "#FFFBEB"
    else:                ql, qc, qbg = "No data",  "#888880", "#F5F4F0"

    parts    = username.split()
    initials = (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()
    hour     = datetime.now().hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")

    badge_html = ""
    for bk in badges[:8]:
        b = ALL_BADGES.get(bk, ("", "Unknown", ""))
        badge_html += (
            f'<div style="display:inline-flex;align-items:center;gap:7px;background:#F5F4F0;'
            f'border:1.5px solid #E0DED8;border-radius:8px;padding:6px 14px;margin:3px;'
            f'font-size:12px;color:#1A1A2E;font-weight:600;">{b[1]}</div>'
        )
    if not badge_html:
        badge_html = '<div style="color:#B0AEA8;font-size:13px;">Answer questions to earn badges.</div>'

    activity_html = ""
    if history:
        for h in history[-5:]:
            sc    = h["score"]
            q_pre = h["question"][:58]
            sc_color = "#059669" if sc >= 8 else ("#D97706" if sc >= 6 else "#DC2626")
            sc_bg    = "#ECFDF5"  if sc >= 8 else ("#FFFBEB"  if sc >= 6 else "#FEF2F2")
            activity_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:11px 0;border-bottom:1px solid #F0EEE8;">'
                f'<span style="font-size:13px;color:#5A5A70;flex:1;padding-right:16px;">{q_pre}...</span>'
                f'<span style="font-size:12px;font-weight:700;color:{sc_color};background:{sc_bg};'
                f'padding:3px 10px;border-radius:6px;white-space:nowrap;">{sc}/10</span>'
                f'</div>'
            )
    else:
        activity_html = '<div style="color:#B0AEA8;font-size:13px;padding:12px 0;">No activity yet.</div>'

    return f"""
<div style="font-family:'Outfit',sans-serif;background:#F5F4F0;min-height:100%;padding:24px;color:#1A1A2E;">
  <div style="display:grid;grid-template-columns:1.4fr 0.8fr 0.8fr 0.8fr;gap:14px;margin-bottom:16px;">
    <div style="background:#FFFFFF;border:1.5px solid #E8E6E0;border-radius:16px;padding:22px;
                box-shadow:0 1px 4px rgba(0,0,0,0.04);">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
        <div style="width:48px;height:48px;border-radius:12px;background:#1A1A2E;
                    display:flex;align-items:center;justify-content:center;
                    font-size:17px;font-weight:800;color:white;flex-shrink:0;">{initials}</div>
        <div>
          <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.8px;">{greeting}</div>
          <div style="font-size:16px;font-weight:700;color:#1A1A2E;margin-top:1px;">{username}</div>
          <div style="font-size:12px;color:#2B5CE6;font-weight:600;margin-top:1px;">{lvl_name}</div>
        </div>
      </div>
      <div style="background:#F5F4F0;border-radius:10px;padding:12px 14px;border:1px solid #E8E6E0;">
        <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
          <span style="font-size:11px;color:#888880;font-weight:600;">XP Progress</span>
          <span style="font-size:11px;color:#2B5CE6;font-weight:700;">{xp} XP</span>
        </div>
        <div style="background:#E8E6E0;border-radius:999px;height:5px;overflow:hidden;">
          <div style="background:#2B5CE6;height:100%;width:{xp_pct}%;border-radius:999px;"></div>
        </div>
        <div style="font-size:10px;color:#B0AEA8;margin-top:5px;">Next level at {next_xp} XP</div>
      </div>
    </div>
    <div style="background:#FFFFFF;border:1.5px solid #E8E6E0;border-radius:16px;padding:22px;
                box-shadow:0 1px 4px rgba(0,0,0,0.04);display:flex;flex-direction:column;
                justify-content:space-between;">
      <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.8px;margin-bottom:10px;">Streak</div>
      <div style="font-size:46px;font-weight:800;color:#059669;line-height:1;">{streak}</div>
      <div style="font-size:12px;color:#888880;margin-top:6px;">days practiced</div>
    </div>
    <div style="background:#FFFFFF;border:1.5px solid #E8E6E0;border-radius:16px;padding:22px;
                box-shadow:0 1px 4px rgba(0,0,0,0.04);display:flex;flex-direction:column;
                justify-content:space-between;">
      <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.8px;margin-bottom:10px;">Practice Time</div>
      <div style="font-size:46px;font-weight:800;color:#D97706;line-height:1;">{weekly}</div>
      <div style="font-size:12px;color:#888880;margin-top:6px;">minutes total</div>
    </div>
    <div style="background:{qbg};border:1.5px solid #E8E6E0;border-radius:16px;padding:22px;
                box-shadow:0 1px 4px rgba(0,0,0,0.04);display:flex;flex-direction:column;
                justify-content:space-between;">
      <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.8px;margin-bottom:10px;">Session Quality</div>
      <div style="font-size:38px;font-weight:800;line-height:1;color:{qc};">
        {avg_score}<span style="font-size:16px;color:#B0AEA8;font-weight:600;">/10</span></div>
      <div style="font-size:12px;color:{qc};font-weight:600;margin-top:6px;">{ql}</div>
    </div>
  </div>
  <div style="background:#FFFFFF;border:1.5px solid #E8E6E0;border-radius:16px;
              padding:20px 22px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
    <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;
                letter-spacing:0.8px;margin-bottom:12px;">Badges ({len(badges)}/11)</div>
    <div>{badge_html}</div>
  </div>
  <div style="background:#FFFFFF;border:1.5px solid #E8E6E0;border-radius:16px;
              padding:20px 22px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
    <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;
                letter-spacing:0.8px;margin-bottom:4px;">Recent Activity</div>
    {activity_html}
  </div>
</div>"""

# ============================================================
# CSS
# ============================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #F5F4F0 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #1A1A2E !important;
}
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; padding: 0 !important; }
.tab-nav {
    background: #FFFFFF !important;
    border-bottom: 2px solid #E8E6E0 !important;
    padding: 0 32px !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
.tab-nav button {
    background: transparent !important;
    color: #888880 !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 15px 22px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85em !important;
    border-radius: 0 !important;
    transition: all 0.18s !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    margin-bottom: -2px !important;
}
.tab-nav button:hover { color: #2B5CE6 !important; }
.tab-nav button.selected { color: #2B5CE6 !important; border-bottom-color: #2B5CE6 !important; }
.gr-box, .gr-form, .gr-panel, .gr-group, .gr-block, .gr-accordion,
[class*="block"], .panel, .form {
    background: #FFFFFF !important;
    border: 1.5px solid #E8E6E0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
}
input[type="text"], textarea, select {
    background: #FAFAF8 !important;
    border: 1.5px solid #D8D6D0 !important;
    border-radius: 10px !important;
    color: #1A1A2E !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95em !important;
    padding: 10px 14px !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #2B5CE6 !important;
    box-shadow: 0 0 0 3px rgba(43, 92, 230, 0.1) !important;
    outline: none !important;
}
label > span { color: #5A5A70 !important; font-weight: 600 !important;
    font-size: 0.76em !important; text-transform: uppercase !important; letter-spacing: 0.7px !important; }
button.primary, .gr-button-primary, button[variant="primary"], [data-testid="primary"] {
    background: #1A1A2E !important; color: #FFFFFF !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 700 !important;
    font-size: 0.92em !important; padding: 13px 28px !important;
    box-shadow: 0 2px 10px rgba(26,26,46,0.15) !important;
    transition: all 0.18s ease !important; cursor: pointer !important;
}
button.primary:hover { background: #2B5CE6 !important; transform: translateY(-1px) !important; }
button.secondary, .gr-button-secondary, button[variant="secondary"], [data-testid="secondary"] {
    background: #FFFFFF !important; color: #1A1A2E !important;
    border: 1.5px solid #D8D6D0 !important; border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
    font-size: 0.92em !important; padding: 11px 22px !important;
    transition: all 0.18s ease !important;
}
button.secondary:hover { background: #F0EEE8 !important; border-color: #2B5CE6 !important; color: #2B5CE6 !important; }
input[type="range"] { accent-color: #2B5CE6 !important; }
.gr-markdown, .gr-markdown p, .gr-markdown li { color: #2E2E42 !important; line-height: 1.7 !important; }
.gr-markdown h2 { color: #1A1A2E !important; font-weight: 700 !important; border-bottom: 2px solid #E8E6E0 !important; padding-bottom: 7px !important; }
.gr-markdown h3 { color: #2B5CE6 !important; font-weight: 600 !important; }
.gr-markdown strong { color: #1A1A2E !important; }
.gr-markdown code { background: #F0EEE8 !important; color: #2B5CE6 !important; border-radius: 5px !important; padding: 2px 6px !important; }
.gr-audio { background: #FAFAF8 !important; border: 1.5px dashed #C8C6C0 !important; border-radius: 12px !important; }
.gr-file  { background: #FAFAF8 !important; border: 1.5px dashed #C8C6C0 !important; border-radius: 12px !important; }
::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: #F5F4F0; }
::-webkit-scrollbar-thumb { background: #D0CEC8; border-radius: 999px; }
footer { display: none !important; }
"""

# ============================================================
# GRADIO UI
# ============================================================
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="blue", secondary_hue="slate", neutral_hue="stone",
        font=gr.themes.GoogleFont("Outfit"),
    ).set(
        body_background_fill="#F5F4F0", body_text_color="#1A1A2E",
        block_background_fill="#FFFFFF", block_border_color="#E8E6E0",
        block_label_text_color="#5A5A70",
        input_background_fill="#FAFAF8", input_border_color="#D8D6D0",
        button_primary_background_fill="#1A1A2E", button_primary_text_color="white",
        button_secondary_background_fill="#FFFFFF", button_secondary_border_color="#D8D6D0",
        button_secondary_text_color="#1A1A2E",
    ),
    title="AI Interview Coach",
) as app:

    session_history = gr.State([])
    user_data_state = gr.State({})

    gr.HTML("""
<div style="background:#FFFFFF;padding:28px 40px;border-bottom:1.5px solid #E8E6E0;
            box-shadow:0 1px 6px rgba(0,0,0,0.05);">
  <div style="display:flex;align-items:center;justify-content:space-between;
              max-width:1120px;margin:0 auto;">
    <div>
      <div style="font-size:20px;font-weight:800;color:#1A1A2E;font-family:'Outfit',sans-serif;
                  letter-spacing:-0.3px;">AI Interview Coach</div>
      <div style="font-size:13px;color:#888880;font-family:'Outfit',sans-serif;margin-top:2px;">
        Practice smarter. Get hired faster.</div>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">
      <span style="background:#F0EEE8;color:#5A5A70;padding:5px 12px;border-radius:6px;
                   font-size:11px;font-weight:600;font-family:'Outfit',sans-serif;
                   border:1px solid #E0DED8;">Groq LLaMA 3.1</span>
      <span style="background:#F0EEE8;color:#5A5A70;padding:5px 12px;border-radius:6px;
                   font-size:11px;font-weight:600;font-family:'Outfit',sans-serif;
                   border:1px solid #E0DED8;">Groq Whisper API</span>
      <span style="background:#F0EEE8;color:#5A5A70;padding:5px 12px;border-radius:6px;
                   font-size:11px;font-weight:600;font-family:'Outfit',sans-serif;
                   border:1px solid #E0DED8;">Emotion AI</span>
      <span style="background:#EFF4FF;color:#2B5CE6;padding:5px 12px;border-radius:6px;
                   font-size:11px;font-weight:700;font-family:'Outfit',sans-serif;
                   border:1px solid #C8D8FF;">XP + Badges</span>
    </div>
  </div>
</div>""")

    with gr.Tabs():

        # ── Practice ──────────────────────────────
        with gr.TabItem("Practice"):
            with gr.Row():
                with gr.Column(scale=3):
                    username_input = gr.Textbox(
                        label="Your Name",
                        placeholder="Enter your name to track progress",
                        max_lines=1,
                    )
                with gr.Column(scale=1):
                    company_mode = gr.Dropdown(
                        ["None", "Google", "Amazon", "Microsoft", "Meta", "Startup"],
                        label="Company Focus", value="None",
                    )
            with gr.Row():
                with gr.Column(scale=2):
                    role       = gr.Textbox(label="Job Role",
                                            placeholder="e.g. Data Scientist, Backend Engineer")
                    resume_file = gr.File(label="Resume PDF (optional)", file_types=[".pdf"])
                with gr.Column(scale=1):
                    experience    = gr.Dropdown(
                        ["Student", "Fresher (0-1 yr)", "Junior (1-3 yrs)",
                         "Mid-level (3-6 yrs)", "Senior (6+ yrs)"],
                        label="Experience Level", value="Fresher (0-1 yr)",
                    )
                    difficulty    = gr.Dropdown(["Easy", "Medium", "Hard"],
                                                label="Difficulty", value="Medium")
                    interview_type = gr.Dropdown(
                        ["Technical", "HR", "Behavioral", "System Design", "Case Study"],
                        label="Interview Type", value="Technical",
                    )

            generate_btn = gr.Button("Generate Question", variant="primary", size="lg")

            gr.HTML('<div style="height:1px;background:#E8E6E0;margin:20px 0;"></div>')
            gr.HTML('<div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Interview Question</div>')
            question_output = gr.Markdown(value="Click Generate Question to begin.")

            gr.HTML('<div style="height:1px;background:#E8E6E0;margin:14px 0;"></div>')
            gr.HTML('<div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Your Answer</div>')

            with gr.Row():
                with gr.Column(scale=3):
                    text_answer = gr.Textbox(
                        label="Type Your Answer",
                        placeholder="Start typing — live stats update as you write.",
                        lines=6,
                    )
                    live_stats_out = gr.Markdown(value="Start typing to see live analysis.")
                with gr.Column(scale=2):
                    audio = gr.Audio(
                        sources=["microphone"], type="filepath",
                        label="Record Your Answer (uses Groq Whisper API)",
                    )

            text_answer.change(get_live_stats, inputs=[text_answer], outputs=[live_stats_out])

            with gr.Row():
                evaluate_btn = gr.Button("Evaluate My Answer", variant="primary")
                next_btn     = gr.Button("Next Question",      variant="secondary")
                reset_btn    = gr.Button("Reset Session",      variant="secondary")

            gr.HTML('<div style="height:1px;background:#E8E6E0;margin:20px 0;"></div>')
            session_progress    = gr.Markdown()
            badge_notification  = gr.Markdown()

            with gr.Row():
                grade_output     = gr.Markdown()
                benchmark_output = gr.Markdown()
                emotion_output   = gr.Markdown()

            gr.HTML('<div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:14px 0 8px;">Score Breakdown</div>')
            with gr.Row():
                tech_bar       = gr.Slider(0, 10, label="Technical",  interactive=False)
                clarity_bar    = gr.Slider(0, 10, label="Clarity",    interactive=False)
                confidence_bar = gr.Slider(0, 10, label="Confidence", interactive=False)
                overall_bar    = gr.Slider(0, 10, label="Overall",    interactive=False)

            gr.HTML('<div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:14px 0 8px;">Detailed Feedback</div>')
            feedback_output = gr.Markdown()

        # ── Dashboard ─────────────────────────────
        with gr.TabItem("Dashboard"):
            dashboard_html    = gr.HTML(build_dashboard_html("", {}, []))
            refresh_dash_btn  = gr.Button("Refresh Dashboard", variant="secondary")

        # ── Session History ───────────────────────
        with gr.TabItem("Session History"):
            history_table = gr.Markdown(value="No questions answered yet.")

        # ── Leaderboard ───────────────────────────
        with gr.TabItem("Leaderboard"):
            gr.HTML('<div style="font-size:18px;font-weight:800;color:#1A1A2E;font-family:Outfit,sans-serif;padding:8px 0 4px;">Global Leaderboard</div>')
            leaderboard_md = gr.Markdown(value=get_leaderboard())
            refresh_lb_btn = gr.Button("Refresh", variant="secondary")

        # ── Report ────────────────────────────────
        with gr.TabItem("Download Report"):
            gr.HTML('<div style="font-size:18px;font-weight:800;color:#1A1A2E;font-family:Outfit,sans-serif;padding:8px 0 4px;">Export Your Report</div>')
            gr.Markdown("PDF with scores, feedback, and full session history.")
            pdf_btn  = gr.Button("Generate PDF Report", variant="primary")
            pdf_file = gr.File(label="PDF Report")

        # ── Tips ──────────────────────────────────
        with gr.TabItem("Tips & Guide"):
            gr.HTML("""
<div style="padding:4px;font-family:'Outfit',sans-serif;color:#1A1A2E;">
  <div style="font-size:20px;font-weight:800;margin-bottom:6px;">How to Ace Your Interview</div>
  <div style="font-size:14px;color:#888880;margin-bottom:20px;">Practical techniques from top candidates</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
    <div style="background:#FFFFFF;border:1.5px solid #E8E6E0;border-radius:14px;padding:22px;">
      <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">The STAR Method</div>
      <div style="display:grid;gap:8px;">
        <div style="background:#F5F4F0;border-radius:8px;padding:10px 14px;font-size:13px;color:#5A5A70;"><strong style="color:#2B5CE6;">S</strong>ituation — Set the scene with context</div>
        <div style="background:#F5F4F0;border-radius:8px;padding:10px 14px;font-size:13px;color:#5A5A70;"><strong style="color:#2B5CE6;">T</strong>ask — Your specific responsibility</div>
        <div style="background:#F5F4F0;border-radius:8px;padding:10px 14px;font-size:13px;color:#5A5A70;"><strong style="color:#2B5CE6;">A</strong>ction — What YOU specifically did</div>
        <div style="background:#F5F4F0;border-radius:8px;padding:10px 14px;font-size:13px;color:#5A5A70;"><strong style="color:#2B5CE6;">R</strong>esult — Quantified outcome</div>
      </div>
    </div>
    <div style="background:#FFFFFF;border:1.5px solid #E8E6E0;border-radius:14px;padding:22px;">
      <div style="font-size:10px;color:#888880;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">Delivery Tips</div>
      <div style="display:grid;gap:7px;">
        <div style="font-size:13px;color:#5A5A70;padding:7px 0;border-bottom:1px solid #F0EEE8;">Speak at 120-150 words per minute</div>
        <div style="font-size:13px;color:#5A5A70;padding:7px 0;border-bottom:1px solid #F0EEE8;">Use a pause instead of filler words</div>
        <div style="font-size:13px;color:#5A5A70;padding:7px 0;border-bottom:1px solid #F0EEE8;">Vary tone to emphasise key points</div>
        <div style="font-size:13px;color:#5A5A70;padding:7px 0;">Aim for 80-150 words per answer</div>
      </div>
    </div>
  </div>
</div>""")

    # ── Wiring ────────────────────────────────────
    generate_btn.click(
        generate_question,
        inputs=[role, experience, difficulty, interview_type,
                resume_file, session_history, company_mode],
        outputs=[question_output, session_history],
    )

    def evaluate_and_update(username, role, question, text_answer, audio,
                            interview_type, history, resume_file, company_mode, user_data):
        results = evaluate_answer(
            username, role, question, text_answer, audio,
            interview_type, history, resume_file, company_mode, user_data,
        )
        (feedback, tech, clarity, conf, overall, grade, emotion, benchmark,
         new_history, session_sum, new_user_data, badge_notif) = results
        table = format_history_table(new_history)
        dash  = build_dashboard_html(username, new_user_data, new_history)
        return (feedback, tech, clarity, conf, overall, grade, emotion, benchmark,
                new_history, session_sum, new_user_data, badge_notif, table, dash)

    evaluate_btn.click(
        evaluate_and_update,
        inputs=[username_input, role, question_output, text_answer, audio,
                interview_type, session_history, resume_file, company_mode, user_data_state],
        outputs=[feedback_output, tech_bar, clarity_bar, confidence_bar, overall_bar,
                 grade_output, emotion_output, benchmark_output,
                 session_history, session_progress, user_data_state,
                 badge_notification, history_table, dashboard_html],
    )

    next_btn.click(
        lambda r, e, d, t, rf, h, cm: (
            *generate_question(r, e, d, t, rf, h, cm), "", None
        ),
        inputs=[role, experience, difficulty, interview_type,
                resume_file, session_history, company_mode],
        outputs=[question_output, session_history, text_answer, audio],
    )

    def do_reset():
        return ([], {}, "Click Generate Question to start.", "", None,
                "", "", "", "", 0, 0, 0, 0, "No questions answered yet.", "", "")

    reset_btn.click(
        do_reset,
        outputs=[session_history, user_data_state, question_output,
                 text_answer, audio,
                 feedback_output, grade_output, emotion_output, benchmark_output,
                 tech_bar, clarity_bar, confidence_bar, overall_bar,
                 history_table, session_progress, badge_notification],
    )

    refresh_dash_btn.click(
        build_dashboard_html,
        inputs=[username_input, user_data_state, session_history],
        outputs=dashboard_html,
    )

    refresh_lb_btn.click(get_leaderboard, outputs=leaderboard_md)

    pdf_btn.click(
        generate_pdf,
        inputs=[username_input, role, question_output, feedback_output,
                overall_bar, grade_output, session_history, user_data_state],
        outputs=pdf_file,
    )

if __name__ == "__main__":
    app.launch(share=False)