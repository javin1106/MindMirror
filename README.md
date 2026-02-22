# MindMirror 🧠  
An emotion-aware AI journaling companion that listens, detects what you’re feeling, and responds with supportive, grounded reflection—adapting the entire UI theme to match your mood.

**Live app:** https://mindmirror.streamlit.app/

---

## What is MindMirror?

MindMirror is a **Streamlit chat-based journal** designed to help you pause, reflect, and feel understood.  
When your message contains emotional content, MindMirror:

- **Detects your emotion (≈ 90% accuracy)** with a trained emotion classifier  
- **Retrieves relevant guidance** from an internal knowledge base (RAG-style retrieval)
- **Generates a thoughtful journaling response** using an LLM
- **Visually reflects your mood** by shifting the app’s theme colors based on the detected emotion  
- Shows an **Emotion Timeline** in the sidebar (recent emotions + confidence)

If you’re just saying “hi” or chatting casually, it switches to a **light conversational mode** without over-analyzing.

---

## Features

- **Emotion detection (≈ 90% accuracy)** with confidence scoring  
  Detects: `sadness`, `joy`, `anger`, `fear`, `love`, `surprise` (plus a neutral fallback)
- **Dynamic emotion-based UI themes**  
  Background, accent colors, and chat styling shift based on mood
- **Emotion Timeline**  
  Recent emotional messages are summarized in the sidebar (with confidence %)
- **RAG-style guidance sources**  
  When in emotional mode, the app shows the retrieved “Guidance sources” used to respond
- **Casual-message filtering**  
  Greetings and short messages skip emotion detection to keep interactions natural

---

## Tech Stack

- **Frontend / App:** Streamlit  
- **Emotion Model:** TensorFlow / Keras  
- **Embeddings + Retrieval:** sentence-transformers + FAISS  
- **LLM Provider:** Groq  
- **Utilities:** python-dotenv, numpy, scikit-learn  

---

## Repository Structure (high level)

- `app.py` — Streamlit application (UI, emotion flow, theme injection, chat loop)
- `core/` — core logic (emotion detector, retriever, LLM wrapper)
- `artifacts/` — trained model + tokenizer + label classes
- `emotion_behaviour_data/` — retrieval knowledge base used for guidance
- `notebooks/` — experimentation / training notebooks
- `kaggle_dataset/` — dataset assets (if included)
- `pipeline_test.py` — pipeline sanity checks / tests
- `requirements.txt` — Python dependencies

---

## How it works

### 1) Classify: “Casual” vs “Emotional”
MindMirror first checks whether the message looks like a greeting / small talk (or is very short).  
If so, it responds conversationally.

### 2) Emotion detection (when applicable)
For meaningful journal-style text, the emotion detector predicts:
- the **emotion label**
- the **confidence score**

If confidence is below a threshold, it stays in casual mode.

### 3) Retrieve helpful context
In emotional mode, relevant guidance snippets are retrieved from `emotion_behaviour_data/` and displayed under **Guidance sources**.

### 4) Generate a reflective response
The LLM generates a supportive, journaling-style reply conditioned on:
- your message
- the detected emotion
- retrieved guidance chunks

### 5) Reflect mood visually
The UI theme changes based on the current emotion, making the experience feel more immersive and emotionally “in sync.”

---

## Run locally

### 1) Clone the repo
```bash
git clone https://github.com/javin1106/MindMirror.git
cd MindMirror
```

### 2) Create a virtual environment (recommended)
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Configure environment variables
This project uses **Groq**. Create a `.env` file in the repo root:

```bash
GROQ_API_KEY=your_key_here
```

> If your `core/llm.py` expects different variable names, update the `.env` accordingly.

### 5) Start the app
```bash
streamlit run app.py
```

---

## Notes & Disclaimer

MindMirror is a reflection tool—not a substitute for professional mental health care.  
If you’re in crisis or may be at risk of harm, please seek help from a qualified professional or local emergency resources.

---

## Contributing

Contributions are welcome—ideas, bug fixes, UI improvements, prompt tuning, or better retrieval data.

1. Fork the repo  
2. Create a feature branch: `git checkout -b feature/my-change`  
3. Commit: `git commit -m "Add my change"`  
4. Push: `git push origin feature/my-change`  
5. Open a pull request

---

## License

Add a license file if you plan to distribute/accept contributions publicly (e.g., MIT, Apache-2.0).

---

## Author

Built by **@javin1106**  
Live demo: https://mindmirror.streamlit.app/