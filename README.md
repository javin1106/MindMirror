# 🧠 MindMirror

> **Your AI-powered journal companion for emotional self-reflection**

MindMirror is a Streamlit-based mental wellness chatbot that detects the emotion behind what you write and responds with empathetic, therapeutically-grounded guidance — all in a UI that dynamically adapts its color theme to match your current mood.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Emotion Detection** | Custom-trained deep learning model classifies text into 6 emotions with confidence scores |
| 📚 **Retrieval-Augmented Generation (RAG)** | Relevant therapeutic techniques are retrieved from a curated knowledge base using FAISS + sentence embeddings |
| 🤖 **LLM Responses** | Groq-powered language model generates warm, grounded, empathetic replies |
| 🎨 **Dynamic Emotion Themes** | The entire UI color scheme shifts to reflect the detected emotion in real time |
| 💬 **Casual & Emotional Modes** | Short greetings and small-talk bypass emotion detection for a natural conversation flow |
| 📊 **Emotion Timeline** | Sidebar tracks the last 5 emotional moments in the conversation |
| 🗑️ **Clear Chat** | One-click reset of the full session |

---

## 🏗️ Architecture

```
User message
     │
     ▼
 is_casual_message()
     │
  ┌──┴──────────────────────────┐
  │ Casual                      │ Emotional
  ▼                             ▼
JournalLLM                EmotionDetector
.generate_casual()         .predict(text)
                               │
                         confidence ≥ 0.50?
                               │
                          ┌────┴────┐
                          │        │
                         Yes       No
                          │        │
                     Retriever   JournalLLM
                     .retrieve() .generate_casual()
                          │
                    JournalLLM
                    .generate(emotion, chunks)
                          │
                     UI theme update
```

**Core components:**

- **`core/detector.py` — EmotionDetector**  
  Loads a pre-trained Keras model (`artifacts/model.keras`) alongside a fitted tokenizer and label encoder. Pads input sequences to length 50 and returns the predicted emotion label with its softmax confidence.

- **`core/retriever.py` — Retriever**  
  Reads per-emotion `.txt` knowledge-base files from `emotion_behaviour_data/`, splits them on `\n\nTechnique:` boundaries, encodes each chunk with `all-MiniLM-L6-v2`, and stores them in per-emotion FAISS `IndexFlatIP` indexes for fast cosine-similarity retrieval.

- **`core/llm.py` — JournalLLM**  
  Wraps the Groq chat completion API. Maintains a rolling 6-message conversation history. Uses a *casual* system prompt for small-talk and an *emotional* system prompt that asks the model to validate feelings and suggest retrieved coping techniques.

- **`app.py` — Streamlit frontend**  
  Orchestrates everything: session state, dynamic CSS theming, sidebar emotion timeline, chat history rendering, and the four-step pipeline (detect → retrieve → generate → display).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Emotion model | TensorFlow / Keras |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| LLM | [Groq](https://console.groq.com/) (`groq/compound`) |
| Env management | `python-dotenv` |
| Language | Python 3.11+ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- A [Groq API key](https://console.groq.com/)

### 1 — Clone the repository

```bash
git clone https://github.com/javin1106/MindMirror.git
cd MindMirror
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Streamlit Cloud deployment?** Add `GROQ_API_KEY` to your app's [Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) instead.

### 5 — Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 📁 Project Structure

```
MindMirror/
├── app.py                        # Streamlit application entry point
├── pipeline_test.py              # CLI pipeline smoke-test
├── requirements.txt              # Python dependencies
│
├── core/
│   ├── detector.py               # EmotionDetector — Keras inference
│   ├── retriever.py              # Retriever — FAISS RAG engine
│   └── llm.py                    # JournalLLM — Groq chat wrapper
│
├── artifacts/
│   ├── model.keras               # Pre-trained emotion classification model
│   ├── tokenizer.pkl             # Fitted Keras tokenizer
│   └── label_classes.pkl         # Emotion label encoder classes
│
├── emotion_behaviour_data/       # Per-emotion CBT knowledge base
│   ├── anger.txt
│   ├── fear.txt
│   ├── joy.txt
│   ├── love.txt
│   ├── sadness.txt
│   └── surprise.txt
│
├── kaggle_dataset/               # Training data (Kaggle emotions dataset)
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
├── notebooks/
│   └── train.ipynb               # Model training notebook
│
└── .devcontainer/
    └── devcontainer.json         # GitHub Codespaces configuration
```

---

## ⚙️ Configuration

| Variable | Where | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` or Streamlit Secrets | Required — Groq API key for LLM responses |

### Tunable constants in `app.py`

| Constant | Default | Description |
|---|---|---|
| `EMOTION_THRESHOLD` | `0.50` | Minimum confidence to treat a message as emotional |
| `MIN_WORDS_FOR_DETECTION` | `4` | Messages shorter than this skip emotion detection |

---

## 🎭 Supported Emotions

| Emotion | Emoji | UI Accent Color |
|---|---|---|
| Sadness | 😢 | `#5DADE2` (soft blue) |
| Joy | 😊 | `#F4D03F` (golden yellow) |
| Anger | 😠 | `#E74C3C` (red) |
| Fear | 😨 | `#8E44AD` (purple) |
| Love | ❤️ | `#EC7063` (rose) |
| Surprise | 😲 | `#48C9B0` (teal) |

---

## 🧪 Running the Pipeline Test

To verify that all three components (detector, retriever, LLM) work end-to-end without the Streamlit UI:

```bash
python pipeline_test.py
```

This runs three sample journal entries through the full pipeline and prints the detected emotion, retrieved chunks, and final LLM response to the console.

---

## ☁️ Deploy on Streamlit Cloud

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your repo.
3. Set `GROQ_API_KEY` in the **Secrets** section of your app settings.
4. Click **Deploy**.

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to your fork and open a Pull Request.

---

## 📄 License

This project is open-source. See the [LICENSE](LICENSE) file for details.
