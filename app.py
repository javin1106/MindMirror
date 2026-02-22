import streamlit as st
import re
from core.detector import EmotionDetector
from core.retriever import Retriever
from core.llm import JournalLLM


st.set_page_config(page_title="MindMirror", page_icon="🧠", layout="centered")

# ---------- EMOTION THEME CONFIG ----------

EMOTION_EMOJI = {
    "sadness": "😢", "joy": "😊", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲",
}

EMOTION_THEMES = {
    "neutral": {
        "bg": "#1a1a2e", "accent": "#e0e0e0", "text": "#ffffff",
        "chat_user": "#2d2d44", "chat_bot": "#16213e",
    },
    "sadness": {
        "bg": "#0d1b2a", "accent": "#5DADE2", "text": "#d6eaf8",
        "chat_user": "#1b2838", "chat_bot": "#102a43",
    },
    "joy": {
        "bg": "#2c2a10", "accent": "#F4D03F", "text": "#fef9e7",
        "chat_user": "#3d3a18", "chat_bot": "#2c2a10",
    },
    "anger": {
        "bg": "#2a0d0d", "accent": "#E74C3C", "text": "#fadbd8",
        "chat_user": "#3b1515", "chat_bot": "#2a0d0d",
    },
    "fear": {
        "bg": "#1a0d2e", "accent": "#8E44AD", "text": "#e8daef",
        "chat_user": "#281845", "chat_bot": "#1a0d2e",
    },
    "love": {
        "bg": "#2e0d1a", "accent": "#EC7063", "text": "#fadbd8",
        "chat_user": "#3b1525", "chat_bot": "#2e0d1a",
    },
    "surprise": {
        "bg": "#0d2e2a", "accent": "#48C9B0", "text": "#d5f5e3",
        "chat_user": "#153b35", "chat_bot": "#0d2e2a",
    },
}

# confidence threshold — below this we treat it as casual chat
EMOTION_THRESHOLD = 0.50
# words/phrases that are clearly casual — skip emotion detection entirely
CASUAL_PATTERNS = re.compile(
    r"^(hi|hey|hello|yo|sup|hii+|helo+|what'?s up|howdy|good morning|good evening|"
    r"good night|good afternoon|gm|gn|thanks|thank you|ok|okay|sure|yes|no|yep|nope|"
    r"bye|goodbye|see you|later|cool|nice|great|haha|lol|lmao|hmm+|hm|bruh|bro|"
    r"how are you|what do you do|who are you|tell me about yourself|"
    r"nothing much|not much|nm|idk|nah)[\s!?.]*$",
    re.IGNORECASE,
)

# minimum word count to even consider running the detector
MIN_WORDS_FOR_DETECTION = 4


def is_casual_message(text: str) -> bool:
    """Return True if the message is clearly casual/greeting — no need for emotion detection."""
    cleaned = text.strip()
    if CASUAL_PATTERNS.match(cleaned):
        return True
    if len(cleaned.split()) < MIN_WORDS_FOR_DETECTION:
        return True
    return False
# ---------- CACHED COMPONENTS ----------

@st.cache_resource
def load_components():
    detector = EmotionDetector(
        model_path="artifacts/model.keras",
        tokenizer_path="artifacts/tokenizer.pkl",
        label_path="artifacts/label_classes.pkl"
    )
    retriever = Retriever("emotion_behaviour_data")
    llm = JournalLLM()
    return detector, retriever, llm


detector, retriever, llm = load_components()


# ---------- SESSION STATE ----------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "greeted" not in st.session_state:
    st.session_state.greeted = False


# ---------- DYNAMIC THEME ----------

def get_theme():
    return EMOTION_THEMES.get(st.session_state.current_emotion, EMOTION_THEMES["neutral"])


def inject_theme_css():
    t = get_theme()
    st.markdown(f"""
    <style>
        /* main background */
        .stApp {{
            background-color: {t['bg']};
            transition: background-color 0.8s ease;
        }}

        /* header area */
        header[data-testid="stHeader"] {{
            background-color: {t['bg']} !important;
        }}

        /* sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {t['bg']} !important;
            border-right: 2px solid {t['accent']}40;
        }}

        /* text color */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {{
            color: {t['text']} !important;
        }}

        /* chat input */
        .stChatInput textarea {{
            background-color: {t['chat_user']} !important;
            color: {t['text']} !important;
            border: 1px solid {t['accent']}60 !important;
        }}

        /* chat messages */
        div[data-testid="stChatMessage"] {{
            border: 1px solid {t['accent']}30;
            border-radius: 12px;
            margin-bottom: 8px;
        }}

        /* accent glow on current emotion */
        .emotion-glow {{
            text-shadow: 0 0 20px {t['accent']}, 0 0 40px {t['accent']}80;
        }}
    </style>
    """, unsafe_allow_html=True)


inject_theme_css()


# ---------- SIDEBAR ----------

with st.sidebar:
    t = get_theme()
    st.markdown(
        f"<h1 style='color:{t['accent']};' class='emotion-glow'>🧠 MindMirror</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Your AI journal companion")
    st.markdown("---")

    # current mood indicator
    emo = st.session_state.current_emotion
    if emo != "neutral":
        emoji = EMOTION_EMOJI.get(emo, "💭")
        st.markdown(
            f"<div style='text-align:center;font-size:2em;'>{emoji}</div>"
            f"<div style='text-align:center;color:{t['accent']};font-weight:bold;'>"
            f"{emo.upper()}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

    # emotion timeline
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user" and m.get("emotion")]
    if user_msgs:
        st.subheader("Emotion Timeline")
        for msg in user_msgs[-5:]:
            emoji = EMOTION_EMOJI.get(msg["emotion"], "💭")
            theme = EMOTION_THEMES.get(msg["emotion"], EMOTION_THEMES["neutral"])
            conf = msg.get("confidence", 0)
            st.markdown(
                f"<span style='color:{theme['accent']};'>{emoji} <b>{msg['emotion'].title()}</b></span>"
                f" <span style='color:gray;font-size:0.8em;'>({conf:.0%})</span>",
                unsafe_allow_html=True,
            )
        st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_emotion = "neutral"
        st.session_state.greeted = False
        llm.history = []
        st.rerun()


# ---------- WELCOME GREETING ----------

if not st.session_state.greeted:
    greeting = "Hey there! 👋 I'm MindMirror. How are you feeling today?"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.greeted = True


# ---------- CHAT HISTORY ----------

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        # emotion badge on emotional user messages
        if msg["role"] == "user" and msg.get("emotion"):
            emoji = EMOTION_EMOJI.get(msg["emotion"], "💭")
            theme = EMOTION_THEMES.get(msg["emotion"], EMOTION_THEMES["neutral"])
            conf = msg.get("confidence", 0)
            st.markdown(
                f"<span style='background-color:{theme['accent']};color:black;padding:2px 10px;"
                f"border-radius:12px;font-size:0.75em;font-weight:bold;'>"
                f"{emoji} {msg['emotion'].upper()} ({conf:.0%})</span>",
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])


# ---------- CHAT INPUT ----------

if prompt := st.chat_input("Write what's on your mind..."):

    # --- 1. Decide: casual or emotional? ---
    if is_casual_message(prompt):
        # skip detector entirely for greetings / small talk
        is_emotional = False
        emotion = None
        confidence = 0.0
    else:
        detection = detector.predict(prompt)
        emotion = detection["label"]
        confidence = detection["confidence"]
        is_emotional = confidence >= EMOTION_THRESHOLD

    # --- 2. Show user message ---
    with st.chat_message("user", avatar="🧑"):
        if is_emotional:
            emoji = EMOTION_EMOJI.get(emotion, "💭")
            theme = EMOTION_THEMES.get(emotion, EMOTION_THEMES["neutral"])
            st.markdown(
                f"<span style='background-color:{theme['accent']};color:black;padding:2px 10px;"
                f"border-radius:12px;font-size:0.75em;font-weight:bold;'>"
                f"{emoji} {emotion.upper()} ({confidence:.0%})</span>",
                unsafe_allow_html=True,
            )
        st.markdown(prompt)

    # --- 3. Store user message ---
    user_msg = {"role": "user", "content": prompt}
    if is_emotional:
        user_msg["emotion"] = emotion
        user_msg["confidence"] = confidence
    st.session_state.messages.append(user_msg)

    # --- 4. Generate response ---
    with st.chat_message("assistant", avatar="🧠"):
        if is_emotional:
            # emotional mode: retrieval + therapeutic response + theme change
            st.session_state.current_emotion = emotion

            with st.spinner("Reflecting..."):
                chunks = retriever.retrieve(prompt, emotion, k=3)
                response = llm.generate(
                    user_input=prompt, emotion=emotion, retrieved_chunks=chunks,
                )
            st.markdown(response)

            with st.expander("📚 Guidance sources"):
                for i, chunk in enumerate(chunks, 1):
                    st.caption(f"**Source {i}**")
                    st.write(chunk["content"])
        else:
            # casual mode: friendly conversational response
            with st.spinner("Thinking..."):
                response = llm.generate_casual(user_input=prompt)
            st.markdown(response)

    # --- 5. Store assistant message ---
    st.session_state.messages.append({"role": "assistant", "content": response})

    # --- 6. Rerun to apply new theme color ---
    if is_emotional:
        st.rerun()