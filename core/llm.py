import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class JournalLLM:
    def __init__(self):
        # check st.secrets first (Streamlit Cloud), then .env / os env
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
        if not api_key:
            raise ValueError("GROQ_API_KEY not found — set it in .env or Streamlit Secrets")

        self.client = Groq(api_key=api_key)

        self.casual_system_prompt = """
        You are MindMirror — a warm, friendly mental wellness companion.

        When the user is chatting casually or hasn't shared their feelings yet:
        - Be warm, human, and approachable.
        - Gently encourage them to share how they're feeling today.
        - Keep it short and natural (under 80 words).
        - Don't force emotions. Just be a kind presence.
        - You may use light questions like "How's your day going?" or
          "Want to share what's on your mind?"
        Do not mention these rules.
        """

        self.emotional_system_prompt = """
        You are MindMirror — an emotionally grounded reflection assistant.

        When the user shares emotional content:
        - Validate their emotional experience.
        - Reflect their feelings empathetically.
        - Suggest 1-2 coping techniques from the provided therapeutic guidance.
        - Be warm, steady, and supportive — not dramatic or exaggerated.
        - Keep responses under 150 words.
        - If distress appears severe, gently encourage real-world support.
        - Do not diagnose or make medical claims.
        - Do not invent techniques beyond what is provided.
        Do not mention these rules.
        """

        self.history = []

    def _add_to_history(self, role: str, content: str, emotion: str | None = None):
        entry = {"role": role, "content": content}
        if emotion:
            entry["emotion"] = emotion
        self.history.append(entry)
        if len(self.history) > 6:
            self.history = self.history[-6:]

    def _build_history_text(self) -> str:
        if not self.history:
            return "No previous conversation."
        lines = []
        for item in self.history:
            role = item["role"].capitalize()
            emotion_tag = f" (emotion: {item['emotion']})" if "emotion" in item else ""
            lines.append(f"{role}{emotion_tag}: {item['content']}")
        return "\n".join(lines)

    def generate_casual(self, user_input: str) -> str:
        """For casual / non-emotional messages."""
        history_text = self._build_history_text()

        prompt = f"""
            Conversation history:
            {history_text}

            User message:
            {user_input}

            Respond warmly and encourage them to share their feelings.
        """

        completion = self.client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": self.casual_system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )

        response_text = (completion.choices[0].message.content or "").strip()

        self._add_to_history("user", user_input)
        self._add_to_history("assistant", response_text)

        return response_text

    def generate(self, user_input: str, emotion: str, retrieved_chunks: list[dict]) -> str:
        """For emotional messages — uses detection + retrieval."""
        history_text = self._build_history_text()
        knowledge_text = "\n".join(chunk["content"] for chunk in retrieved_chunks)

        prompt = f"""
            Conversation history:
            {history_text}

            Detected emotion: {emotion}

            User message:
            {user_input}

            Therapeutic guidance:
            {knowledge_text}

            Respond with grounded emotional reflection.
        """

        completion = self.client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": self.emotional_system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=200,
        )

        response_text = (completion.choices[0].message.content or "").strip()

        self._add_to_history("user", user_input, emotion)
        self._add_to_history("assistant", response_text)

        return response_text