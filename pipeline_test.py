from core.detector import EmotionDetector
from core.retriever import Retriever
from core.llm import JournalLLM


def run_pipeline(entry: str):
    print("\n" + "=" * 70)
    print("USER ENTRY:")
    print(entry)
    print("=" * 70)

    # 1️⃣ Load components
    detector = EmotionDetector(
        model_path="artifacts/model.keras",
        tokenizer_path="artifacts/tokenizer.pkl",
        label_path="artifacts/label_classes.pkl"
    )

    retriever = Retriever("emotion_behaviour_data")  # or your knowledge_base path
    llm = JournalLLM()

    # 2️⃣ Emotion detection
    detection = detector.predict(entry)
    emotion = detection["label"]
    confidence = detection["confidence"]

    print(f"\nDetected Emotion: {emotion}")
    print(f"Confidence: {round(confidence, 3)}")

    # 3️⃣ Retrieval
    chunks = retriever.retrieve(entry, emotion, k=3)

    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk["content"][:300])

    # 4️⃣ LLM generation
    response = llm.generate(entry, emotion, chunks)

    print("\nFINAL RESPONSE:")
    print(response)


if __name__ == "__main__":
    test_inputs = [
        "I feel exhausted and nothing feels meaningful lately.",
        "I keep getting angry at small things and I don't like who I'm becoming.",
        "Explain quicksort algorithm."
    ]

    for text in test_inputs:
        run_pipeline(text)