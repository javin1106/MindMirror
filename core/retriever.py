import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    # kb_path -> knowledge base path
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.chunk_size = 400
        self.overlap = 50

        self.emotion_chunks = {}
        self.emotion_indexes = {}

        self._load_and_index()

    def _chunk_text(self, text: str):
        sections = text.split("\n\nTechnique:")
        chunks = []

        for i, section in enumerate(sections):
            if i == 0:
                chunks.append(section.strip())
            else:
                chunks.append("Technique:" + section.strip())

        return chunks 

    def _load_and_index(self):
        for filename in os.listdir(self.kb_path):
            if not filename.endswith(".txt"):
                continue
                
            emotion = filename.replace(".txt", "")
            filepath = os.path.join(self.kb_path, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            chunks = self._chunk_text(text)

            embeddings = self.model.encode(
                chunks,
                convert_to_numpy = True, 
                normalize_embeddings = True
            ).astype("float32")
            
            embeddings = np.array(embeddings).astype("float32")

            dimension = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

            self.emotion_chunks[emotion] = chunks
            self.emotion_indexes[emotion] = index

    def retrieve(self, query: str, emotion: str, k: int = 3):
        if emotion not in self.emotion_indexes:
            return []

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)
        index = self.emotion_indexes[emotion]
        distances, indices = index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
                
            results.append({
                "content":self.emotion_chunks[emotion][idx],
                "score": float(score)
            })
        
        return results


# -----
# Testing
# -----

# retriever = Retriever("emotion_behaviour_data")

# results = retriever.retrieve(
#     "I feel hopeless and tired of everything.",
#     emotion="sadness",
#     k=3
# )

# for r in results:
#     print("----")
#     print(r["content"][:200])