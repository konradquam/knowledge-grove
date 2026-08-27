from sentence_transformers import SentenceTransformer 
from knowledge_grove.constants import EMBEDDING_DIM
from functools import lru_cache

class EmbeddingModel:
    def __init__(self, model_name: str = 'intfloat/e5-base-v2'):
        self._model = SentenceTransformer(model_name)
        actual_dim = self._model.get_embedding_dimension()
        if actual_dim != EMBEDDING_DIM:
            raise ValueError(
                f"{model_name} produces {actual_dim}-dim embeddings, "
                f"but the schema expects {EMBEDDING_DIM} (EMBEDDING_DIM)"
            )

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for the given text using the specified model."""
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using the specified model."""
        return self._model.encode(texts).tolist()

@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Return a cached instance of the embedding model."""
    return EmbeddingModel()