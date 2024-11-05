import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

# Attempt to import MLFlowEmbedding. This may fail in CI/CD environments due to dependency issues.
try:
    from llama_index.embeddings.mlflow import MLFlowEmbedding
except ImportError:
    MLFlowEmbedding = None


@pytest.mark.skipif(MLFlowEmbedding is None, reason="MLFlowEmbedding is missing")
def test_mlflow_embedding_class():
    """Test to ensure MLFlowEmbedding inherits from BaseEmbedding."""
    names_of_base_classes = [b.__name__ for b in MLFlowEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes
