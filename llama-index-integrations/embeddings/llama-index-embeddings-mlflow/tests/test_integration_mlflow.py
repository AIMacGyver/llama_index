import os
import pytest
from dotenv import load_dotenv

load_dotenv()

try:
    from llama_index.embeddings.mlflow import MLFlowEmbedding
except ImportError:
    MLFlowEmbedding = None

# Get environment variables for Databricks-hosted MLflow
MLFLOW_ENDPOINT = os.getenv("MLFLOW_ENDPOINT")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


@pytest.fixture()
def mlflow_embedding():
    """Fixture to initialize MLFlowEmbedding for Databricks environment."""
    assert DATABRICKS_HOST, "DATABRICKS_HOST is required for Databricks-hosted MLflow"
    assert DATABRICKS_TOKEN, "DATABRICKS_TOKEN is required for Databricks-hosted MLflow"
    assert MLFLOW_ENDPOINT, "MLFLOW_ENDPOINT is required for the MLflow model name"

    # Initialize MLFlowEmbedding with Databricks client
    return MLFlowEmbedding(
        endpoint=MLFLOW_ENDPOINT, client=None, client_name="databricks"
    )


@pytest.mark.skipif(
    MLFlowEmbedding is None or not DATABRICKS_HOST or not DATABRICKS_TOKEN,
    reason="MLFlowEmbedding class could not be imported or Databricks endpoint is not configured.",
)
def test_completion(mlflow_embedding):
    """Test embedding generation with MLflow."""
    try:
        embeddings = mlflow_embedding.get_text_embedding(
            "Testing MLflow Embedding integration."
        )
        assert isinstance(embeddings, list), "Embeddings should be a list."
        assert all(
            isinstance(value, float) for value in embeddings
        ), "Each embedding should be a float."
        assert len(embeddings) > 0, "Embeddings should not be empty."
    except Exception as e:
        pytest.skip(
            f"Skipping test: Unable to access model {MLFLOW_ENDPOINT} at endpoint {DATABRICKS_HOST}. Error: {e}"
        )
