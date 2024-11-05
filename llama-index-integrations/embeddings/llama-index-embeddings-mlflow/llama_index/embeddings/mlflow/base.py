"""MLFlow embeddings file."""

import os
import logging
from typing import Any, List, Dict, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.mlflow.providers import CloudProvider
from mlflow.deployments import (
    get_deploy_client,
    BaseDeploymentClient,
    DatabricksDeploymentClient,
)
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class MLFlowEmbedding(BaseEmbedding):
    """MLFlow Embedding class.

    Embeds text using a model deployed via MLflow.

    Args:
        endpoint (str): The name or ID of the deployed MLflow model endpoint.
        client_name (Optional[str]): The cloud provider name, e.g., "databricks".
        input_key (str):  The key in the input dictionary for the text to embed. Defaults to "input".
        output_key (str): The key in the output dictionary containing the embedding. Defaults to "embedding".
        max_retries (int): Maximum number of retries for requests.
        timeout (float): Timeout for each request.
       **kwargs: Additional keyword arguments.

    """

    endpoint: str = Field(description="The deployed MLflow model endpoint name/ID")
    client_name: Optional[str] = Field(
        default=None, description="The cloud provider name, e.g., databricks"
    )
    input_key: str = Field(default="input", description="Input key for the text")
    output_key: str = Field(
        default="embedding", description="Output key for the embedding"
    )
    max_retries: int = Field(default=5, description="Maximum number of retries", ge=0)
    timeout: float = Field(default=30.0, description="Timeout for each request", ge=0)
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs"
    )

    _client: Optional[BaseDeploymentClient] = PrivateAttr(default=None)

    def __init__(
        self,
        endpoint: str,
        client_name: Optional[str],
        input_key: str = "input",
        output_key: str = "embedding",
        max_retries: int = 5,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        load_dotenv()

        super().__init__(
            endpoint=endpoint,
            client_name=client_name,
            input_key=input_key,
            output_key=output_key,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs,
        )
        self.endpoint = endpoint
        self.client_name = client_name
        self.input_key = input_key
        self.output_key = output_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = self._get_client()

    def _get_client(self) -> BaseDeploymentClient:
        """Instantiate and return an MLflow client if not already set."""
        if not self._client:
            logger.info(
                f"Creating MLflow client for cloud provider: {self.client_name}"
            )
            if self.client_name == CloudProvider.DATABRICKS.value:
                self._client = self._get_databricks_client()
            else:
                logger.error(f"Unsupported Cloud Provider: {self.client_name}")
                raise ValueError(f"Unsupported Cloud Provider: {self.client_name}")
        return self._client

    def _get_databricks_client(self) -> DatabricksDeploymentClient:
        """Return a Databricks client."""
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not databricks_host or not databricks_token:
            missing_vars = []
            if not databricks_host:
                missing_vars.append("DATABRICKS_HOST")
            if not databricks_token:
                missing_vars.append("DATABRICKS_TOKEN")
            raise OSError(
                f"Missing required environment variable(s) for Databricks client: {', '.join(missing_vars)}"
            )

        try:
            client = get_deploy_client(self.client_name)
            logger.info("Databricks client created successfully.")
            return client
        except Exception as e:
            logger.error(f"Failed to create Databricks client: {e}")
            raise

    @classmethod
    def class_name(cls) -> str:
        return "MLFlowEmbedding"

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Helper method to get embeddings for a list of texts."""
        logger.info(f"Sending request to MLFlow model endpoint: {self.endpoint}")
        client = self._get_client()

        # Attempt the request with retries for robustness
        for attempt in range(self.max_retries):
            try:
                response = client.predict(
                    endpoint=self.endpoint, inputs={self.input_key: texts}
                ).data

                if isinstance(response, list):
                    return [item.get(self.output_key, []) for item in response]
                else:
                    logger.error(
                        f"Unexpected response format from MLflow model: {self.endpoint}; response: {response}"
                    )
                    raise ValueError("Invalid response format from MLflow model.")
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for MLflow model request: {e}"
                )
                if attempt < self.max_retries - 1:
                    continue
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed for MLflow model request."
                    )
                    raise
        raise RuntimeError("Failed to obtain embeddings after all retry attempts.")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._get_embeddings(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_embeddings(texts)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get single text embedding."""
        return self._get_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get single text embedding."""
        return self._get_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronously get query embedding."""
        return await self._aget_text_embedding(query)
