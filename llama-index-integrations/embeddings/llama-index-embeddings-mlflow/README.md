# LlamaIndex Embeddings Integration: MLflow

This integration adds support for embedding models hosted on MLflow serving endpoints, with specific integration for Databricks. By leveraging the MLflow deployment API, it enables embedding text with models deployed on Databricks, offering flexibility to use either a direct client object or automatically create a Databricks client within the integration. While this setup is tailored to Databricks and has only been tested with it, the design is structured to be easily extendable for other cloud providers in the future.

The integration aligns with existing MLflow conventions and allows for setting credentials and endpoints directly or through environment variables.

## Installation

To use this integration, install the following packages:

```bash
pip install llama-index
pip install llama-index-embeddings-mlflow
```

## Usage

### Setting Up MLflow with Databricks API Credentials

You can provide the Databricks MLflow client and model endpoint information directly as arguments or set them up as environment variables for easier access.

### Option 1: Directly Passing a Client and an Endpoint

This method allows you to initialize the embedding class with a specific MLflow client, which is useful for testing different cloud providers.

```python
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.mlflow import MLFlowEmbedding
from mlflow.deployments import get_deploy_client

load_dotenv()

# Create the Databricks MLflow client
client = get_deploy_client("databricks")

# Initialize the MLFlowEmbedding class with model endpoint and Databricks MLflow client
embed_model = MLFlowEmbedding(
    endpoint="databricks-gte-large-en",
    client=client,
    client_name=None,  # Optional when passing a client directly
)
Settings.embed_model = embed_model

# Embed some text
embeddings = embed_model.get_text_embedding(
    "Testing MLflow embedding integration with Databricks."
)
print(embeddings)
```

### Option 2: Using Environment Variables

You can set up environment variables to specify the Databricks host, token, and MLflow endpoint, which simplifies deployment and testing.

#### Environment Variable Setup

Define the following variables:

```bash
export DATABRICKS_HOST="<YOUR_DATABRICKS_HOST>"
export DATABRICKS_TOKEN="<YOUR_DATABRICKS_TOKEN>"
export MLFLOW_ENDPOINT="<YOUR_MLFLOW_MODEL_ENDPOINT>"
```

#### Using Environment Variables in Code

```python
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.mlflow import MLFlowEmbedding

load_dotenv()

# Initialize the MLFlowEmbedding class
embed_model = MLFlowEmbedding(
    endpoint=os.getenv("MLFLOW_ENDPOINT"),
    client=None,  # Client will be created within the class based on environment variables
    client_name="databricks",
)
Settings.embed_model = embed_model

# Embed some text
embeddings = embed_model.get_text_embedding(
    "Testing MLflow embedding integration with Databricks."
)
print(embeddings)
```

### Notes

- **Client Name**: The `client_name` parameter is set to `"databricks"` for Databricks-specific configuration, enabling automatic client creation if a client is not passed directly.
- **Retries and Timeouts**: The integration allows configuring `max_retries` and `timeout` to handle network or service issues with robust retry mechanisms.
- **Extensibility**: This setup is designed to easily adapt for other cloud providers or MLflow-hosted models. By modifying `client_name` and endpoint setup, you can extend support for other platforms or custom MLflow deployments.

### Example Environment Configuration

For easier testing or deployment, you may add the following variables to your `.env` file:

```bash
DATABRICKS_HOST=<YOUR_DATABRICKS_HOST>
DATABRICKS_TOKEN=<YOUR_DATABRICKS_TOKEN>
MLFLOW_ENDPOINT=<YOUR_MLFLOW_MODEL_ENDPOINT>
```
