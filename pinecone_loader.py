"""
This module loads environment variables, initializes OpenAI embeddings,
configures and connects to Pinecone, and then processes text transcripts
from 'prepare_chunks.py' to embed and upload them to a Pinecone vector store.
It handles index creation if the specified index does not already exist.

"""

import os
import logging
import sys
from typing import List

from dotenv import load_dotenv

# Third-party imports
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
# E0611: No name 'ServerlessSpec' in module 'pinecone'
# This error can occur if the 'pinecone-client' library is an older version
# or if Pylint's environment differs from the runtime environment.
# Ensure 'pinecone-client' is updated (pip install --upgrade pinecone-client).
# Temporarily disabling this specific Pylint check if runtime is fine.
from pinecone import Pinecone, ServerlessSpec, PineconeApiException # pylint: disable=no-name-in-module

# Local/Application-specific imports
from prepare_chunks import read_and_chunk_transcripts

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Load environment variables ---
load_dotenv() # This line is crucial for loading .env file

# --- Configuration Constants ---
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_EMBEDDING_DIMENSION = 1536  # Dimension for 'text-embedding-ada-002'
PINECONE_METRIC = 'cosine'          # Cosine similarity is common for text embeddings
PINECONE_CLOUD = "aws"
# Note: PINECONE_ENVIRONMENT is used as the region for ServerlessSpec
INDEX_NAME = "financial-literacy-chatbot" # Your chosen Pinecone index name
TRANSCRIPTS_DIRECTORY = 'transcripts/'
UPLOAD_BATCH_SIZE = 100

def initialize_openai_embeddings() -> OpenAIEmbeddings:
    """
    Initializes and returns an OpenAIEmbeddings instance.
    Ensures the OPENAI_API_KEY environment variable is set.

    Returns:
        OpenAIEmbeddings: An initialized OpenAIEmbeddings model.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
        Exception: For other errors during initialization.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("Configuration Error: OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    try:
        embeddings_model = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=openai_api_key
        )
        logging.info("Initialized OpenAIEmbeddings model with '%s'.", OPENAI_EMBEDDING_MODEL)
        return embeddings_model
    except Exception as err: # pylint: disable=broad-except
        # Catching broad exception here because OpenAIEmbeddings can raise various
        # exceptions depending on configuration or network issues.
        logging.error("Error initializing OpenAIEmbeddings: %s", err)
        raise # Re-raise the exception after logging

def initialize_pinecone_client() -> Pinecone:
    """
    Initializes and returns a Pinecone client instance.
    Ensures PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables are set.

    Returns:
        Pinecone: An initialized Pinecone client.

    Raises:
        ValueError: If Pinecone API key or environment variables are not set.
        Exception: For other errors during initialization.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

    if not pinecone_api_key or not pinecone_environment:
        logging.error(
            "Pinecone API key or environment not set. "
            "Please add PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file."
        )
        raise ValueError("Pinecone credentials missing.")

    try:
        pinecone_client = Pinecone(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        logging.info("Initialized Pinecone client for environment: %s.", pinecone_environment)
        return pinecone_client
    except Exception as err: # pylint: disable=broad-except
        # Catching broad exception here as Pinecone client initialization can fail
        # due to various network or credential issues.
        logging.error("Error initializing Pinecone client: %s", err)
        raise # Re-raise the exception

def check_and_create_pinecone_index(
    pinecone_client: Pinecone,
    index_name: str,
    index_params: dict # Grouping dimension, metric, cloud, region
) -> None:
    """
    Checks if a Pinecone index exists and creates it if it does not.

    Args:
        pinecone_client (Pinecone): The initialized Pinecone client.
        index_name (str): The name of the Pinecone index.
        index_params (dict): A dictionary containing 'dimension', 'metric',
                             'cloud', and 'region' for index creation.

    Raises:
        PineconeApiException: If there's an API error during index creation.
        Exception: For other unexpected errors.
    """
    logging.info("Checking for Pinecone index '%s'...", index_name)
    try:
        existing_indexes = pinecone_client.list_indexes()
        is_index_existing = False

        for idx_info in existing_indexes:
            # Handle both dict and object types for index info
            if (isinstance(idx_info, dict) and idx_info.get('name') == index_name) or \
               (hasattr(idx_info, 'name') and idx_info.name == index_name):
                is_index_existing = True
                break

        if not is_index_existing:
            logging.info(
                "Pinecone index '%s' does not exist. Attempting to create it "
                "with ServerlessSpec (cloud='%s', region='%s').",
                index_name, index_params['cloud'], index_params['region']
            )
            try:
                pinecone_client.create_index(
                    name=index_name,
                    dimension=index_params['dimension'],
                    metric=index_params['metric'],
                    spec=ServerlessSpec(
                        cloud=index_params['cloud'],
                        region=index_params['region']
                    )
                )
                logging.info("Pinecone index '%s' created successfully.", index_name)
            except PineconeApiException as err:
                if err.status == 409:  # HTTP 409 Conflict indicates index already exists
                    logging.warning(
                        "Pinecone index '%s' already exists (caught 409 Conflict during "
                        "create_index). Proceeding to connect.",
                        index_name
                    )
                else:
                    logging.error("Failed to create Pinecone index '%s': %s", index_name, err)
                    raise # Re-raise if it's another API error
        else:
            logging.info("Pinecone index '%s' already exists.", index_name)

    except Exception as err: # pylint: disable=broad-except
        logging.error("Error during Pinecone index existence check or creation: %s", err)
        raise # Re-raise the exception

def upload_embeddings_to_pinecone(
    pinecone_client: Pinecone,
    embeddings_model: OpenAIEmbeddings,
    chunks_to_upload: List[Document],
    index_name: str,
    batch_size: int
) -> None:
    """
    Uploads document embeddings to the specified Pinecone index in batches.

    Args:
        pinecone_client (Pinecone): The initialized Pinecone client.
        embeddings_model (OpenAIEmbeddings): The embeddings model to use.
        chunks_to_upload (List[Document]): A list of LangChain Document objects to embed and upload.
        index_name (str): The name of the Pinecone index.
        batch_size (int): The number of documents to upload in each batch.
    """
    try:
        index = pinecone_client.Index(index_name)
        current_vector_count = index.describe_index_stats().total_vector_count
        logging.info("Pinecone index '%s' currently contains %d vectors.",
                     index_name, current_vector_count)

        if current_vector_count == 0:
            logging.info("Pinecone index is empty. Proceeding with initial embedding upload.")
            total_batches = (len(chunks_to_upload) + batch_size - 1) // batch_size
            for i in range(0, len(chunks_to_upload), batch_size):
                batch = chunks_to_upload[i:i + batch_size]
                logging.info(
                    "Processing batch %d/%d (%d documents)...",
                    i // batch_size + 1, total_batches, len(batch)
                )
                PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings_model,
                    index_name=index_name
                )
                logging.info("Uploaded batch starting with document %d to Pinecone.", i)

            final_vector_count = index.describe_index_stats().total_vector_count
            logging.info(
                "Finished uploading all embeddings to Pinecone index '%s'. "
                "Total vectors now: %d",
                index_name, final_vector_count
            )
        else:
            logging.info(
                "Pinecone index '%s' already contains %d vectors. "
                "Skipping embedding upload to avoid duplication.",
                index_name, current_vector_count
            )
    except Exception as err: # pylint: disable=broad-except
        logging.error("Error during embedding upload to Pinecone: %s", err, exc_info=True)
        raise # Re-raise the exception

def main():
    """
    Main function to orchestrate the process of initializing clients,
    loading data, and uploading embeddings to Pinecone.
    """
    # Initialize OpenAI Embeddings
    try:
        embeddings_model = initialize_openai_embeddings()
    except (ValueError, Exception) as err: # pylint: disable=broad-except
        logging.critical("Exiting: Failed to initialize OpenAI Embeddings. Reason: %s", err)
        sys.exit(1)

    # Initialize Pinecone Client
    try:
        pinecone_client = initialize_pinecone_client()
    except (ValueError, Exception) as err: # pylint: disable=broad-except
        logging.critical("Exiting: Failed to initialize Pinecone client. Reason: %s", err)
        sys.exit(1)

    # Data Loading, Cleaning, and Chunking (using prepare_chunks.py)
    chunks_data: List[Document] = []
    try:
        chunks_data = read_and_chunk_transcripts(TRANSCRIPTS_DIRECTORY)
        if not chunks_data:
            logging.warning(
                "No chunks read from '%s'. Ensure files exist and content is "
                "processed by prepare_chunks.py.", TRANSCRIPTS_DIRECTORY
            )
            logging.critical("Exiting: No transcript chunks found for embedding.")
            sys.exit(1)
        logging.info(
            "Successfully loaded %d LangChain Documents (with metadata) from transcripts.",
            len(chunks_data)
        )
    except Exception as err: # pylint: disable=broad-except
        logging.critical(
            "Exiting: Failed to process transcripts. Check 'prepare_chunks.py' and "
            "'%s' directory for issues. Reason: %s", TRANSCRIPTS_DIRECTORY, err
        )
        sys.exit(1)

    # Pinecone Index Setup and Embedding
    logging.info("Starting Pinecone data loading process...")
    try:
        # Group index creation parameters into a dictionary
        pinecone_index_creation_params = {
            "dimension": OPENAI_EMBEDDING_DIMENSION,
            "metric": PINECONE_METRIC,
            "cloud": PINECONE_CLOUD,
            "region": os.getenv("PINECONE_ENVIRONMENT")
        }

        # Check and create index if necessary
        check_and_create_pinecone_index(
            pinecone_client,
            INDEX_NAME,
            pinecone_index_creation_params # Pass the dictionary
        )

        # Upload embeddings if the index was empty
        upload_embeddings_to_pinecone(
            pinecone_client,
            embeddings_model,
            chunks_data,
            INDEX_NAME,
            UPLOAD_BATCH_SIZE
        )
        logging.info("Pinecone data loading process complete. Index is ready for queries.")

    except Exception as err: # pylint: disable=broad-except
        logging.critical(
            "Exiting: Critical error during Pinecone index setup and data loading: %s",
            err, exc_info=True
        )
        sys.exit(1)

if __name__ == "__main__":
    main()