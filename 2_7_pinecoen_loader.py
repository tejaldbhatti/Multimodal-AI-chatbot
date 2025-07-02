from dotenv import load_dotenv
load_dotenv() # This line is crucial for loading .env file

import os
import logging
from typing import List

# prepare_chunks.py is a separate file that handles reading, cleaning, and chunking transcripts.
# It now returns LangChain Document objects with metadata.
from prepare_chunks import read_and_chunk_transcripts

# LangChain Imports
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document # Ensure Document is imported for type hinting and clarity

# Pinecone Imports
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import PineconeApiException

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize OpenAI client (for embeddings) ---
try:
    # It's good practice to ensure the API key is accessible and valid early.
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    # OpenAIEmbeddings will use this key, no need to explicitly import openai and set its api_key here
    # unless you are making direct calls to the openai client for other purposes.
    # For LangChain components, passing it directly or relying on environment variable is sufficient.
    logging.info("OpenAI API key found.")
except ValueError as e:
    logging.error(f"Configuration Error: {e}")
    exit("Exiting: OpenAI API key is missing. Please set OPENAI_API_KEY environment variable.")
except Exception as e:
    logging.error(f"Error during OpenAI API key check: {e}")
    exit("Exiting: Failed to verify OpenAI API key.")


# --- Initialize LangChain's OpenAIEmbeddings ---
try:
    # The openai_api_key parameter ensures the key is passed explicitly to the embeddings model.
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    logging.info("Initialized OpenAIEmbeddings model with 'text-embedding-ada-002'.")
except Exception as e:
    logging.error(f"Error initializing OpenAIEmbeddings: {e}")
    exit("Exiting: Failed to initialize OpenAIEmbeddings. Check API key and network connectivity.")

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "us-east-1" or "gcp-starter"
INDEX_NAME = "financial-literacy-chatbot" # Your chosen Pinecone index name

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logging.error("Pinecone API key or environment not set. Please add PINECONE_API_KEY and PINECONE_ENVIRONMENT to your .env file.")
    exit("Exiting: Pinecone credentials missing.")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    logging.info("Initialized Pinecone client.")
except Exception as e:
    logging.error(f"Error initializing Pinecone client: {e}")
    exit("Exiting: Failed to initialize Pinecone. Check API key and environment. Ensure your environment matches your Pinecone project settings.")

# --- Data Loading, Cleaning, and Chunking (using prepare_chunks.py) ---
chunks: List[Document] = [] # Initialize chunks list as list of LangChain Document objects
try:
    # CRUCIAL CHANGE HERE: read_and_chunk_transcripts already returns Document objects with metadata.
    # We directly assign its output to 'chunks' to preserve the metadata.
    chunks = read_and_chunk_transcripts('transcripts/') 
    
    if not chunks:
        logging.warning("No chunks read from 'transcripts/'. Ensure files exist and content is processed by prepare_chunks.py.")
        exit("Exiting: No transcript chunks found for embedding.")

    logging.info(f"Successfully loaded {len(chunks)} LangChain Documents (with metadata) from transcripts.")
except Exception as e:
    logging.error(f"Error reading, cleaning, or chunking transcripts: {e}")
    exit("Exiting: Failed to process transcripts. Check 'prepare_chunks.py' and 'transcripts/' directory for issues.")

# --- Pinecone Index Setup and Embedding ---
if __name__ == "__main__":
    logging.info("Starting Pinecone data loading process...")
    try:
        # Check if index exists by listing all indexes
        existing_indexes = pc.list_indexes()
        
        index_exists = False
        # Iterate through the list of index descriptions to find if our index name exists
        for idx_info in existing_indexes:
            # Pinecone client can return different types for index info, handle both dict and object
            if isinstance(idx_info, dict) and idx_info.get('name') == INDEX_NAME:
                index_exists = True
                break
            elif hasattr(idx_info, 'name') and idx_info.name == INDEX_NAME:
                index_exists = True
                break

        if not index_exists:
            logging.info(f"Pinecone index '{INDEX_NAME}' does not exist. Attempting to create it with ServerlessSpec (cloud='aws', region='{PINECONE_ENVIRONMENT}').")
            try:
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=1536, # Dimension for text-embedding-ada-002
                    metric='cosine', # Cosine similarity is common for text embeddings
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
                )
                logging.info(f"Pinecone index '{INDEX_NAME}' created successfully.")
                index_exists = True # Mark as existing after successful creation
            except PineconeApiException as e_create_api:
                if e_create_api.status == 409: # HTTP 409 Conflict indicates index already exists
                    logging.warning(f"Pinecone index '{INDEX_NAME}' already exists (caught 409 Conflict during create_index). Proceeding to connect.")
                    index_exists = True # Mark as existing
                else:
                    logging.error(f"Failed to create Pinecone index '{INDEX_NAME}': {e_create_api}")
                    raise e_create_api # Re-raise if it's another API error

        if index_exists:
            logging.info(f"Connecting to Pinecone index: {INDEX_NAME}.")
            index = pc.Index(INDEX_NAME)

            # Check if the index is empty. If so, upload embeddings.
            # This prevents re-uploading all data every time the script runs if data is already there.
            current_vector_count = index.describe_index_stats().total_vector_count
            logging.info(f"Pinecone index '{INDEX_NAME}' currently contains {current_vector_count} vectors.")
            
            if current_vector_count == 0:
                logging.info("Pinecone index is empty. Proceeding with initial embedding upload.")
                BATCH_SIZE = 100 # Adjust batch size based on your data volume and Pinecone limits
                
                # Iterate through chunks in batches and upload them to Pinecone
                # PineconeVectorStore.from_documents handles embedding and uploading,
                # and importantly, it preserves the metadata of each Document.
                for i in range(0_0, len(chunks), BATCH_SIZE):
                    batch = chunks[i:i + BATCH_SIZE]
                    logging.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} documents)...")
                    PineconeVectorStore.from_documents(
                        documents=batch,
                        embedding=embeddings_model,
                        index_name=INDEX_NAME # Specify the index name to upload to
                    )
                    logging.info(f"Uploaded batch starting with document {i} to Pinecone.")
                
                # After all batches are uploaded, get the final count
                final_vector_count = index.describe_index_stats().total_vector_count
                logging.info(f"Finished uploading all embeddings to Pinecone index '{INDEX_NAME}'. Total vectors now: {final_vector_count}")
            else:
                logging.info(f"Pinecone index '{INDEX_NAME}' already contains {current_vector_count} vectors. Skipping embedding upload to avoid duplication.")
            
            logging.info("Pinecone data loading process complete. Index is ready for queries.")

        else:
            logging.error(f"Pinecone index '{INDEX_NAME}' could not be created or connected to. Please check your Pinecone dashboard.")
            exit("Exiting: Pinecone index setup failed.")

    except Exception as e:
        logging.error(f"Critical error during Pinecone index setup and data loading: {e}", exc_info=True) # exc_info=True prints traceback
        exit("Exiting: Failed to set up Pinecone index or load data. Ensure network connectivity, correct API key/environment, and valid Pinecone region (e.g., 'us-west-2').")

# This print statement is for debugging and can be removed in production.
# It's good to avoid printing sensitive info like full API keys.
# print("API Key (start):", os.getenv("PINECONE_API_KEY")[:8], "...")
# print("Environment:", os.getenv("PINECONE_ENVIRONMENT"))
# print("Available Indexes:", pc.list_indexes()) # This can be useful for verification
