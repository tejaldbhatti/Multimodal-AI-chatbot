"""
This module provides functions for reading, cleaning, and chunking text
transcripts. It's designed to preprocess textual data for use in applications
like retrieval-augmented generation (RAG) systems.
"""
import os
import re
import glob
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Configure logging (broken into multiple lines to avoid C0301)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants for default values and common patterns
DEFAULT_TRANSCRIPTS_DIR = 'transcripts'
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# Regex patterns for cleaning
TIMESTAMP_PATTERN = r'\[\d{2}:\d{2}:\d{2}\]|\(\d{2}:\d{2}:\d{2}\)'
COMMON_ARTIFACTS_PATTERN = r'\[music\]|\[applause\]|\(music\)|\(applause\)'
SPECIAL_CHARS_PATTERN = r'[^a-z0-9\s.,?!;:\'"()-]'
MULTIPLE_SPACES_PATTERN = r'\s+'

# Common filler words to remove (expanded for better coverage)
# Split into multiple lines for C0301 (line-too-long)
FILLER_WORDS = [
    "uh", "um", "ah", "oh", "like", "you know", "kind of", "sort of",
    "hmm", "mhm", "yeah", "right", "okay", "so", "well", "actually",
    "basically", "literally", "totally", "definitely", "really", "just",
    "i mean", "you see", "i guess", "alright", "all right", "and so",
    "but uh", "and uh", "that's uh", "if you will", "you got it",
    "you know what i mean", "at the end of the day", "believe it or not",
    "to be honest", "in my opinion", "you know what", "whatnot"
]

def clean_text(text: str) -> str:
    """
    Cleans and normalizes the input text by:
    - Lowercasing the text.
    - Removing timestamps (e.g., [00:01:23] or (00:01:23)).
    - Removing common transcript artifacts like [music], [applause].
    - Removing common filler words.
    - Removing special characters, keeping only alphanumeric, spaces, and basic punctuation.
    - Replacing multiple spaces with a single space.
    - Stripping leading/trailing whitespace.

    Args:
        text (str): The input string to be cleaned.

    Returns:
        str: The cleaned and normalized string.
    """
    text = text.lower()

    # Remove timestamps and common transcript markers
    text = re.sub(TIMESTAMP_PATTERN, '', text)
    text = re.sub(COMMON_ARTIFACTS_PATTERN, '', text)

    # Remove common filler words
    # Sort filler words by length in descending order to handle longer phrases first
    sorted_filler_words = sorted(FILLER_WORDS, key=len, reverse=True)
    for filler in sorted_filler_words:
        # Use word boundaries (\b) to ensure whole words are removed
        text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text)

    # Remove special characters that are not letters, numbers, or basic punctuation.
    text = re.sub(SPECIAL_CHARS_PATTERN, '', text)

    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(MULTIPLE_SPACES_PATTERN, ' ', text).strip()
    return text

def read_and_chunk_transcripts(transcripts_dir: str = DEFAULT_TRANSCRIPTS_DIR,
                               chunk_size: int = DEFAULT_CHUNK_SIZE,
                               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Document]:
    """
    Reads all text files from a specified directory, cleans them, and then
    chunks the content using LangChain's RecursiveCharacterTextSplitter.
    Each generated chunk will include metadata about its original source file,
    which is crucial for attributing information back to its origin in a RAG system.

    Args:
        transcripts_dir (str): The directory containing transcript text files.
                                Defaults to 'transcripts/'.
        chunk_size (int): The maximum size (in characters) of each text chunk.
                            Defaults to 500.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.
                                This helps maintain context across chunk boundaries. Defaults to 100.

    Returns:
        list[Document]: A list of cleaned and chunked text segments. Each segment
                        is a LangChain Document object containing `page_content` (the text)
                        and `metadata` (a dictionary, typically including 'source' filename).
    """
    all_documents: list[Document] = []

    # Construct the path to find all .txt files within the specified directory
    transcript_files = glob.glob(os.path.join(transcripts_dir, "*.txt"))

    # Check if any transcript files were found
    if not transcript_files:
        logging.warning(
            "No .txt files found in directory: %s. "
            "Please ensure your transcripts are in this directory and are "
            "saved as .txt files.",
            transcripts_dir
        )
        return []

    # Initialize LangChain's RecursiveCharacterTextSplitter.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    logging.info("Starting to process transcript files from: %s", transcripts_dir)
    for filepath in transcript_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                cleaned_content = clean_text(content)

                if not cleaned_content.strip():
                    logging.warning(
                        "File %s was empty or contained only unparseable content "
                        "after cleaning. Skipping this file.",
                        filepath
                    )
                    continue

                filename = os.path.basename(filepath)
                doc_for_file = Document(page_content=cleaned_content, metadata={"source": filename})

                chunks_from_file = text_splitter.split_documents([doc_for_file])

                all_documents.extend(chunks_from_file)
                logging.info("Successfully processed and chunked file: %s. "
                             "Generated %d chunks.", filename, len(chunks_from_file))

        except (IOError, OSError) as err:
            # Catch specific I/O related errors
            logging.error("Error accessing or reading file %s: %s", filepath, err)
            continue
        except Exception as err: # pylint: disable=broad-except
            # Catch any other unexpected errors during processing a file.
            # broad-except is justified here to ensure robust file processing
            # by catching unforeseen issues and continuing, rather than crashing.
            logging.error("An unexpected error occurred while processing file %s: %s",
                          filepath, err)
            continue

    if not all_documents:
        logging.warning("No documents were generated in total. "
                        "This might be due to empty files, unparseable content, "
                        "or no .txt files in the directory.")
        return []

    # Broken into multiple lines to avoid C0301
    print(f"✅ Successfully loaded and processed {len(all_documents)} chunks "
          f"from transcripts in '{transcripts_dir}'.")

    return all_documents

if __name__ == "__main__":
    # Ensure the default transcripts directory exists
    os.makedirs(DEFAULT_TRANSCRIPTS_DIR, exist_ok=True)
    logging.info("Using '%s' as the default transcripts directory.",
                 DEFAULT_TRANSCRIPTS_DIR)

    processed_chunks = read_and_chunk_transcripts(
        transcripts_dir=DEFAULT_TRANSCRIPTS_DIR,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )