import os
import re
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    """
    Cleans and normalizes the input text by:
    - Lowercasing the text.
    - Removing timestamps (e.g., [00:01:23]).
    - Removing common filler words or transcript artifacts.
    - Stripping leading/trailing whitespace.
    - Replacing multiple spaces with a single space.
    """
    text = text.lower() # Lowercase the text
    
    # Remove timestamps like [00:01:23] or (00:01:23)
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\[music\]|\[applause\]|\(\d{2}:\d{2}:\d{2}\)|\(music\)|\(applause\)', '', text)
    
    # Remove common filler words or transcription errors
    filler_words = [
        "uh", "um", "ah", "oh", "like", "you know", "kind of", "sort of",
        "hmm", "mhm", "yeah", "right", "okay", "so", "well", "actually",
        "basically", "literally", "totally", "definitely", "really", "just",
        "i mean", "you see", "i guess", "alright", "all right", "and so",
        "but uh", "and uh", "that's uh", "if you will", "you got it"
    ]
    for filler in filler_words:
        text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text) # Remove whole words

    # Remove special characters that are not letters, numbers, or basic punctuation
    # Keep standard punctuation for sentence splitting: .?!,;:'"()
    text = re.sub(r'[^a-z0-9\s.,?!;:\'"()-]', '', text)

    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space and strip whitespace
    return text

def read_and_chunk_transcripts(transcripts_dir: str = 'transcripts/',
                                chunk_size: int = 500,
                                chunk_overlap: int = 100) -> list[str]:
    """
    Reads all text files from a specified directory, cleans them, and then
    chunks the combined text using LangChain's RecursiveCharacterTextSplitter
    with specified chunk size and overlap.

    Args:
        transcripts_dir (str): The directory containing transcript text files.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        list[str]: A list of cleaned and chunked text segments.
    """
    all_text = ""
    file_count = 0
    
    # Use glob to find all .txt files in the directory
    transcript_files = glob.glob(os.path.join(transcripts_dir, "*.txt"))

    if not transcript_files:
        logging.warning(f"No .txt files found in directory: {transcripts_dir}")
        return []

    for filepath in transcript_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                cleaned_content = clean_text(content) # Clean each file's content
                all_text += cleaned_content + "\n\n" # Add double newline between files to help splitter
            file_count += 1
        except Exception as e:
            logging.error(f"Error reading or cleaning file {filepath}: {e}")
            continue

    if not all_text.strip():
        logging.warning("All transcript files were empty or contained only unparseable content after cleaning.")
        return []

    logging.info(f"Successfully read and cleaned {file_count} transcript files.")

    # Initialize LangChain's RecursiveCharacterTextSplitter
    # This splitter attempts to split by paragraphs, then sentences, then words,
    # ensuring semantic coherence where possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False, # Use standard separators by default
    )

    logging.info(f"Splitting text into chunks with size {chunk_size} and overlap {chunk_overlap}...")
    chunks = text_splitter.split_text(all_text)
    
    # Changed output to match previous format for consistency
    print(f"✅ Loaded {len(chunks)} chunks from transcripts.")
    
    return chunks

# Example usage (for testing this file independently)
if __name__ == "__main__":
    # Create a dummy transcripts directory and file for testing
    dummy_dir = 'test_transcripts'
    os.makedirs(dummy_dir, exist_ok=True)
    
    with open(os.path.join(dummy_dir, 'test1.txt'), 'w', encoding='utf-8') as f:
        f.write("This is a test transcript. [00:00:05] It talks about finance. Uh, investment planning is important. (music) You know, save money! Right?")
    
    with open(os.path.join(dummy_dir, 'test2.txt'), 'w', encoding='utf-8') as f:
        f.write("Another transcript. [00:01:10] Budgeting is crucial for financial health. So, don't forget to track expenses. Yeah.")

    print("--- Running prepare_chunks.py independently for testing ---")
    
    # Call the function with default parameters
    processed_chunks = read_and_chunk_transcripts(transcripts_dir=dummy_dir)
    
    if processed_chunks:
        for i, chunk in enumerate(processed_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk)
        print(f"\n📦 Total chunks generated: {len(processed_chunks)}")
    else:
        print("No chunks generated. Check logs for warnings/errors.")
        
    # Clean up dummy directory
    # import shutil
    # shutil.rmtree(dummy_dir)
    # print(f"\nCleaned up {dummy_dir} directory.")
    
    print(processed_chunks)
