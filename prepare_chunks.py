import os
import re
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import logging

# Configure logging to show informational messages
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
    
    # Remove timestamps like [00:01:23] or (00:01:23), and common transcript markers
    text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\[music\]|\[applause\]|\(\d{2}:\d{2}:\d{2}\)|\(music\)|\[laughter\]|\(applause\)', '', text)
    
    # Remove common filler words or transcription errors to clean up text content
    filler_words = [
        "uh", "um", "ah", "oh", "like", "you know", "kind of", "sort of",
        "hmm", "mhm", "yeah", "right", "okay", "so", "well", "actually",
        "basically", "literally", "totally", "definitely", "really", "just",
        "i mean", "you see", "i guess", "alright", "all right", "and so",
        "but uh", "and uh", "that's uh", "if you will", "you got it"
    ]
    for filler in filler_words:
        # Use word boundaries (\b) to ensure whole words are removed
        text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text) 

    # Remove special characters that are not letters, numbers, or basic punctuation.
    # Keep standard punctuation for sentence splitting: .?!,;:'"()-
    text = re.sub(r'[^a-z0-9\s.,?!;:\'"()-]', '', text)

    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def read_and_chunk_transcripts(transcripts_dir: str = 'transcripts/',
                                chunk_size: int = 500,
                                chunk_overlap: int = 100) -> list[Document]:
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
    all_documents = [] # This list will store all the LangChain Document objects

    # Construct the path to find all .txt files within the specified directory
    transcript_files = glob.glob(os.path.join(transcripts_dir, "*.txt"))

    # Check if any transcript files were found
    if not transcript_files:
        logging.warning(f"No .txt files found in directory: {transcripts_dir}. Please ensure your transcripts are in this directory and are saved as .txt files.")
        return []

    # Initialize LangChain's RecursiveCharacterTextSplitter.
    # This splitter is robust for various text types, attempting to split by
    # paragraphs, then sentences, then words to maintain semantic coherence.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use standard Python len() for length calculation
        is_separator_regex=False, # Use standard separators (e.g., newlines, spaces)
    )

    logging.info(f"Starting to process transcript files from: {transcripts_dir}")
    for filepath in transcript_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                cleaned_content = clean_text(content) # Clean the raw content of the file
                
                # If content is empty after cleaning, log a warning and skip to the next file
                if not cleaned_content.strip():
                    logging.warning(f"File {filepath} was empty or contained only unparseable content after cleaning. Skipping this file.")
                    continue

                # Extract just the filename (e.g., "video1.txt") from the full path
                filename = os.path.basename(filepath)
                
                # Create a single LangChain Document object for the entire cleaned content of the current file.
                # This initial document carries the source metadata.
                doc_for_file = Document(page_content=cleaned_content, metadata={"source": filename})
                
                # Split this single document into smaller chunks.
                # The `split_documents` method automatically propagates the metadata
                # from the parent document to the new smaller chunks.
                chunks_from_file = text_splitter.split_documents([doc_for_file])
                
                # Add the newly generated chunks from this file to our overall list
                all_documents.extend(chunks_from_file)
                logging.info(f"Successfully processed and chunked file: {filename}. Generated {len(chunks_from_file)} chunks.")
                
        except Exception as e:
            # Log any errors encountered while reading, cleaning, or chunking a file
            logging.error(f"Error processing file {filepath}: {e}")
            continue # Continue to the next file even if one fails

    # Final check: if no documents were generated at all
    if not all_documents:
        logging.warning("No documents were generated in total. This might be due to empty files, unparseable content, or no .txt files in the directory.")
        return []

    # Print a summary of the total chunks generated
    print(f"✅ Successfully loaded and processed {len(all_documents)} chunks from transcripts in '{transcripts_dir}'.")
    
    return all_documents

# --- Main execution block ---
# This part runs when the script is executed directly.
# It calls the function to process transcripts from the default 'transcripts/' directory
# and prints the total number of chunks.

if __name__ == "__main__":
    # Ensure the default transcripts directory exists for the script to run without error
    # You should place your actual .txt transcript files in this directory.
    default_transcripts_dir = 'transcripts'
    os.makedirs(default_transcripts_dir, exist_ok=True)
    logging.info(f"Using '{default_transcripts_dir}' as the default transcripts directory.")

    # Call the function to read and chunk all transcripts found in the default directory
    # You can adjust chunk_size and chunk_overlap as needed for your specific data.
    processed_chunks = read_and_chunk_transcripts(
        transcripts_dir=default_transcripts_dir,
        chunk_size=500,  # Example: Adjust as per your content and LLM context window
        chunk_overlap=100 # Example: Adjust for better context preservation
    )

    # The function itself already prints the total count, but you can add more
    # specific actions here if needed, like saving the chunks to a vector store.
    
    # Example of how you might inspect the first few chunks (for verification)
    # if processed_chunks:
    #     print("\n--- First 3 Chunks (for verification) ---")
    #     for i, chunk in enumerate(processed_chunks[:3]):
    #         print(f"Chunk {i+1} (Source: {chunk.metadata.get('source', 'N/A')}):")
    #         print(f"  Content: {chunk.page_content[:200]}...") # Print first 200 chars
    #     if len(processed_chunks) > 3:
    #         print(f"...and {len(processed_chunks) - 3} more chunks.")
    # else:
    #     print("No chunks were generated. Please check the 'transcripts/' directory and logs for errors.")

