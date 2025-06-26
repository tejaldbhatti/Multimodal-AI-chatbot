import os
import glob

CHUNK_SIZE = 500

def read_and_chunk_transcripts(folder_path="transcripts"):
    all_chunks = []

    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        for i in range(0, len(content), CHUNK_SIZE):
            chunk = content[i:i+CHUNK_SIZE]
            all_chunks.append(chunk)

    print(f"✅ Loaded {len(all_chunks)} chunks from transcripts.")
    return all_chunks

# Call the function (you can use either 'transcripts' or 'transcripts/')
chunks = read_and_chunk_transcripts('transcripts/')
