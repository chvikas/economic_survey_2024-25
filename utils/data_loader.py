import json
from langchain.docstore.document import Document

def load_and_split_data(json_file_path):
    """Loads data from JSON, converts to Documents. No splitting needed now."""
    with open(json_file_path, 'r') as f:
      data = json.load(f)

    # Directly use 'chunk' (which is a string) as page_content
    # No further splitting needed as chunks are already in chunks.json
    docs = [Document(page_content=chunk) for chunk in data]
    return docs

if __name__ == '__main__':
    # Example usage
    file_path = '../chunks.json'  # Relative path to chunks.json
    splits = load_and_split_data(file_path)
    print(f"Number of document splits (now just number of chunks loaded): {len(splits)}")
    for doc in splits[:2]:
      print(f'Content: {doc.page_content[:50]}...\n') # No Header metadata now