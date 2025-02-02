import PyPDF2
import json

def extract_text_from_pdf(pdf_path):
    print("Extracting text from PDF...")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, max_chunk_size=3500, overlap=50):
    print("Chunking text...")
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # Add overlap
            current_chunk.append(word)
            current_length = sum(len(w) + 1 for w in current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def save_chunks_to_file(chunks, output_file):
    print(f"Saving chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, indent=4)  # Save as JSON for readability

def print_part_of_chunks(chunks, num_chunks_to_print=5):
    print(f"Printing the first {num_chunks_to_print} chunks:")
    for i, chunk in enumerate(chunks[:num_chunks_to_print]):
        print(f"Chunk {i+1}:")
        print(chunk[:200] + "...")  # Print first 200 characters of each chunk
        print("-" * 50)

# Path to the PDF file
pdf_path = "D:/pro/economic_survey/Economic_survey_2024-25.pdf"

# Extract text from PDF
text = extract_text_from_pdf(pdf_path)

# Chunk the text
chunks = chunk_text(text, max_chunk_size=1024, overlap=100)
print(f"Total chunks: {len(chunks)}")

# Save chunks to a file
output_file = "chunks.json"
save_chunks_to_file(chunks, output_file)