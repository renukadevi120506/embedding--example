from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Text to convert into vectors
sentences = [
    "I love learning Python.",
    "Python programming is interesting.",
    "I enjoy cooking food."
]

# Generate embeddings
embeddings = model.encode(sentences)

# Display results
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Vector dimensions:", len(embedding))
    print("First five values:", embedding[:5])
    print()