import os
from datasets import load_dataset
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import nest_asyncio
# Load environment variables
load_dotenv("/Users/oksanak/Projects/.env")

# Initialize embed model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load and save the dataset
dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")
Path("data").mkdir(parents=True, exist_ok=True)
for i, persona in enumerate(dataset):
    with open(Path("data") / f"persona_{i}.txt", "w") as f:
        f.write(persona["persona"])

reader = SimpleDirectoryReader(input_dir="data")
documents = reader.load_data()
# print(f"Number of documents: {len(documents)}")
# print("First document content:", documents[0].text[:200] if documents else "No documents found")


db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection(name="alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create the pipeline with SentenceSplitter
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50)
    ],
    vector_store=vector_store
)

# Process the documents
nodes = pipeline.run(documents=documents)
# print(f"Number of nodes created: {len(nodes)}")
# print("First node content:", nodes[0].text[:200] if nodes else "No nodes created")

# Create index directly from nodes
index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)

llm = Groq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=500,
    api_key=os.getenv("GROQ_API_KEY")
)

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="refine",
    similarity_top_k=5,
    streaming=False
)

# Debug: Print the retrieved nodes
# retriever = index.as_retriever(similarity_top_k=5)
# retrieved_nodes = retriever.retrieve("What is the name of the someone that is interested in AI and technology?")
# print("\nRetrieved nodes:")
# for node in retrieved_nodes:
#     print(f"Score: {node.score}")
#     print(f"Content: {node.text[:200]}")

print("\nQuerying...")
response = query_engine.query(
    "Respond using a persona that describes author and travel experiences?"
)
print("\nResponse:")
print(response)