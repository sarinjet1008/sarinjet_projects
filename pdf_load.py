from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Setting up environment variables for file paths
DATA_PATH = r"data"  # Path to the directory containing PDF files
CHROMA_PATH = r"chroma_db"  # Path to the ChromaDB database directory

# Initialize a persistent ChromaDB client to store and manage data
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or get a collection named "rutgers_ece_handbook" in ChromaDB
collection = chroma_client.get_or_create_collection(name="rutgers_ece_handbook")

# Load PDF documents from the specified directory
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()  # Load all PDF files into a list of document objects

# Configure the text splitter for breaking down documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum number of characters in each chunk
    chunk_overlap=200,      # Overlap between consecutive chunks to preserve context
    length_function=len,    # Function to calculate the length of text
    is_separator_regex=False  # Indicates whether the separator is a regex
)

# Split the raw documents into manageable chunks
chunks = text_splitter.split_documents(raw_documents)

# Prepare data for insertion into ChromaDB
# Initialize lists to hold document content, metadata, and unique IDs
documents = []
metadata = []
ids = []

# Iterate through chunks and populate the lists with relevant data
for i, chunk in enumerate(chunks):
    documents.append(chunk.page_content)  # Extract the text content of the chunk
    ids.append(f"ID{i}")  # Assign a unique ID for each chunk
    metadata.append(chunk.metadata)  # Extract metadata associated with the chunk

# Add the prepared data into the ChromaDB collection
collection.upsert(
    documents=documents,  # List of document content
    metadatas=metadata,   # List of metadata associated with each document
    ids=ids               # List of unique IDs for the documents
)

# The data is now stored in ChromaDB and can be queried or processed further.
