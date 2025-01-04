import chromadb 
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv(override=True)

# Setting up file paths for data and ChromaDB database
DATA_PATH = r"data"  # Path to the directory containing data
CHROMA_PATH = r"chroma_db"  # Path to the ChromaDB database directory

# Initialize a persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or retrieve a collection named "rutgers_ece_handbook" in ChromaDB
collection = chroma_client.get_or_create_collection(name="rutgers_ece_handbook")

# Prompt the user for their query about the ECE Graduate program at Rutgers
user_query = input("What do you want to know about the ECE Graduate program at Rutgers University? \n\n")

# Query the ChromaDB collection for relevant information based on the user's input
results = collection.query(
    query_texts=[user_query],  # The query text provided by the user
    n_results=1  # Number of results to return
)

# Initialize the OpenAI client for generating responses
client = OpenAI()

# Construct the system prompt for the AI model
system_prompt = """
You are a helpful assistant. You answer questions about the Electrical and Computer Engineering Graduate program at Rutgers University. 
You provide answers based on the knowledge provided to you. DO NOT make things up.
If you don't know the answer, just say: I don't know the answer.
--------------------
The data:
"""+str(results['documents'])+"""
"""

#print(system_prompt)

response = client.chat.completions.create(
    model="gpt-4o",
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_query}    
    ]
)



print("\n\n---------------------\n\n")

print(response.choices[0].message.content)
