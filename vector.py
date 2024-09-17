import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from decouple import config

# Configurations
os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')


storage = "./vector_store"
# Load documents and build index
documents = SimpleDirectoryReader(
    "data"
).load_data()

# Store the date in the vector store 
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=storage)

# Load the data from the vector store
storage_context = StorageContext.from_defaults(persist_dir=storage)
index = load_index_from_storage(storage_context)

# Chat with the data
chat_engine = index.as_chat_engine()
response = chat_engine.chat("What happened in 1962")
print(response)