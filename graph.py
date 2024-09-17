import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# from decouple import config
# os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

# Configurations
api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=api_key)

load_dotenv()

storage_dir = "./graph_store"
# Load documents and build index
documents = SimpleDirectoryReader("data").load_data()


# create and Store the date in the Graph store
index = PropertyGraphIndex.from_documents(documents)
index.storage_context.persist(storage_dir)

# Load the data from the vector store
storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
index = load_index_from_storage(storage_context)

# Chat with the data
chat_engine = index.as_chat_engine()
response = chat_engine.chat("What is this document talking about?")
print(response)
