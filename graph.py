from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader



documents = SimpleDirectoryReader(
    "../../examples/data/paul_graham"
).load_data()

# create
index = PropertyGraphIndex.from_documents(documents)

# save
index.storage_context.persist("./storage")

# load
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)