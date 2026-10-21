import os 
from dotenv import load_dotenv
import json 
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma 
from langchain_core.documents import Document 

def load_dande_electronics(path="data/dandes_electronics_kb.json"):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["documents"]

def load_as_documents(path="data/dandes_electronics_kb.json"):
    tickets = load_dande_electronics(path)

    documents = []
    for ticket in tickets:
        doc = Document(
            page_content=f"{ticket['title']}\n\n{ticket['content']}",
            metadata={
                "id": ticket["id"],
                "title": ticket["title"],
                "doc_type": ticket["doc_type"],
                "product": ticket["metadata"]["product"],
                "brand": ticket["metadata"]["brand"],
                "category": ticket["metadata"]["category"],
                **(
                    {"policy_code": ticket["metadata"]["policy_code"]}
                    if ticket["metadata"]["policy_code"] is not None
                    else {}
                ),
                "date": ticket["metadata"]["date"],
                **(
                    {"related_policy": ticket["metadata"]["related_policy"]}
                    if ticket["metadata"]["related_policy"] is not None
                    else {}
                )
            }
        )
        documents.append(doc)
    return documents


documents = load_as_documents()

load_dotenv() 
da_api_key = os.getenv("OPENAI_API_KEY") 

embeddings_model = OpenAIEmbeddings( 
    api_key=da_api_key, 
    model="text-embedding-3-small" 
) 

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

vectorstore = Chroma.from_documents( 
    documents=documents, 
    embedding=embeddings_model, 
    persist_directory=CHROMA_DB_PATH, 
    collection_name="dandes_v1" 
)  

print(f"Number of document indexed: {vectorstore._collection.count()}") 
 