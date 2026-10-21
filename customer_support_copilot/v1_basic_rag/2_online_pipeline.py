import os 
from dotenv import load_dotenv 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv() 
da_api_key = os.getenv("OPENAI_API_KEY") 

llm_brain = ChatOpenAI(
    api_key= da_api_key,
    model="gpt-5.6-luna",
    max_tokens = 200
)

embeddings_model = OpenAIEmbeddings( 
 api_key=da_api_key, 
 model="text-embedding-3-small" 
) 

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

vectorstore = Chroma( 
    persist_directory=CHROMA_DB_PATH,
    embedding_function= embeddings_model,
    collection_name= "dandes_v1"
) 

#======================================= 
# Shared Config + Helper Functions 
#======================================= 
TOP_K = 5 
SYSTEM_INSTRUCTIONS = """You are a support assistant for Dandes Electronics. 
    Answer the user's question using ONLY the context provided below. 
    Each piece of context includes a [Source: ID - Title] line before the 
    body text. Treat the Title as part of the actual content, not just a 
    label — any specific detail mentioned in the Title applies to the 
    entire chunk, even if not repeated in the body text. 
    If, after considering both the Title and the content, the answer is 
    still not present, say: 
    "I don't know based on the available information." 
    Do not make up information beyond what's in the context. Be concise and helpful. 
"""

def format_docs(docs):
    parts = []
    for doc in docs:
        source = doc.metadata["id"]
        title = doc.metadata["title"]
        parts.append(f"[Source: {source} - {title}]\n{doc.page_content}")
    return "\n\n".join(parts)

def run_pipeline(user_query): 
    query_vector = embeddings_model.embed_query(user_query) 
    retrieved_docs = vectorstore.similarity_search_by_vector( 
        embedding=query_vector, 
        k=TOP_K 
    ) 
    
    context = format_docs(retrieved_docs) 
    augmented_prompt = f"""Context: 
    {context} 
    Question: {user_query} 
    """ 
    messages = [ 
        SystemMessage(content=SYSTEM_INSTRUCTIONS), 
        HumanMessage(content=augmented_prompt) 
    ] 
    response = llm_brain.invoke(messages) 
    generated_answer = response.content 

    sources = [f"{doc.metadata['id']}" for doc in retrieved_docs] 
    return {"answer": generated_answer, "sources": sources} 

#======================================= 
# Test with Multiple Questions 
#======================================= 
test_questions = [ 
 "Which macbook have cooling fan?", 
 "when will i get COD refund?",
 "When will be Dipawali celebrated?"  
] 

for i, question in enumerate(test_questions, start=1): 
    print("\n" + "=" * 60) 
    print(f"QUESTION {i}: {question}") 
    res = run_pipeline(question) 
    print("-" * 60) 
    print(f"Answer : {res['answer']}")
    print("-" * 60) 
    print(f"Sources : {', '.join(res['sources'])}")
