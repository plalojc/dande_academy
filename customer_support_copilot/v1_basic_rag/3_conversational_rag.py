import os 
from dotenv import load_dotenv 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain_chroma import Chroma 
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser 

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

contextualize_prompt = ChatPromptTemplate.from_messages([ 
    ("system", "Given a chat history and the latest user question, " 
    "rewrite the question into a standalone question that " 
    "can be understood without the chat history. Do NOT " 
    "answer the question, just reformulate it if needed, " 
    "otherwise return it as is."), 
    MessagesPlaceholder("chat_history"), 
    ("human", "{input}") 
]) 

reformulate_chain = contextualize_prompt | llm_brain | StrOutputParser() 

def format_docs(docs): 
 parts = [] 
 for doc in docs: 
    source = doc.metadata["id"] 
    title = doc.metadata["title"] 
    parts.append(f"[Source: {source} - {title}]\n{doc.page_content}") 
 return "\n\n".join(parts) 

SYSTEM_INSTRUCTIONS = """You are a support assistant for Dandes Electronics. 
Answer using ONLY the context below. Use the Title as part of the 
meaning, not just the body text. If the answer isn't in the context, say: 
"I don't know based on the available information." 
Context: {context}
""" 
qa_prompt = ChatPromptTemplate.from_messages([ 
    ("system", SYSTEM_INSTRUCTIONS), 
    MessagesPlaceholder("chat_history"), 
    ("human", "{input}") 
]) 

qa_chain = qa_prompt | llm_brain | StrOutputParser() 

chat_history = [] 
def chat(user_message): 
    # 4a: Reformulate the question using chat history (skip on first turn) 
    if len(chat_history) == 0: 
        standalone_question = user_message 
    else: 
        standalone_question = reformulate_chain.invoke({ 
            "input": user_message, 
            "chat_history": chat_history 
        }) 
    # 4b: Retrieve using the reformulated, standalone question 
    retrieved_docs = retriever.invoke(standalone_question) 
    context = format_docs(retrieved_docs) 

    # 4c: Generate the answer using context + full chat history + original question 
    answer = qa_chain.invoke({ 
        "input": user_message, 
        "chat_history": chat_history, 
        "context": context 
    })

    # 4d: Update history with the REAL user message (not the reformulated one) 
    chat_history.append(HumanMessage(content=user_message)) 
    chat_history.append(AIMessage(content=answer)) 
    sources = [doc.metadata["id"] for doc in retrieved_docs] 

    return { 
        "query": user_message, 
        "standalone_question": standalone_question, 
        "answer": answer, 
        "sources": sources 
    }

conversation = [ 
 "What is the return window for laptops?", 
 "What if the box is opened?"
]

for i, message in enumerate(conversation, start=1): 
    result = chat(message) 
    print("\n" + "=" * 60)  
    print(f"Question : {result['query']}") 
    print(f"Answer : {result['answer']}") 
    print(f"Sources : {', '.join(result['sources'])}") 