import os
import streamlit as st
import sqlite3
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")


FILE_PATH = "paul_graham_essay.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-pro"
FLAG_FILE = "app_initialized.flag"
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Check if flag file exists to determine if this is a fresh start
    if not os.path.exists(FLAG_FILE):
        cursor.execute("DELETE FROM chat_history")  # Clear history only once
        with open(FLAG_FILE, "w") as f:
            f.write("initialized")  # Create the flag file to mark the first run

    conn.commit()
    conn.close()

# if "app_started" not in st.session_state:
#     init_db()
#     st.session_state["app_started"] = True  

init_db()

@st.cache_data
def load_and_split_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    return texts
    
texts = load_and_split_text(FILE_PATH)    

def create_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

vector_store = create_vector_store(texts)
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={"k":10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1, max_tokens=150, timeout=None)


SYSTEM_PROMPT = (
    "You arw an assistant for question-answering tasks. "
    "Use the following pieces of context to answer the question at the end."
    " If you don't know the answer, just "
    "say that you don't know, don't try to make up an answer. Use three sentences maximum and keep"
    " the answer as concise as possible."
    "Always be nice and friendly."
    "Always say 'thanks for asking!' at the end of the answer"
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

# Function to store chat in the database
def save_chat(user_query, bot_response):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_query, bot_response) VALUES (?, ?)", (user_query, bot_response))
    conn.commit()
    conn.close()

# Function to fetch previous chats
def get_chat_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT user_query, bot_response, timestamp FROM chat_history ORDER BY timestamp ASC LIMIT 5")
    history = cursor.fetchall()
    conn.close()
    return history
    
# Streamlit UI

# st.title("📖 What can I help with?")
# query = st.chat_input("Message: ") 

# if query:
#     with st.spinner("Thinking..."):
#         try:
#             question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
#             rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#             response = rag_chain.invoke({"input": query})

#             st.write(response.get("answer", "Sorry, I don't know the answer."))
#         except Exception as e:
#             st.error(f"An error occurred: {e}")


# st.title("📖 What can I help with?")
st.title("WELCOME TO COBSAi")
query = st.chat_input("Enter prompt message here... ")

if query:
    with st.spinner("Thinking..."):
        try:
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": query})
            bot_response = response.get("answer", "Sorry, I don't know the answer.")
            
            # Store chat in database
            save_chat(query, bot_response)
            
            # st.write(bot_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display previous chats
# st.subheader("📜 Recent Chat History")
chat_history = get_chat_history()
for user_query, bot_response, timestamp in chat_history:
    # st.write(f"🕒 {timestamp}")
    st.write(f"**You:** {user_query}")
    st.write(f"**Bot:** {bot_response}")
    # st.write("---")



