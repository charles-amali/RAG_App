import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon=":brain:", layout="centered")


FILE_PATH = "paul_graham_essay.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-pro"

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Missing API Key! Please set GOOGLE_API_KEY in the .env file.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data
def load_and_split_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    return texts
    
texts = load_and_split_text(FILE_PATH)    

def create_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

vector_store = create_vector_store(texts)
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={"k":10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=API_KEY, temperature=0, max_tokens=500, request_options={"timeout":5000})


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

st.markdown("<h1 style='text-align: center;'>Welcome To GrahamBot</h1>", unsafe_allow_html=True)

query = st.chat_input("Enter prompt message here... ")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        try:
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": query})
            bot_response = response.get("answer", "Sorry, I don't know the answer to that question.")
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
            with st.chat_message("assistant"):
                st.markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, there was an error. Please try again."})






