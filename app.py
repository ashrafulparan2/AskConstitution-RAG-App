import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login

# Custom CSS for chat-like appearance
st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        align-self: flex-end;
        word-wrap: break-word;
        display: inline-block;
        float: right;
        clear: both;
    }
    .bot-message {
        background-color: #e9ecef;
        color: black;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        word-wrap: break-word;
        display: inline-block;
        float: left;
        clear: both;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 10px;
    }
    .stButton > button {
        border-radius: 20px;
        background-color: #007bff;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load FAISS DB
DB_FAISS_PATH = "./vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Hugging Face login
login(token=secrets.Read)

# Load the LLM
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": "", "max_length": "512"}
    )
    return llm

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context

Context: {context}
Question: {question}
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Set up QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Streamlit App
st.title("Chat with the Constitution of Bangladesh")

# Session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for q, a in st.session_state.conversation:
        st.markdown(f'<div class="user-message">{q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message">{a}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input form for user query
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Ask anything about the Constitution of Bangladesh:", key="user_input")
    submit_button = st.form_submit_button(label="Send")

# Process user query
if submit_button and user_query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({'query': user_query})
        answer = response["result"]
        # Optionally include source documents in the answer
        # sources = "\n\n*Sources:*\n" + "\n".join([doc.page_content[:100] + "..." for doc in response["source_documents"]])
        # answer_with_sources = answer + sources
        st.session_state.conversation.append((user_query, answer))
    st.rerun()

# Add a subtle footer
st.markdown("<hr><small>Powered by LangChain & Mistral AI</small>", unsafe_allow_html=True)