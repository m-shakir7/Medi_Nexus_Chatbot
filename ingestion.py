import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load API Key
load_dotenv()

# --- Page Config (Research Focused Icons) ---
st.set_page_config(
    page_title="Medi-Nexus Research Portal", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for the UI ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8f9fc; }
    
    /* Dark Header Section */
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 40px;
        border-radius: 0px 0px 20px 20px;
        color: white;
        text-align: center;
        margin-top: -60px;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-card h3 { margin: 0; color: #64748b; font-size: 14px; text-transform: uppercase; }
    .metric-card p { margin: 5px 0 0; color: #1e293b; font-size: 20px; font-weight: bold; }

    /* Chat Styling */
    .stChatMessage {
        border-radius: 15px !important;
        background-color: white !important;
        border: 1px solid #f1f5f9 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar (Control Center) ---
with st.sidebar:
    st.markdown("### 🧬 Medi-Nexus AI")
    st.markdown("---")
    st.markdown("**🔬 Laboratory Control**")
    st.write(f"**Model:** GPT-4o-mini")
    st.write(f"**Status:** ⚡ Active")
    st.markdown("---")
    if st.button("🗑️ Clear Research Logs"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Secure Clinical Intelligence Environment")

# --- Header Display ---
st.markdown("""
    <div class='main-header'>
        <h1 style='font-size: 45px;'>🔬 Medi-Nexus Research</h1>
        <p style='font-size: 18px; opacity: 0.8;'>Advanced Clinical Intelligence Portal</p>
    </div>
    """, unsafe_allow_html=True)

# --- Metric Row ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='metric-card'><h3>Clinical Records</h3><p>Verified Papers</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'><h3>Processing Unit</h3><p>Real-time Logic</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><h3>Consultation</h3><p>Confidential</p></div>", unsafe_allow_html=True)

# --- Database Logic ---
@st.cache_resource
def load_medical_db():
    DB_PATH = "./chroma_db_medical"
    if not os.path.exists(DB_PATH):
        if os.path.exists("medical_data/") and len(os.listdir("medical_data/")) > 0:
            with st.status("🧬 Analyzing Biomarkers & Docs...") as status:
                from ingestion import create_medical_db
                create_medical_db()
                status.update(label="System Ready!", state="complete")
        else:
            st.error("No clinical data found in 'medical_data' folder.")
            st.stop()
    return Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

db = load_medical_db()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 5})

# --- The "Unbound" Prompt ---
template = """You are a professional Clinical Research Assistant powered by GPT-4o-mini.

HIERARCHY:
1. PDFs (Context): Priority source for all clinical, medical, or chemical queries.
2. IDENTITY: If asked who you are, state you are GPT-4o-mini Research AI.
3. GENERAL KNOWLEDGE: For everything else (world facts, Karachi, general chat), answer using your own knowledge.

Context: {context}
Question: {question}

Analysis Output:"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = ({"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()} 
             | prompt | llm | StrOutputParser())

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Enter clinical query or general research question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🧬 Cross-referencing clinical database..."):
            response = rag_chain.invoke(query)
            st.markdown(response)
            
            source_docs = retriever.invoke(query)
            if source_docs:
                with st.expander("🧬 Reference Origin"):
                    for s in {d.metadata.get('source') for d in source_docs}:
                        st.write(f"🧪 Source: {s}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})

            