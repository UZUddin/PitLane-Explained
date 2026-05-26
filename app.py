import streamlit as st
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from docling_core.types.doc.labels import DocItemLabel
from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_replicate import ChatReplicate
from ibm_granite_community.notebook_utils import get_env_var
import itertools

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PitLane Explained",
    page_icon="🏎️",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e0e0e; }
    .stApp { background-color: #0e0e0e; color: #ffffff; }
    .answer-box {
        background-color: #1a1a2e;
        border-left: 4px solid #e10600;
        padding: 20px;
        border-radius: 8px;
        margin-top: 10px;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #e10600;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff1801;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border-radius: 6px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e10600 !important;
        color: white !important;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; padding: 10px 0 20px 0;'>
    <h1 style='color: #e10600; font-size: 3em;'>🏎️ PitLane Explained</h1>
    <p style='color: #cccccc; font-size: 1.2em;'>Your AI Race Day Companion — Powered by IBM Granite</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Load models ──────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading AI models... this takes a minute")
def load_models():
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    
    embeddings_model_path = "ibm-granite/granite-embedding-small-english-r2"
    embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)
    embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
    
    model = ChatReplicate(
        model="ibm-granite/granite-4.1-8b",
        replicate_api_token=os.environ["REPLICATE_API_TOKEN"],
        model_kwargs={"max_completion_tokens": 1000, "min_tokens": 100},
    )
    return embeddings_model, embeddings_tokenizer, model

# ── Load documents ───────────────────────────────────────────
@st.cache_resource(show_spinner="📄 Loading F1 knowledge base...")
def load_rag_chain(_embeddings_model, _embeddings_tokenizer, _model):
    converter = DocumentConverter()
    
    sources = ["https://en.wikipedia.org/wiki/2024_Monaco_Grand_Prix"]
    conversions = {
        source.split("/")[-1]: converter.convert(source=source).document
        for source in sources
    }
    
    doc_id = 0
    texts = []
    for source, docling_document in conversions.items():
        for chunk in HybridChunker(tokenizer=_embeddings_tokenizer).chunk(docling_document):
            items = chunk.meta.doc_items
            if len(items) == 1 and isinstance(items[0], TableItem):
                continue
            refs = " ".join(map(lambda item: item.get_ref().cref, items))
            texts.append(Document(
                page_content=chunk.text,
                metadata={"doc_id": (doc_id := doc_id + 1), "source": source, "ref": refs},
            ))
    
    tables = []
    for source, docling_document in conversions.items():
        for table in docling_document.tables:
            if table.label in [DocItemLabel.TABLE]:
                tables.append(Document(
                    page_content=table.export_to_markdown(docling_document),
                    metadata={"doc_id": (doc_id := doc_id + 1), "source": source},
                ))
    
    vector_db = Chroma(embedding_function=_embeddings_model)
    for doc in itertools.chain(texts, tables):
        vector_db.add_documents([doc])
    
    prompt_template = ChatPromptTemplate.from_template("{input}")
    combine_docs_chain = create_stuff_documents_chain(llm=_model, prompt=prompt_template)
    rag_chain = create_retrieval_chain(
        retriever=vector_db.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )
    return rag_chain

# ── Safe invoke ──────────────────────────────────────────────
def safe_invoke(chain, input_dict, retries=3, wait=15):
    for attempt in range(retries):
        try:
            return chain.invoke(input_dict)
        except Exception as e:
            if "429" in str(e) or "throttled" in str(e):
                st.warning(f"⏳ Rate limit hit. Retrying in {wait}s... ({attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise e
    raise Exception("Max retries exceeded")

# ── Main app ─────────────────────────────────────────────────
embeddings_model, embeddings_tokenizer, model = load_models()
rag_chain = load_rag_chain(embeddings_model, embeddings_tokenizer, model)

st.success("✅ AI ready! Ask me anything about F1.")

tab1, tab2 = st.tabs(["💬 Ask Anything", "🏁 Race Summary"])

# ── Tab 1: Q&A ───────────────────────────────────────────────
with tab1:
    st.markdown("### Ask any F1 question")
    
    if "selected_question" in st.session_state:
        default_q = st.session_state.selected_question
    else:
        default_q = ""
    
    question = st.text_input(
        "Type your question here:",
        value=default_q,
        placeholder="e.g. What does the yellow flag mean?"
    )
    
    if st.button("Ask 🏎️", type="primary"):
        if question:
            with st.spinner("Granite is thinking..."):
                if beginner_mode:
                    full_question = f"{question} Please explain in simple terms for someone who has never watched F1 before."
                else:
                    full_question = question
                
                output = safe_invoke(rag_chain, {"input": full_question})
                
                st.markdown("### 🏎️ Answer:")
                st.markdown(f"<div class='answer-box'>{output['answer']}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please type a question first!")

# ── Tab 2: Race Summary ──────────────────────────────────────
with tab2:
    st.markdown("### Get the story of a race")
    st.markdown("Get an engaging summary of the 2024 Monaco Grand Prix written for casual fans.")
    
    if st.button("Generate Race Story 🏁", type="primary"):
        with st.spinner("Generating race story..."):
            summary_prompt = """
            Give me a short, engaging 3-paragraph summary of the 2024 Monaco Grand Prix 
            written for someone who has never watched F1 before. Explain what happened, 
            why it was exciting, and what made the winner's victory special. 
            Use simple language and avoid technical jargon.
            """
            
            output = safe_invoke(rag_chain, {"input": summary_prompt})
            
            st.markdown("### 🏁 Race Story:")
            st.markdown(output['answer'])

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    Built with IBM Granite · Docling · LangChain · Streamlit<br>
    IBM SkillsBuild AI Builders Challenge 2026
</div>
""", unsafe_allow_html=True)
