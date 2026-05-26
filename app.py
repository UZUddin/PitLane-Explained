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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background — dark charcoal not pure black */
    .stApp { background-color: #111111; color: #e8e8e8; }
    
    /* Sidebar — slightly lighter than main */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }
    [data-testid="stSidebar"] * { color: #cccccc !important; }
    [data-testid="stSidebar"] h3 { 
        color: #ffffff !important; 
        font-size: 0.85em !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600 !important;
    }
    
    /* Sidebar buttons — subtle not loud */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent;
        color: #aaaaaa !important;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        font-weight: 400;
        width: 100%;
        text-align: left;
        margin: 2px 0;
        font-size: 0.85em;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #c00500;
        color: #ffffff !important;
        background-color: #1f1f1f;
    }
    
    /* Main content padding */
    .block-container { 
        padding-top: 2.5rem;
        max-width: 860px;
    }
    
    /* Answer box */
    .answer-box {
        background-color: #1a1a1a;
        border-left: 3px solid #c00500;
        padding: 24px 28px;
        border-radius: 4px;
        margin-top: 12px;
        color: #e0e0e0;
        line-height: 1.8;
        font-size: 0.95em;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #c00500;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9em;
        letter-spacing: 0.05em;
        padding: 0.5rem 1.8rem;
        transition: background-color 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #a00400;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #2a2a2a;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #666666;
        font-weight: 500;
        font-size: 0.9em;
        letter-spacing: 0.03em;
        padding: 0.75rem 1.5rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #c00500 !important;
        background-color: transparent !important;
    }
    
    /* Text input */
    .stTextInput input {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95em;
        padding: 10px 14px;
    }
    .stTextInput input:focus {
        border-color: #c00500;
        box-shadow: none;
    }
    .stTextInput input::placeholder { color: #555555; }
    
    /* Labels */
    .stTextInput label { 
        color: #888888 !important;
        font-size: 0.8em !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500 !important;
    }
    
    /* Success/warning messages */
    .stAlert { 
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 4px;
        color: #888888 !important;
        font-size: 0.85em;
    }
    
    /* Headings */
    h1 { color: #ffffff; font-weight: 700; }
    h2, h3 { 
        color: #ffffff; 
        font-weight: 600;
        font-size: 1.1em;
        letter-spacing: 0.02em;
    }
    
    /* Toggle */
    .stToggle label { color: #888888 !important; font-size: 0.85em !important; }
    
    /* Divider */
    hr { border-color: #2a2a2a !important; }
    
    /* Caption */
    .stCaption { color: #555555 !important; font-size: 0.75em !important; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #111111; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)



# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style='padding: 20px 0 30px 0; border-bottom: 1px solid #2a2a2a; margin-bottom: 28px;'>
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
        <div style='width: 4px; height: 36px; background-color: #c00500; border-radius: 2px;'></div>
        <h1 style='color: #ffffff; font-size: 2em; margin: 0; font-weight: 700; letter-spacing: -0.02em;'>PitLane Explained</h1>
    </div>
    <p style='color: #555555; font-size: 0.9em; margin: 0 0 0 16px; letter-spacing: 0.03em;'>
        AI Race Day Companion &nbsp;·&nbsp; Powered by IBM Granite
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### How to Use")
    st.markdown("""
    <p style='color: #666; font-size: 0.85em; line-height: 1.8;'>
    1. Wait for AI to load<br>
    2. Select a tab<br>
    3. Ask a question or generate a race summary
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("### Try Asking")
    example_questions = [
        "What happens during a safety car?",
        "Explain DRS",
        "What is an undercut strategy?",
        "How does qualifying work?",
        "What do the flag colors mean?",
        "What is a virtual safety car?"
    ]
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.selected_question = q
    st.markdown("---")
    beginner_mode = st.toggle("Beginner Mode", value=True)
    st.caption("Simplifies all explanations for new fans")

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

st.success("AI ready! Ask me anything about F1.")

tab1, tab2 = st.tabs(["Ask Anything", "Race Summary"])

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
                st.markdown("### Answer:")
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
            st.markdown(f"<div class='answer-box'>{output['answer']}</div>", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 10px 0; color: #888888; font-size: 0.85em; font-family: Inter, sans-serif;'>
    <p style='margin-bottom: 6px;'>
        Built with 
        <a href='https://www.ibm.com/granite' target='_blank' style='color: #e10600; text-decoration: none;'>IBM Granite</a> · 
        <a href='https://www.docling.ai' target='_blank' style='color: #e10600; text-decoration: none;'>Docling</a> · 
        LangChain · Streamlit
    </p>
    <p style='margin-bottom: 6px;'>IBM SkillsBuild AI Builders Challenge 2026</p>
    <p>
        <a href='https://github.com/UZUddin/PitLane-Explained' target='_blank' style='color: #e10600; text-decoration: none; font-weight: 600;'>
            ⭐ View on GitHub
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
