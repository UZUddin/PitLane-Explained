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
    
    /* White content area */
    .stApp { background-color: #f4f4f4; color: #1a1a1a; }


    
    /* Remove default top padding */
    [data-testid="stAppViewContainer"] > section > div {
        padding-top: 1rem !important;
    }

    /* Dark navy sidebar like F1 nav */
    [data-testid="stSidebar"] {
        background-color: #15151e;
        border-right: none;
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-size: 0.75em !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 600 !important;
        margin-top: 20px;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent;
        color: #aaaaaa !important;
        border: 1px solid #2a2a3a;
        border-radius: 3px;
        font-weight: 400;
        width: 100%;
        text-align: left;
        margin: 2px 0;
        font-size: 0.82em;
        padding: 8px 12px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #e10600;
        color: #ffffff !important;
        background-color: #1f1f2e;
    }
    
    /* Content area */
    .block-container {
        padding-top: 2rem !important;
        max-width: 900px;
    }
    
    /* Answer box */
    .answer-box {
        background-color: #f8f8f8;
        border-left: 3px solid #e10600;
        padding: 24px 28px;
        border-radius: 3px;
        margin-top: 12px;
        color: #1a1a1a;
        line-height: 1.8;
        font-size: 0.95em;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #e10600;
        color: white;
        border: none;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.85em;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.5rem 2rem;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #c00500;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 2px solid #eeeeee;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #999999;
        font-weight: 500;
        font-size: 0.85em;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #e10600 !important;
        background-color: transparent !important;
        font-weight: 700 !important;
    }
    
    /* Text input */
    .stTextInput input {
        background-color: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 3px;
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
        font-size: 0.95em;
        padding: 10px 14px;
    }
    .stTextInput input:focus {
        border-color: #e10600;
        box-shadow: none;
    }
    .stTextInput input::placeholder { color: #aaaaaa; }
    
    /* Labels */
    .stTextInput label {
        color: #666666 !important;
        font-size: 0.75em !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600 !important;
    }
    
    /* Success message */
    .stAlert {
        background-color: #f8f8f8 !important;
        border: 1px solid #eeeeee !important;
        border-radius: 3px;
        font-size: 0.85em;
    }
    
    /* Headings */
    h1 { color: #15151e; font-weight: 700; }
    h2, h3, h4 {
        color: #15151e;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Toggle */
    .stToggle label { color: #666666 !important; font-size: 0.82em !important; }
    
    /* Divider */
    hr { border-color: #eeeeee !important; }
    
    /* Caption */
    .stCaption { color: #999999 !important; font-size: 0.75em !important; }
</style>
""", unsafe_allow_html=True)



# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style='background-color: #15151e; padding: 24px 32px; margin: 0 0 24px 0; border-radius: 4px;'>
    <div style='display: flex; align-items: center; gap: 14px;'>
        <div style='width: 4px; height: 30px; background-color: #e10600; border-radius: 1px; flex-shrink: 0;'></div>
        <div>
            <div style='color: #ffffff; font-size: 1.5em; font-weight: 700; font-family: Inter, sans-serif;'>PitLane Explained</div>
            <div style='color: #666680; font-size: 0.75em; margin-top: 3px; letter-spacing: 0.1em; text-transform: uppercase; font-family: Inter, sans-serif;'>AI Race Day Companion &nbsp;·&nbsp; Powered by IBM Granite</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:

    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/250px-F1.svg.png", width=120)
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


def show_podium():
    # Podium visualization using native Streamlit
    col2, col1, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<div style='text-align:center; font-size:0.75em; color:#999; text-transform:uppercase; letter-spacing:0.08em;'>1st Place</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-weight:600;'>Charles Leclerc</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-size:0.8em; color:#e10600;'>Ferrari</div>", unsafe_allow_html=True)
        st.markdown("<div style='background:linear-gradient(to bottom, #FFD700, #FFA500); height:120px; border-radius:4px 4px 0 0; display:flex; align-items:center; justify-content:center;'><span style='font-size:2.5em; font-weight:700; color:white;'>1</span></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='text-align:center; font-size:0.75em; color:#999; text-transform:uppercase; letter-spacing:0.08em;'>2nd Place</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-weight:600;'>Carlos Sainz</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-size:0.8em; color:#e10600;'>Ferrari</div>", unsafe_allow_html=True)
        st.markdown("<div style='background:linear-gradient(to bottom, #C0C0C0, #A8A8A8); height:80px; border-radius:4px 4px 0 0; display:flex; align-items:center; justify-content:center;'><span style='font-size:2.5em; font-weight:700; color:white;'>2</span></div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div style='text-align:center; font-size:0.75em; color:#999; text-transform:uppercase; letter-spacing:0.08em;'>3rd Place</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-weight:600;'>Oscar Piastri</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-size:0.8em; color:#FF8000;'>McLaren</div>", unsafe_allow_html=True)
        st.markdown("<div style='background:linear-gradient(to bottom, #CD7F32, #A0522D); height:60px; border-radius:4px 4px 0 0; display:flex; align-items:center; justify-content:center;'><span style='font-size:2.5em; font-weight:700; color:white;'>3</span></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align:center; font-size:0.75em; color:#999; text-transform:uppercase; letter-spacing:0.1em; border-top:1px solid #eee; padding-top:12px; margin:12px 0 20px 0;'>2024 Monaco Grand Prix — Final Podium</div>", unsafe_allow_html=True)
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
# ── Tab 2: Race Summary ──────────────────────────────────────
with tab2:
    st.markdown("### 2024 Monaco Grand Prix")
    st.markdown("<p style='color: #666; font-size: 0.9em;'>A race story written for fans new to the sport.</p>", unsafe_allow_html=True)
    
    # Podium visualization
    st.markdown("""
    <div style='display: flex; justify-content: center; align-items: flex-end; gap: 16px; padding: 24px 0; font-family: Inter, sans-serif;'>
        
        <div style='text-align: center;'>
            <div style='font-size: 0.75em; color: #999; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.08em;'>2nd</div>
            <div style='font-size: 0.85em; font-weight: 600; color: #1a1a1a; margin-bottom: 4px;'>Carlos Sainz</div>
            <div style='font-size: 0.75em; color: #e10600; margin-bottom: 8px; font-weight: 500;'>Ferrari</div>
            <div style='background: linear-gradient(to bottom, #C0C0C0, #A8A8A8); height: 80px; width: 110px; border-radius: 4px 4px 0 0; display: flex; align-items: center; justify-content: center;'>
                <span style='font-size: 2em; font-weight: 700; color: white;'>2</span>
            </div>
        </div>
        
        <div style='text-align: center;'>
            <div style='font-size: 0.75em; color: #999; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.08em;'>1st</div>
            <div style='font-size: 0.85em; font-weight: 600; color: #1a1a1a; margin-bottom: 4px;'>Charles Leclerc</div>
            <div style='font-size: 0.75em; color: #e10600; margin-bottom: 8px; font-weight: 500;'>Ferrari</div>
            <div style='background: linear-gradient(to bottom, #FFD700, #FFA500); height: 120px; width: 110px; border-radius: 4px 4px 0 0; display: flex; align-items: center; justify-content: center;'>
                <span style='font-size: 2em; font-weight: 700; color: white;'>1</span>
            </div>
        </div>
        
        <div style='text-align: center;'>
            <div style='font-size: 0.75em; color: #999; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.08em;'>3rd</div>
            <div style='font-size: 0.85em; font-weight: 600; color: #1a1a1a; margin-bottom: 4px;'>Oscar Piastri</div>
            <div style='font-size: 0.75em; color: #FF8000; margin-bottom: 8px; font-weight: 500;'>McLaren</div>
            <div style='background: linear-gradient(to bottom, #CD7F32, #A0522D); height: 60px; width: 110px; border-radius: 4px 4px 0 0; display: flex; align-items: center; justify-content: center;'>
                <span style='font-size: 2em; font-weight: 700; color: white;'>3</span>
            </div>
        </div>
        
    </div>
    <div style='text-align: center; font-size: 0.75em; color: #999; text-transform: uppercase; letter-spacing: 0.1em; border-top: 1px solid #eeeeee; padding-top: 12px; margin-bottom: 20px;'>
        Final Podium
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Generate Race Story", type="primary"):
        with st.spinner("Generating race story..."):
            summary_prompt = """
            Give me a short, engaging 3-paragraph summary of the 2024 Monaco Grand Prix 
            written for someone who has never watched F1 before. Explain what happened, 
            why it was exciting, and what made the winner's victory special. 
            Use simple language and avoid technical jargon.
            """
            output = safe_invoke(rag_chain, {"input": summary_prompt})
            st.markdown("### Race Story")
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
            View on GitHub
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
