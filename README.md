# 🏎️ PitLane Explained
### Your AI Race Day Companion for Casual F1 Fans

🔗 **[Live App → pitlane-explained.streamlit.app](https://pitlane-explained.streamlit.app)**

---

## The Problem
Formula 1 is the most data-rich sport in the world — but none of that reaches casual fans in a way they can actually understand. New viewers feel lost watching races, confused by strategy calls, flags, and terminology that experienced fans take for granted.

## The Solution
PitLane Explained is an AI-powered race day companion that lets anyone ask plain-English questions about F1 and get clear, accurate answers — powered by IBM Granite.

## Features
- **💬 Ask Anything** — Type any F1 question and get a plain-English answer grounded in official race data
- **🏁 Race Summary** — Get an engaging narrative summary of a race written for casual fans
- **🟢 Beginner Mode** — Toggle on to get every answer explained as if you've never watched F1

## Technical Approach
- **IBM Granite** (`granite-4.1-8b`) — LLM for answer generation and race summaries
- **IBM Granite Embeddings** (`granite-embedding-small-english-r2`) — Text embeddings for semantic search
- **Docling** — Document parsing and chunking of race data sources
- **LangChain + ChromaDB** — RAG pipeline connecting Granite to F1 knowledge base
- **Streamlit** — Frontend interface

### Architecture
User Question → Granite Embeddings → ChromaDB Vector Search →
Retrieved F1 Context → IBM Granite LLM → Plain-English Answer

## Why It Matters
F1 viewership is growing rapidly globally, with millions of new fans joining every season. PitLane Explained bridges the gap between raw race data and fan understanding — making the sport accessible to anyone, anywhere, at any time.

## IBM Technologies Used
- IBM Granite 4.1 8B Instruct (via Replicate)
- IBM Granite Embedding Small English R2
- Docling for document processing

## Setup & Installation
```bash
git clone https://github.com/UZUddin/PitLane-Explained.git
cd PitLane-Explained
pip install -r requirements.txt
streamlit run app.py
```

Set your Replicate API token:
```bash
export REPLICATE_API_TOKEN="your_token_here"
```

## IBM SkillsBuild AI Builders Challenge
Built for the May 2026 F1 Challenge — IBM SkillsBuild AI Builders Challenge


