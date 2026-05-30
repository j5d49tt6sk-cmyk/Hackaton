# Company Brain

Company Brain is an AI-powered knowledge platform for preserving organizational
intelligence, expert knowledge, decision history, and institutional memory.

It is designed for a SIX hackathon demo and focuses on evidence-based answers,
source traceability, onboarding acceleration, and Expert Twins for specific
knowledge domains.

## Architecture

```text
company_brain/
  answering.py        Grounded answer generation with sources and confidence
  chunking.py         Text normalization and chunk splitting
  config.py           Environment variable settings
  embeddings.py       OpenAI embedding client
  ingestion.py        Recursive ingestion pipeline
  loaders.py          PDF, XLSX, DOCX, TXT, MD, and CSV readers
  metadata.py         Expert/topic inference from file paths
  models.py           Typed data models
  retrieval.py        Vector retrieval and Expert Twin filtering
  supabase_store.py   Supabase insert and match RPC wrapper

ingest.py             CLI for indexing documents
app.py                Streamlit chat UI
sql/match_documents.sql
                      Supabase pgvector RPC and indexes
```

## Supabase Setup

Create the `documents` table:

```sql
create table documents (
  id bigserial primary key,
  content text not null,
  source text,
  file_name text,
  expert text,
  topic text,
  chunk_index int,
  metadata jsonb,
  embedding vector(1536)
);
```

Then run `sql/match_documents.sql` in the Supabase SQL editor. It creates:

- `match_documents(...)` RPC for vector similarity search
- HNSW cosine index on `embedding`
- B-tree index on `expert`

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```bash
OPENAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
COMPANY_BRAIN_PASSWORD=realesthacks
```

Use the Supabase service role key only on the backend or local demo machine.
Do not expose it in a public frontend.

## Ingest Documents

The pipeline recursively scans a file or folder and supports:

- PDF
- XLSX
- DOCX
- TXT
- MD
- CSV

```bash
python ingest.py ./data
```

Optional overrides:

```bash
python ingest.py ./data/compliance --expert "Compliance Expert" --topic "MiFID"
python ingest.py ./data/esg --expert "ESG Expert"
python ingest.py ./data/transcripts --expert "Internal Expert"
```

Without overrides, the pipeline infers Expert Twins from file and folder names:

- Compliance Expert: MiFID, FATCA, SFDR, compliance, regulatory
- ESG Expert: ESG, sustainability, taxonomy, climate
- Internal Expert: meeting, transcript, minutes, internal, discussion

## Run the Demo UI

```bash
python3 app.py
```

In VS Code, open `app.py` and press Run. If dependencies are missing, install
them once with:

```bash
python3 -m pip install -r requirements.txt
```

The app opens on the first free port starting at `8501`. Use the terminal output
URL, usually `http://localhost:8501`. The demo password is `realesthacks`.

The UI supports:

- Guided case questionnaire
- Direct questions
- Expert Twin filtering
- Retrieved evidence inspection
- Source citations
- Confidence labels
- Decision Trail generation when the retrieved context contains decisions,
  alternatives, reasoning, or outcomes

## Retrieval Flow

1. Embed the user question with `text-embedding-3-small`
2. Call Supabase RPC `match_documents`
3. Optionally filter by `expert`
4. Return chunks with similarity scores
5. Generate an answer using only retrieved chunks

If the retrieved documents do not contain enough information, the model is
instructed to say so instead of inventing an answer.
