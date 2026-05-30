# Company Brain

Company Brain is a Streamlit RAG application for preserving institutional
knowledge from SIX regulatory documents and internal transcripts.

## Current Architecture

```text
File upload or batch ingestion
-> Supabase Storage bucket: rag-documents
-> documents: file metadata and storage path
-> document_texts: raw and cleaned extracted text
-> document_chunks: chunk text, metadata, and pgvector embeddings
-> match_documents RPC: semantic search over document_chunks
-> OpenAI answer generation
-> chat_messages: conversation history and cited chunks
```

## Repository Structure

```text
SIX_Hack_Zurich/                  Local hackathon source documents
app.py                            Streamlit app
ingest.py                         Batch ingestion CLI
company_brain/
  answering.py                    Grounded answer generation
  chunking.py                     Section-aware chunking
  config.py                       Environment settings
  embeddings.py                   OpenAI embeddings
  ingestion.py                    Storage-first ingestion pipeline
  loaders.py                      PDF, DOCX, XLSX extraction
  metadata.py                     Expert/topic inference
  models.py                       Shared data models
  retrieval.py                    Semantic retrieval
  supabase_store.py               Supabase tables, RPC, Storage, chat messages
sql/match_documents.sql           RPC for vector search
supabase/migrations/001_initial_schema.sql
                                  Canonical Supabase schema
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```bash
OPENAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_STORAGE_BUCKET=rag-documents
COMPANY_BRAIN_PASSWORD=realesthacks
```

Create a Supabase Storage bucket named `rag-documents`.

Run the SQL in `supabase/migrations/001_initial_schema.sql` in Supabase. If the
tables already exist, compare the migration with the deployed schema and apply
the missing columns, indexes, and RPC.

Verify the backend after `.env`, the Storage bucket, and the Supabase schema are
ready:

```bash
python3 check_backend.py
```

To test only Supabase connectivity without calling OpenAI:

```bash
python3 check_backend.py --skip-openai
```

## Batch Ingestion

The local SIX documents are in:

```bash
SIX_Hack_Zurich
```

Run:

```bash
.venv/bin/python ingest.py SIX_Hack_Zurich
```

Supported source types:

- PDF: stores page numbers in chunk metadata
- DOCX: stores headings where available
- XLSX: stores sheet names in chunk metadata

## Streamlit App

Run:

```bash
.venv/bin/python app.py
```

The app supports:

- password-gated hackathon demo access
- guided case finder
- direct question answering
- Expert Twin filtering
- document upload to Supabase Storage
- indexing uploaded PDF/DOCX/XLSX files
- persisted chat messages in Supabase
- evidence display with file, page, sheet, and heading metadata
