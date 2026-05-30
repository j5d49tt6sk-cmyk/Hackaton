create extension if not exists vector;

insert into storage.buckets (id, name, public)
values ('rag-documents', 'rag-documents', false)
on conflict (id) do nothing;

create table if not exists documents (
  id bigserial primary key,
  file_name text not null,
  original_file_name text,
  storage_bucket text not null default 'rag-documents',
  storage_path text,
  mime_type text,
  file_size bigint,
  source text,
  expert text,
  topic text,
  metadata jsonb not null default '{}'::jsonb,
  status text not null default 'uploaded',
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists document_texts (
  id bigserial primary key,
  document_id bigint not null references documents(id) on delete cascade,
  raw_text text not null,
  cleaned_text text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists document_chunks (
  id bigserial primary key,
  document_id bigint not null references documents(id) on delete cascade,
  document_text_id bigint references document_texts(id) on delete cascade,
  chunk_index int not null,
  content text not null,
  embedding vector(1536),
  page_number int,
  sheet_name text,
  heading text,
  expert text,
  topic text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create table if not exists chat_messages (
  id bigserial primary key,
  session_id text not null,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  sources jsonb not null default '[]'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists documents_expert_idx on documents (expert);
create index if not exists documents_status_idx on documents (status);
create index if not exists document_chunks_document_id_idx on document_chunks (document_id);
create index if not exists document_chunks_expert_idx on document_chunks (expert);
create index if not exists document_chunks_embedding_hnsw_idx
on document_chunks
using hnsw (embedding vector_cosine_ops);
create index if not exists chat_messages_session_id_idx on chat_messages (session_id, created_at);

create or replace function touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists documents_touch_updated_at on documents;
create trigger documents_touch_updated_at
before update on documents
for each row
execute function touch_updated_at();

create or replace function match_documents(
  query_embedding vector(1536),
  match_count int default 8,
  match_expert text default null,
  similarity_threshold float default 0.0
)
returns table (
  id bigint,
  document_id bigint,
  content text,
  source text,
  file_name text,
  expert text,
  topic text,
  chunk_index int,
  page_number int,
  sheet_name text,
  heading text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    document_chunks.id,
    document_chunks.document_id,
    document_chunks.content,
    documents.source,
    documents.file_name,
    coalesce(document_chunks.expert, documents.expert) as expert,
    coalesce(document_chunks.topic, documents.topic) as topic,
    document_chunks.chunk_index,
    document_chunks.page_number,
    document_chunks.sheet_name,
    document_chunks.heading,
    document_chunks.metadata || jsonb_build_object(
      'document_metadata', documents.metadata,
      'storage_bucket', documents.storage_bucket,
      'storage_path', documents.storage_path,
      'mime_type', documents.mime_type
    ) as metadata,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  join documents on documents.id = document_chunks.document_id
  where
    document_chunks.embedding is not null
    and (match_expert is null or coalesce(document_chunks.expert, documents.expert) = match_expert)
    and (1 - (document_chunks.embedding <=> query_embedding)) >= similarity_threshold
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
$$;
