create extension if not exists vector;

create or replace function match_documents(
  query_embedding vector(1536),
  match_count int default 8,
  match_expert text default null,
  similarity_threshold float default 0.0
)
returns table (
  id bigint,
  content text,
  source text,
  file_name text,
  expert text,
  topic text,
  chunk_index int,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    documents.id,
    documents.content,
    documents.source,
    documents.file_name,
    documents.expert,
    documents.topic,
    documents.chunk_index,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where
    documents.embedding is not null
    and (match_expert is null or documents.expert = match_expert)
    and (1 - (documents.embedding <=> query_embedding)) >= similarity_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;

create index if not exists documents_embedding_hnsw_idx
on documents
using hnsw (embedding vector_cosine_ops);

create index if not exists documents_expert_idx
on documents (expert);
