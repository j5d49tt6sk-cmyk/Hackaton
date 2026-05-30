create extension if not exists vector;

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

create index if not exists document_chunks_embedding_hnsw_idx
on document_chunks
using hnsw (embedding vector_cosine_ops);

create index if not exists document_chunks_expert_idx
on document_chunks (expert);
