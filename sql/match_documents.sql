create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists profiles (
  user_id uuid primary key,
  access_level int not null check (access_level >= 1)
);

alter table document_chunks
add column if not exists access_level int not null default 1;

alter table document_chunks
add column if not exists access_tag text not null default 'L1 Public';

update documents
set access_tag = case access_level
  when 1 then 'L1 Public'
  when 2 then 'L2 Internal'
  when 3 then 'L3 Confidential'
  when 99 then 'L99 Email Restricted'
  else 'L' || access_level::text || ' ' || access_tag
end
where access_tag not like 'L' || access_level::text || ' %';

update document_chunks
set
  access_level = documents.access_level,
  access_tag = documents.access_tag
from documents
where
  document_chunks.document_id = documents.id
  and (
    document_chunks.access_level is distinct from documents.access_level
    or document_chunks.access_tag is distinct from documents.access_tag
  );

create or replace function requester_access_level(requester_user_id uuid)
returns int
language sql
stable
security invoker
as $$
  select coalesce(
    (
      select profiles.access_level
      from profiles
      where profiles.user_id = requester_user_id
      limit 1
    ),
    (
      select employee_accounts.access_level
      from employee_accounts
      where
        employee_accounts.user_id = requester_user_id
        and employee_accounts.active = true
      limit 1
    ),
    0
  );
$$;

alter table document_chunks enable row level security;

drop policy if exists document_chunks_select_by_access_level on document_chunks;
create policy document_chunks_select_by_access_level
on document_chunks
for select
using (
  access_level < 99
  and access_level <= requester_access_level(auth.uid())
);

create or replace function match_documents(
  query_embedding vector(1536),
  match_count int default 8,
  match_expert text default null,
  similarity_threshold float default 0.0,
  requester_user_id uuid default auth.uid()
)
returns table (
  id uuid,
  document_id uuid,
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
security invoker
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
      'mime_type', documents.mime_type,
      'access_level', document_chunks.access_level,
      'access_tag', document_chunks.access_tag
    ) as metadata,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  join documents on documents.id = document_chunks.document_id
  where
    document_chunks.embedding is not null
    and (match_expert is null or coalesce(document_chunks.expert, documents.expert) = match_expert)
    and document_chunks.access_level < 99
    and document_chunks.access_level <= requester_access_level(requester_user_id)
    and (1 - (document_chunks.embedding <=> query_embedding)) >= similarity_threshold
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
$$;

create index if not exists document_chunks_embedding_hnsw_idx
on document_chunks
using hnsw (embedding vector_cosine_ops);

create index if not exists document_chunks_expert_idx
on document_chunks (expert);
