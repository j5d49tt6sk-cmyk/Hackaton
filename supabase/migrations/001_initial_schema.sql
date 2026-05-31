create extension if not exists vector;
create extension if not exists pgcrypto;

insert into storage.buckets (id, name, public)
values ('rag-documents', 'rag-documents', false)
on conflict (id) do nothing;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  file_name text not null,
  original_file_name text,
  storage_bucket text not null default 'rag-documents',
  storage_path text,
  mime_type text,
  file_size bigint,
  source text,
  expert text,
  topic text,
  access_level int not null default 1,
  access_tag text not null default 'L1 Public',
  metadata jsonb not null default '{}'::jsonb,
  status text not null default 'uploaded',
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table documents
add column if not exists access_level int not null default 1;

alter table documents
add column if not exists access_tag text not null default 'L1 Public';

create table if not exists profiles (
  user_id uuid primary key,
  access_level int not null check (access_level >= 1)
);

create table if not exists document_texts (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  raw_text text not null,
  cleaned_text text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

alter table document_texts
add column if not exists access_level int not null default 1;

alter table document_texts
add column if not exists access_tag text not null default 'L1 Public';

create table if not exists document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  document_text_id uuid references document_texts(id) on delete cascade,
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

alter table document_chunks
add column if not exists access_level int not null default 1;

alter table document_chunks
add column if not exists access_tag text not null default 'L1 Public';

create table if not exists chat_messages (
  id bigserial primary key,
  session_id text not null,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  sources jsonb not null default '[]'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists employee_accounts (
  user_id uuid primary key,
  full_name text not null,
  email text not null unique,
  department text not null,
  access_level int not null check (access_level >= 1),
  active boolean not null default true,
  created_at timestamptz not null default now()
);

insert into employee_accounts (
  user_id,
  full_name,
  email,
  department,
  access_level
)
values
  (
    '11111111-1111-4111-8111-111111111111',
    'Anna Keller',
    'anna.keller@six-demo.local',
    'Operations',
    1
  ),
  (
    '22222222-2222-4222-8222-222222222222',
    'Ben Meier',
    'ben.meier@six-demo.local',
    'Compliance',
    2
  ),
  (
    '33333333-3333-4333-8333-333333333333',
    'Clara Rossi',
    'clara.rossi@six-demo.local',
    'Regulatory Management',
    3
  )
on conflict (user_id) do update
set
  full_name = excluded.full_name,
  email = excluded.email,
  department = excluded.department,
  access_level = excluded.access_level,
  active = true;

create index if not exists documents_expert_idx on documents (expert);
create index if not exists documents_status_idx on documents (status);
create index if not exists documents_access_level_idx on documents (access_level);
create index if not exists document_chunks_document_id_idx on document_chunks (document_id);
create index if not exists document_chunks_expert_idx on document_chunks (expert);
create index if not exists document_chunks_access_level_idx on document_chunks (access_level);
create index if not exists document_chunks_embedding_hnsw_idx
on document_chunks
using hnsw (embedding vector_cosine_ops);
create index if not exists chat_messages_session_id_idx on chat_messages (session_id, created_at);
create index if not exists employee_accounts_access_level_idx on employee_accounts (access_level);
create index if not exists profiles_access_level_idx on profiles (access_level);

update documents
set access_tag = case access_level
  when 1 then 'L1 Public'
  when 2 then 'L2 Internal'
  when 3 then 'L3 Confidential'
  when 99 then 'L99 Email Restricted'
  else 'L' || access_level::text || ' ' || access_tag
end
where access_tag not like 'L' || access_level::text || ' %';

update document_texts
set
  access_level = documents.access_level,
  access_tag = documents.access_tag
from documents
where
  document_texts.document_id = documents.id
  and (
    document_texts.access_level is distinct from documents.access_level
    or document_texts.access_tag is distinct from documents.access_tag
  );

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

alter table documents enable row level security;
alter table document_texts enable row level security;
alter table document_chunks enable row level security;
alter table profiles enable row level security;

drop policy if exists profiles_select_own on profiles;
create policy profiles_select_own
on profiles
for select
using (user_id = auth.uid());

drop policy if exists documents_select_by_access_level on documents;
create policy documents_select_by_access_level
on documents
for select
using (
  access_level < 99
  and access_level <= requester_access_level(auth.uid())
);

drop policy if exists document_texts_select_by_access_level on document_texts;
create policy document_texts_select_by_access_level
on document_texts
for select
using (
  access_level < 99
  and access_level <= requester_access_level(auth.uid())
);

drop policy if exists document_chunks_select_by_access_level on document_chunks;
create policy document_chunks_select_by_access_level
on document_chunks
for select
using (
  access_level < 99
  and access_level <= requester_access_level(auth.uid())
);

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
