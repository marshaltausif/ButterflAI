-- ButterflAI — Complete Supabase Schema v3
-- Supabase Dashboard → SQL Editor → New query → Run all

create extension if not exists "uuid-ossp";

create table if not exists public.profiles (
  id uuid references auth.users(id) on delete cascade primary key,
  email text, display_name text, avatar_url text, kaggle_user text,
  preferred_ai text default 'gemini', total_jobs integer default 0,
  created_at timestamptz default now(), updated_at timestamptz default now()
);

create table if not exists public.jobs (
  id uuid default uuid_generate_v4() primary key,
  user_id uuid references public.profiles(id) on delete cascade,
  job_key text unique not null, status text default 'pending',
  goal text not null, task_type text, model_arch text,
  dataset_id text, dataset_source text, dataset_name text,
  num_classes integer, classes jsonb, config jsonb,
  best_val_acc float, epochs_run integer, elapsed_seconds integer,
  ai_provider text, was_auto_fixed boolean default false, fix_count integer default 0,
  drive_folder_id text, drive_link text,
  created_at timestamptz default now(), updated_at timestamptz default now()
);

create table if not exists public.model_outputs (
  id uuid default uuid_generate_v4() primary key,
  job_id uuid references public.jobs(id) on delete cascade,
  drive_folder_id text, drive_link text,
  model_file_id text, streamlit_file_id text,
  history_file_id text, config_file_id text,
  best_val_acc float, model_size_bytes bigint,
  files jsonb, download_bundle_url text,
  created_at timestamptz default now()
);

create table if not exists public.consistency_reports (
  id uuid default uuid_generate_v4() primary key,
  job_id uuid references public.jobs(id) on delete cascade,
  overall text, summary text, checks jsonb, fixes jsonb,
  code_before text, code_after text,
  was_fixed boolean default false, fix_count integer default 0,
  fixed_at timestamptz, created_at timestamptz default now()
);

create table if not exists public.dataset_cache (
  id uuid default uuid_generate_v4() primary key,
  query_hash text unique not null, query text not null,
  results jsonb not null, hit_count integer default 1,
  created_at timestamptz default now(),
  expires_at timestamptz default (now() + interval '7 days')
);

create table if not exists public.training_epochs (
  id uuid default uuid_generate_v4() primary key,
  job_id uuid references public.jobs(id) on delete cascade,
  epoch integer not null, train_loss float, train_acc float,
  val_loss float, val_acc float, is_best boolean default false,
  logged_at timestamptz default now()
);

alter table public.profiles            enable row level security;
alter table public.jobs                enable row level security;
alter table public.model_outputs       enable row level security;
alter table public.consistency_reports enable row level security;
alter table public.training_epochs     enable row level security;
alter table public.dataset_cache       enable row level security;

create policy "own_profiles" on public.profiles  for all using (auth.uid() = id);
create policy "own_jobs"     on public.jobs       for all using (auth.uid() = user_id);
create policy "own_outputs"  on public.model_outputs
  for all using (job_id in (select id from public.jobs where user_id = auth.uid()));
create policy "own_reports"  on public.consistency_reports
  for all using (job_id in (select id from public.jobs where user_id = auth.uid()));
create policy "own_epochs"   on public.training_epochs
  for all using (job_id in (select id from public.jobs where user_id = auth.uid()));
create policy "cache_read"   on public.dataset_cache for select using (true);

create or replace function update_ts() returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end; $$;
create trigger ts_profiles before update on public.profiles for each row execute function update_ts();
create trigger ts_jobs     before update on public.jobs     for each row execute function update_ts();

create or replace function inc_jobs() returns trigger language plpgsql security definer as $$
begin update public.profiles set total_jobs = total_jobs + 1 where id = new.user_id; return new; end; $$;
create trigger on_job after insert on public.jobs for each row execute function inc_jobs();
