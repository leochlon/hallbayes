# Generate Boilerplate/Content Verification Skill

Use this when generating tests, docs, synthetic data, config files, or migrations.

**Goal:** avoid boilerplate that is "almost right" (plausible, compiles, but encodes wrong assumptions).

---

## Artifact request
- What are we generating?
- Target language/framework/tooling?
- Any constraints (naming, style, lint, CI)?

## Evidence pack
**This is part of the skill.** Before generating anything, collect an evidence pack: the small snippets that define reality (schema, contracts, examples, constraints). Strawberry verifies against spans — it does not fetch them.

If you don’t have spans yet, stop and either collect them (repo/docs/experiments) or ask the user to paste them.

> List the spans you used (`S0`, `S1`, ...) and what they represent.

Suggested spans:
- S0: Schema / type definitions / API signatures
- S1: Existing examples in the repo (similar tests/migrations/configs)
- S2: Requirements or PRD excerpt (business rules)
- S3: Tooling constraints (lint rules, migration framework docs, CI output)

## Output (artifact)
Generate the artifact (tests/docs/migration/etc.).

## Verification trace (required)
After the artifact, write a short trace: **8–12 steps**.
Each step must be a **claim** plus **citations** to evidence spans.

Example step shape:
```json
{"idx": 0, "claim": "The table name is account_users.", "cites": ["S0"]}
```

## Verification
Run the verifier:

- Tool: `audit_trace_budget` (on the trace text)
- Recommended settings:
  - `require_citations=true`
  - `context_mode="cited"`
  - `default_target=0.95` for “review gate” behavior

If anything is flagged:
- revise the artifact (or downgrade the claim to an assumption)
- explicitly request the missing evidence instead of guessing
- If the verifier is run 3 times in a row, **STOP** and return only the claims that passed plus the claims that flagged and why they flagged.

---

## Worked example: migration for case-insensitive unique email per tenant

### Without Strawberry
**User:** Generate a migration to add a unique constraint on `users.email` and update docs.

**Assistant (uncited):**
```sql
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE(email);
```
Docs: “Email must be unique and case-insensitive.”

**Typical hidden problems:**
- table might not be `users`
- uniqueness might be **per tenant**
- case-insensitive uniqueness usually requires `lower(email)` (or `citext`) — easy to promise, easy to forget to implement

### With Strawberry

#### 1) Evidence pack
**S0 — schema**
```text
CREATE TABLE account_users (
  id uuid primary key,
  email text not null,
  created_at timestamptz not null
);
```

**S1 — existing indices**
```text
-- migrations/20240101_init.sql
CREATE INDEX account_users_email_idx ON account_users(email);
```

**S2 — requirement**
```text
PRD excerpt: "Emails must be unique per tenant. Case sensitivity: treat emails as case-insensitive."
```

**S3 — tenant model**
```text
ALTER TABLE account_users ADD COLUMN tenant_id uuid references tenants(id);
```

#### 2) Copy/paste prompt
> Generate the migration + doc update.  
> Then produce a trace (8–12 steps) of the key requirements and design choices.  
> Every step must cite spans `[S#]`.  
> Run `audit_trace_budget(steps=..., require_citations=true, context_mode="cited", default_target=0.95)` on the trace text.  
> If a step is unsupported, mark it as an assumption or ask for more evidence.
> If the verifier is run 3 times in a row, stop and return only the claims that passed plus the claims that flagged and why they flagged.

#### 3) Verification trace (what gets audited)
```json
[
  {"idx": 0, "claim": "The table name is account_users.", "cites": ["S0"]},
  {"idx": 1, "claim": "Uniqueness is required per tenant, so the uniqueness key must include tenant_id.", "cites": ["S2","S3"]},
  {"idx": 2, "claim": "Emails must be treated case-insensitively.", "cites": ["S2"]},
  {"idx": 3, "claim": "Enforce case-insensitive uniqueness via a unique index on (tenant_id, lower(email)).", "cites": ["S2","S3"]},
  {"idx": 4, "claim": "There is already a non-unique index on email.", "cites": ["S1"]}
]
```

#### 4) Verifier call (example)
```json
{
  "tool": "audit_trace_budget",
  "args": {
    "steps": [
      {"idx": 0, "claim": "The table name is account_users.", "cites": ["S0"]},
      {"idx": 1, "claim": "Uniqueness is required per tenant...", "cites": ["S2","S3"]},
      {"idx": 2, "claim": "Emails must be treated case-insensitively.", "cites": ["S2"]},
      {"idx": 3, "claim": "Enforce case-insensitive uniqueness via lower(email)...", "cites": ["S2","S3"]},
      {"idx": 4, "claim": "There is already a non-unique index on email.", "cites": ["S1"]}
    ],
    "spans": [
      {"sid":"S0","text":"..."},
      {"sid":"S1","text":"..."},
      {"sid":"S2","text":"..."},
      {"sid":"S3","text":"..."}
    ],
    "require_citations": true,
    "context_mode": "cited",
    "default_target": 0.95
  }
}
```

#### 5) Final artifact (now grounded)
**Migration**
```sql
-- Enforce case-insensitive uniqueness per tenant
CREATE UNIQUE INDEX account_users_tenant_lower_email_uniq
ON account_users (tenant_id, lower(email));
```

**Docs update (only what you can prove)**
- Emails must be unique per tenant. [S2][S3]
- Uniqueness is case-insensitive (enforced by a lower(email) unique index). [S2]
