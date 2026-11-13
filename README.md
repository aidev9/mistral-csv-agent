# drive-data-analyst

A conversational data analyst built on the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/overview).
You give it a link to a CSV file in your Google Drive and ask questions in plain language;
it answers with real numbers from your data.

## How it works

- **Google Drive connector** (`google_drive_mcp`): the agent uses your own Google Drive
  connection to find files and, when a file is not shared by link, to read its content.
  It acts as the user who runs the session (on-behalf-of), so everyone sees their own Drive.
  Only CSVs inside the Drive folder named `data` may be loaded — enforced in the agent's
  instructions and, hard, in `load_csv` itself, which rejects any other folder.
- **Fixed analysis tools** — no code execution, no shell, no sandbox. The model can only
  call these deterministic Python functions with fixed arguments:
  | Tool | What it does |
  | --- | --- |
  | `load_csv` | Loads a CSV from a Drive link/file id (direct download when shared by link, otherwise via the connector's file content); rejects files outside the `data` folder |
  | `describe_columns` | Per column: inferred type, nulls, distinct count, min/mean/max or most frequent values |
  | `filter_rows` | Counts and samples rows matching one condition (12 operators, case-insensitive strings), with share of total |
  | `group_by_aggregate` | Group-by (multi-column cross-tabs) with count/sum/mean/median/min/max/nunique, optional filters, ordering, limits, and share of total per group |
  | `time_trends` | Buckets rows by day/week/month on a date column and computes a metric per bucket, with peak and spread — for spikes, drift and seasonality |
  | `preview_rows` | Shows a slice of raw rows |
- **Stateful sessions**: the dataset and the whole conversation live in a durable session
  stored by Mistral, so follow-ups remember the file and everything discussed. If a worker
  restart clears the in-memory dataset, the tools say so and the agent reloads the file
  itself from the link already in the conversation.
- **Managed deployment**: `agents deploy` registers the agent as a workflow; Mistral builds
  the Docker image from the pushed commit and runs the worker. Because the agent acts on
  behalf of the session user (Drive connector), the deployment is hardened by default.

## Project layout

- `main.py` — the agent: model, instructions, tools, connector.
- `analyst_tools.py` — the fixed tools and the CSV engine (pure Python `csv`, no pandas).
- `tests/test_tools.py` — unit tests for parsing, filtering and aggregation (no API key needed).

## Local development

```bash
uv sync
uv run pytest -q          # unit tests, no API key needed
uv run agents info        # inspect the agent
uv run --env-file .env agents chat   # chat locally (needs MISTRAL_API_KEY in .env)
```

## Deploy (Mistral-managed)

Requirements: the project pushed to this GitHub repository, the GitHub connector connected
in [Mistral Studio](https://console.mistral.ai/build/connectors), and a `MISTRAL_API_KEY`
whose user has a Workflows-capable workspace role (Developer or Workspace Contributor).

```bash
uv run --env-file .env agents deploy
```

Then chat with the deployed agent by name:

```bash
uv run --env-file .env agents chat drive-data-analyst
```

or open it in Vibe Code Web at the URL printed by the deploy
(`https://chat.mistral.ai/managed-agents/drive-data-analyst?source=agents_api`).

## Notes and limits

- The model calls the analysis tools directly (`model_access="direct"`); it never writes
  code to compute results. The Google Drive connector's own tools are reached through the
  SDK's standard programmatic-tool-calling channel.
- CSVs are parsed with Python's `csv` module: values are typed as number/text/null
  (`N/A`, `NA`, `null`, `none`, `nan` and empty cells are nulls); string comparisons are
  case-insensitive. Very large files (hundreds of MB) are not the target: the file content
  passes through the session when the connector read path is used, and the dataset is held
  in worker memory per session.
- Direct download works only for files shared "anyone with the link"; otherwise the agent
  falls back to reading the content through your Drive connection.
