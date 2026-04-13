# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This is a monorepo with two main projects:

- **`gpustack/`** — Python backend (FastAPI + SQLModel/SQLAlchemy + Alembic migrations)
- **`gpustack-ui/`** — React frontend (UmiJS/Max framework, Ant Design, TypeScript)

## Build Commands (Backend)

All backend commands run from `gpustack/` directory. The project uses `uv` for dependency management and `make` with shell scripts in `hack/`.

```bash
cd gpustack

make install          # Install dev tools (uv, pre-commit hooks)
make deps             # Sync and lock all dependencies
make generate         # Run code generation (python -m gpustack.codegen.generate)
make lint             # Run pre-commit checks on all files
make test             # Run pytest
make build            # Build package (uv build)
make ci               # Full CI: install + deps + lint + test + build
```

### Running Tests

```bash
cd gpustack
uv run pytest                    # Run all tests
uv run pytest tests/scheduler/   # Run tests for a specific module
uv run pytest tests/test_file.py::test_name  # Run a single test
uv run pytest -m unit            # Run only unit tests
```

Tests use `pytest-asyncio`. The root `conftest.py` sets up a temp directory and global Config fixture.

### Database Migrations

Uses Alembic. Default DB URL: `postgresql://root@localhost:5432/gpustack?sslmode=disable`.

```bash
cd gpustack
DATABASE_URL="..." alembic revision --autogenerate -m "description"
# Or use the helper script:
./hack/generate-migration-revision.sh "description"
```

## Build Commands (Frontend)

```bash
cd gpustack-ui
pnpm install         # Install dependencies
pnpm dev             # Dev server
pnpm build           # Production build
pnpm preview         # Preview production build
```

## Architecture

### Backend (`gpustack/gpustack/`)

GPUStack is a GPU cluster manager for AI model deployment. The entry point is `main.py` using argparse with subcommands (`start`, `reload-config`, `reset-admin-password`, etc.).

**Server-Worker Architecture:**
- **Server** (`server/`) — FastAPI app serving the API and UI. Key files:
  - `server/app.py` — FastAPI app factory with lifespan management
  - `server/server.py` — Server class orchestrating startup
  - `server/services.py` — Business logic service layer
  - `server/db.py` — Database connection (SQLite/PostgreSQL/MySQL via SQLModel)
  - `server/controllers.py` — Model instance state machine controller
- **Worker** (`worker/`) — Runs on GPU nodes, manages model serving:
  - `worker/worker.py` — Main worker process
  - `worker/backends/` — Pluggable inference engines: `vllm`, `sglang`, `ascend_mindie`, `vox_box`, `custom`
  - `worker/serve_manager.py` — Manages model serving lifecycle
  - `worker/inference_backend_manager.py` — Manages inference backend processes
- **Scheduler** (`scheduler/`) — Allocates GPU resources and selects engines for model deployments
- **Gateway** (`gateway/`) — AI proxy routing with Higress integration, plugin system for request/response transformation

**Key Subsystems:**
- `routes/` — FastAPI route handlers (models, workers, clusters, benchmarks, auth, OpenAI-compatible API, etc.)
- `schemas/` — SQLModel/Pydantic schemas for DB models and API serialization
- `api/` — API middleware, auth, error handling
- `client/` — Generated API clients (from OpenAPI spec via codegen)
- `migrations/` — Alembic database migrations
- `config/` — Configuration management with environment variable support
- `k8s/` — Kubernetes integration (Jinja2 manifest templates for worker deployment)
- `cloud_providers/` — Cloud provider integration (DigitalOcean)
- `detectors/` — GPU/hardware detection (fastfetch, runtime, custom detectors)
- `codegen/` — OpenAPI-based code generation for API clients

### Frontend (`gpustack-ui/src/`)

React app built with UmiJS Max framework and Ant Design 6.
- `pages/` — Route-based page components (dashboard, llmodels, playground, cluster-management, etc.)
- `services/` — API client layer (profile, system)
- `components/` — Shared UI components
- `models/` — State management
- `locales/` — i18n translation files

### Configuration

Environment variables with `GPUSTACK_` prefix. Key env vars are read via `utils/envs.py`. The `Config` class in `config/config.py` is the central configuration object, set globally via `set_global_config()`.

### Inference Backend Extension

Backends are pluggable. Each backend in `worker/backends/` extends `base.py`. The `custom.py` backend supports user-defined engines. Community backends are bundled from `gpustack/community-inference-backends` repo during build.

### Running the Server Locally

```bash
cd gpustack
uv run gpustack start --database-url postgresql://postgres:mysecretpassword@localhost:5432/postgres --gateway-mode disabled --api-port 80
```

### Format Converter Module (`converter/`)

Protocol translation between Anthropic Messages API and OpenAI Chat Completions API. Enables custom model providers with format conversion. Key files: `converters.py` (core logic), `streaming.py` (SSE streaming), `router.py` (format routing), `provider_proxy.py` (provider proxy).

## Conventions

- Python formatting: **black** (line-length 88, skip string normalization, exclude migrations)
- Linting: **pre-commit** framework with flake8
- DB models: SQLModel (SQLAlchemy + Pydantic hybrid)
- API follows OpenAI-compatible standards for LLM endpoints
- Frontend uses pnpm, not npm
- Frontend dev server runs on `http://localhost:9000`
- Frontend routes defined in `gpustack-ui/config/routes.ts`
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found
