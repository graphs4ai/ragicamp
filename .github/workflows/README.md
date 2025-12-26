# GitHub Actions Workflows

This directory contains CI/CD workflows for RAGiCamp.

## Workflows Overview

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main/develop | Main CI pipeline |
| `pr-check.yml` | Pull requests | Quick PR validation (conditional) |
| `release.yml` | Tags (v*) | Create releases |

---

## `ci.yml` - Continuous Integration

**Runs on:** Every push and PR to `main` or `develop`

### Job Flow

```
┌─────────────────────────────────────────────────────┐
│                   PARALLEL (fast)                   │
│  ┌─────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  lint   │  │ import-check │  │validate-configs│  │
│  └────┬────┘  └──────┬───────┘  └───────┬────────┘  │
│       │              │                  │           │
└───────┼──────────────┼──────────────────┼───────────┘
        │              │                  │
        └──────────────┼──────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │     test       │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   ci-success   │
              └────────────────┘
```

### Jobs

| Job | Duration | What it does |
|-----|----------|--------------|
| **lint** | ~1 min | Black, isort, mypy checks |
| **import-check** | ~1 min | Verify package imports |
| **validate-configs** | ~1 min | Validate YAML configs |
| **test** | ~3-5 min | Full test suite with coverage |
| **ci-success** | - | Final status check |

### Features

- ✅ **UV caching** - Uses `enable-cache: true` for fast installs
- ✅ **Parallel fast checks** - lint, import-check, validate-configs run in parallel
- ✅ **Concurrency control** - Cancels in-progress runs on new push
- ✅ **Coverage reports** - Uploaded to Codecov on main
- ✅ **Test artifacts** - JUnit XML for GitHub integration

### Fail Conditions

The CI **fails** if:
- Formatting is wrong (black/isort)
- Tests fail
- Configs are invalid
- Package won't import

---

## `pr-check.yml` - Pull Request Checks

**Runs on:** All pull requests

### Smart Detection

Only runs relevant checks based on what changed:

| Changed Path | Triggered Job |
|--------------|---------------|
| `src/**` | smoke-test |
| `tests/**` | smoke-test |
| `pyproject.toml`, `uv.lock` | smoke-test |
| `docs/**`, `*.md` | docs-check |
| `conf/**`, `experiments/configs/**` | config-check |

### Jobs

| Job | Purpose |
|-----|---------|
| **changes** | Detect what files changed |
| **pr-size** | Warns on large PRs |
| **smoke-test** | Quick import test (conditional) |
| **docs-check** | Link validation (conditional) |
| **config-check** | Config validation (conditional) |

---

## `release.yml` - Releases

**Runs on:** Git tags matching `v*` (e.g., `v0.2.0`)

### Steps

1. **Build** - Creates wheel and sdist
2. **Test Install** - Tests on Python 3.9-3.12
3. **Release** - Creates GitHub release with artifacts

### Creating a Release

```bash
# Tag and push
git tag v0.2.0
git push origin v0.2.0

# Or manually trigger in GitHub Actions
```

---

## Local Testing

Before pushing, run these locally:

```bash
# Auto-fix formatting
make format

# Check linting (without fixing)
make lint

# Run tests
make test

# Run tests with coverage
make test-cov

# Validate configs
make validate-all-configs

# Full pre-push check (recommended!)
make pre-push
```

---

## Configuration

### Required Secrets

| Secret | Required | Purpose |
|--------|----------|---------|
| `CODECOV_TOKEN` | Optional | Coverage uploads |
| `PYPI_API_TOKEN` | Optional | PyPI publishing |

### Adding Secrets

1. Go to Repository → Settings → Secrets
2. Add new repository secret

---

## Troubleshooting

### Tests pass locally but fail in CI

1. Check Python version matches (3.12)
2. Ensure `uv.lock` is committed
3. Run `make format` before pushing

### Lint fails

```bash
# Auto-fix formatting
make format

# Commit the changes
git add -A && git commit -m "fix: formatting"
```

### Config validation fails

```bash
# Check specific config
uv run python scripts/utils/validate_config.py experiments/configs/my_config.yaml

# Check Hydra configs
uv run python -c "from hydra import compose, initialize_config_dir; ..."
```

---

## Performance

Typical CI run times:

| Job | Duration |
|-----|----------|
| Lint | ~1 min |
| Import Check | ~1 min |
| Config Validation | ~1 min |
| Tests | ~3-5 min |
| **Total** | **~5-7 min** |

With UV caching, subsequent runs are significantly faster.

---

## Skip CI

Add `[skip ci]` to commit message:

```bash
git commit -m "docs: update readme [skip ci]"
```
