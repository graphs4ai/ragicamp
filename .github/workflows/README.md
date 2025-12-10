# GitHub Actions Workflows

This directory contains CI/CD workflows for RAGiCamp.

## Workflows Overview

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main/develop | Main CI pipeline |
| `pr-check.yml` | Pull requests | Quick PR validation |
| `release.yml` | Tags (v*) | Create releases |

---

## `ci.yml` - Continuous Integration

**Runs on:** Every push and PR to `main` or `develop`

### Jobs

| Job | Duration | What it does |
|-----|----------|--------------|
| **lint** | ~2 min | Black, isort, mypy checks |
| **test** | ~5 min | Full test suite with coverage |
| **validate-configs** | ~1 min | Validate YAML configs |
| **import-check** | ~1 min | Verify package imports |
| **ci-success** | - | Final status check |

### Features

- ✅ **Dependency caching** - Fast subsequent runs
- ✅ **Concurrency control** - Cancels in-progress runs on new push
- ✅ **Coverage reports** - Uploaded to Codecov
- ✅ **Test artifacts** - JUnit XML for GitHub integration
- ✅ **Hydra config validation** - Tests new config system

### Fail Conditions

The CI **fails** if:
- Formatting is wrong (black/isort)
- Tests fail
- Configs are invalid
- Package won't import

---

## `pr-check.yml` - Pull Request Checks

**Runs on:** All pull requests

### Jobs

| Job | Purpose |
|-----|---------|
| **pr-size** | Warns on large PRs |
| **changes** | Detects what changed |
| **smoke-test** | Quick import test (if src changed) |
| **docs-check** | Link validation (if docs changed) |

### Smart Detection

Only runs relevant checks based on what changed:
- `src/` changed → Run smoke tests
- `docs/` changed → Check markdown links
- `conf/` changed → Validate configs

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
# Formatting (auto-fix)
make format

# Tests
make test

# Validate configs
make validate-all-configs

# Full CI-like check
make lint && make test && make validate-all-configs
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

## Badges

Add to README.md:

```markdown
[![CI](https://github.com/YOUR_USER/ragicamp/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/ragicamp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/YOUR_USER/ragicamp/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USER/ragicamp)
```

---

## Troubleshooting

### Tests pass locally but fail in CI

1. Check Python version matches (3.12)
2. Ensure `uv.lock` is committed
3. Check for OS-specific issues

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
uv run python -m ragicamp.cli.run --cfg job
```

### Coverage is low

```bash
# See what's not covered
make test-coverage
open htmlcov/index.html
```

---

## Performance

Typical CI run times:

| Job | Duration |
|-----|----------|
| Lint | ~2 min |
| Tests | ~5-7 min |
| Config Validation | ~1 min |
| Import Check | ~1 min |
| **Total** | **~8-10 min** |

With caching, subsequent runs are faster.

---

## Customization

### Skip CI

Add `[skip ci]` to commit message:

```bash
git commit -m "docs: update readme [skip ci]"
```

### Run specific job

Use workflow_dispatch:

```bash
gh workflow run ci.yml
```

### Add new checks

1. Add job to `ci.yml`
2. Add to `needs` in `ci-success`
3. Update this README
