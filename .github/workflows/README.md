# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### `ci.yml` - Continuous Integration

Runs on every push and pull request to `main` or `develop` branches.

**Jobs:**

1. **Tests** - Run test suite on Python 3.12
   - Runs all unit tests with pytest
   - Continues on test failures to see all results
   - Uploads test results as artifacts

2. **Lint** - Code quality checks
   - Black formatting check
   - isort import sorting check
   
3. **Coverage** - Test coverage reporting
   - Generates coverage reports (XML, HTML, terminal)
   - Uploads coverage to Codecov (optional)
   - Uploads HTML report as artifact

4. **Config Validation** - Validate all YAML configs
   - Tests all config files in `experiments/configs/`
   - Ensures configs are valid before merge

5. **Summary** - Overall CI status
   - Aggregates all job results
   - Fails if any check fails

## Local Testing

Before pushing, run these locally:

```bash
# Run all tests
make test

# Check formatting
make format

# Run with coverage
make test-coverage

# Validate configs
for config in experiments/configs/*.yaml; do
  python scripts/validate_config.py "$config"
done
```

## CI Status Badge

Add this to your README.md:

```markdown
![CI Status](https://github.com/YOUR_USERNAME/ragicamp/workflows/CI/badge.svg)
```

## Secrets Configuration

For Codecov integration (optional), add these secrets to your GitHub repository:

- `CODECOV_TOKEN` - Your Codecov upload token

## Troubleshooting

### Tests fail locally but pass in CI

- Check Python version (CI runs 3.12)
- Ensure all dependencies are installed: `uv sync --extra dev`
- Make sure you're using the same Python version: `python --version`

### Formatting fails

```bash
# Auto-fix formatting
make format
```

### Coverage is too low

Run locally to see what's not covered:

```bash
make test-coverage
# Open htmlcov/index.html in browser
```

## Customization

To modify CI behavior:

1. **Change Python version**: Edit `python-version` in each job
2. **Change test command**: Edit the `Run tests` step
3. **Add new checks**: Add a new job following existing patterns
4. **Skip CI**: Add `[skip ci]` to commit message

## Performance

Current CI run time: ~8-12 minutes

- Tests: ~5-7 minutes
- Lint: ~1-2 minutes
- Coverage: ~2-3 minutes
- Config validation: ~1 minute

