# Live provider health checks

Run `make health-check` to verify that a real provider completes simulation,
evaluation, and HTML reporting. This check makes paid model requests and runs
separately from the offline test suite.

```bash
make health-check PROVIDER_CONFIG=src/usersim/cli/models_openai.toml HEALTH_LOCALE=ko_KR
```

Credentials follow the CLI's precedence: exported environment variables win,
then `.env.local`, then `.env`. The check searches from the working directory
upward, using the nearest directory containing either env file. Keep these
files private and untracked. Custom providers declare the credential variable
name through `api_key` in their model configuration.

The helper prints a final JSON record with `status`, `stage`, and `detail`:

| Status | Meaning | Exit status |
| --- | --- | --- |
| `pass` | Simulation, evaluation, and reporting completed; the report exists. | 0 |
| `skip` | Valid configuration needs credentials that are absent or empty. | 0 |
| `failure` | Configuration, dependencies, execution, timeout, or expected outputs failed. | 1 |

A skip does not establish provider health. Invalid or missing model files and
missing required aliases fail before checking credentials. Model parameters are
validated through the CLI's existing configuration conversion before a missing
credential can produce a skip. For example, zero `timeout` or
`max_parallel_requests` fails without starting a provider client or pipeline.
When a command fails, inspect its output and the reported stage; the helper
does not print credential values or dump subprocess arguments in its final
record.

Each command has a 900-second timeout. Override it with `HEALTH_TIMEOUT` in
seconds. The check uses one row and at most two turns, with `HEALTH_PROBE`
defaulting to `sov_ai_dynamic`. Temporary run artifacts are removed on exit.
Use the CLI commands directly with explicit output paths when investigating
artifacts from a failed run.

The scheduled/manual `Health checks` workflow runs each provider independently
with only that provider's secret. Provider failures fail their jobs; missing
credentials appear as explicit skips in the summary. Setup failures without a
helper outcome also appear as failures. Pull requests never receive these
credentials. For a local summary, set `HEALTH_SUMMARY_FILE` to an output path.

Regression tests in `tests/test_check_health.py` exercise credential precedence,
invalid configuration, skipped checks, command failures, timeouts, and missing
outputs without calling a live provider.
