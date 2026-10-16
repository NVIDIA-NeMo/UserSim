# Installed CLI end-to-end tests

Run `make test-e2e` to test a built UserSim wheel through its CLI and real HTTP client. The harness uses synthetic stored inputs and a deterministic OpenAI-compatible provider on loopback; it requires no provider credentials or downloaded persona datasets.

Setup needs uv, a supported Python, and access to dependency indexes and the tokenizer data URL. The repository's frozen lock supplies runtime dependencies. The test harness installs its own pinned pytest and PyArrow into a second environment. No development dependencies or editable UserSim install enter the runtime environment.

```bash
make test-e2e
```

The equivalent direct command is:

```bash
scripts/mise exec -- uv run --frozen python scripts/run_e2e.py
```

Pass an existing wheel to test the original release artifact without rebuilding:

```bash
scripts/mise exec -- uv run --frozen python scripts/run_e2e.py --wheel dist/usersim-VERSION-py3-none-any.whl
# Or require exactly one wheel in the release directory:
scripts/mise exec -- uv run --frozen python scripts/run_e2e.py --wheel-dir dist
```

`--wheel` and `--wheel-dir` are mutually exclusive and cannot be combined with `--run-only`. An empty or ambiguous wheel directory fails setup. `--uv /path/to/uv` selects the setup executable when it is not the one on `PATH`.

## Isolation and offline execution

The runner creates a directory outside the checkout containing `runtime/`, `harness/`, a copied test suite, a fresh home directory, and a tokenizer cache. It verifies that UserSim imports from `runtime/`. After admission, the runner writes empty `.env` and `.env.local` files in its owned work directory on every invocation, including `--run-only`. Admission checks these files and the direct preparation, cache, environment, and readiness entries before any output changes: symlinks, wrong file/directory kinds, and multiply linked writable files are rejected. Normal interpreter symlinks inside a prepared virtualenv remain supported. These stop the CLI's upward dotenv search before it reaches files outside the harness. CLI subprocesses run in fresh case directories beneath that boundary outside the source tree; the tests never import UserSim from the checkout or patch its probes, scorers, storage, or reporting.

The runtime receives a small environment allowlist and a synthetic local API key. External sockets are rejected by a test-only `sitecustomize` guard; loopback remains available to the fake provider. Hugging Face offline settings, extension-discovery suppression, and `NEMO_TELEMETRY_ENABLED=false` and other telemetry opt-outs are explicit. A subprocess verifies that installed DataDesigner telemetry is disabled. The ordinary suite's network guard and root conftest are not loaded.

Prepare environments while network access is available, then execute them without installation or downloads:

```bash
scripts/mise exec -- uv run --frozen python scripts/run_e2e.py --work-dir /tmp/usersim-e2e --prepare-only
scripts/mise exec -- uv run --frozen python scripts/run_e2e.py --work-dir /tmp/usersim-e2e --run-only
```

`--work-dir` must resolve outside the checkout and cannot be an ancestor of it. Paths containing `..` use their canonical locations for this check; symlinked work and artifact paths are rejected. Reuse the same path for both phases; the prepared environment and copied suite are a snapshot of setup. Changes to tests or wheels require another preparation. A new preparation removes the previous `installed.json` receipt before changing the environment, so an interrupted setup cannot advertise the old wheel as a valid prepared runtime. The container prepares these environments at image build time and runs the second command with external networking disabled.

## Scenarios and assertions

| Scenario | Contract checked |
| --- | --- |
| Complete workflow | `simulate → eval → report → select` preserves identities across Parquet reload, obtains real judge scores, writes report HTML, and selects the scored rows. |
| Stored-input resume | A second simulation reuses its run and trajectory IDs, preserves `input_id`, writes no duplicate rows, and makes no provider requests. |
| Tool calling | A weather function call travels through the real HTTP client, tool execution, and persisted transcript. A transient server error is retried. |
| Non-Latin locale | A Japanese input keeps its locale and native-script user text through persistence and evaluation. |
| Guarded tool request | A three-turn health-disclosure scenario forwards explicit `tool_choice`, preserves configured `extra_body`, and merges per-call nested reasoning settings. Its shipped synthetic clinical profile emits the expected placeholder warning. |
| Malformed response | Invalid evaluator JSON produces a nonzero CLI exit and empty judge results rather than successful scores. |
| Failed request | A provider HTTP 400 causes a nonzero CLI exit without published evaluation rows. |
| Network boundary | An attempted external connection fails with the harness's explicit network error. |

These cases verify client integration and artifact contracts. They do not measure model quality or establish acceptance by real providers. The health-disclosure case uses the shipped placeholder bank to exercise request plumbing and does not establish clinical validity.

## Results and diagnosis

Each execution exclusively creates `output/e2e/run-*/` and prints its exact path. Use `--artifacts ROOT` to select another output root; existing attempts and unrelated root files are left alone. A root on a bind mount works because only a new child is created, rather than renaming the mount. The runner prints its external work directory and copies scenario logs, synthetic inputs, recorded provider request bodies, trajectory/evaluation/selection Parquet, and report files into the artifact directory. `junit.xml` records the test result. The exported `installed.json` receipt retains the runtime import and asset locations and adds the selected wheel path and `wheel_sha256` captured during preparation. The digest binds prepared/run-only evidence to the wheel bytes, even if the original path later names a different file. Existing prepared environments without the digest need preparation again to provide this evidence. `installed.json` is the sole installation receipt. Captured requests contain only synthetic data and the local test model names.

Rejected admission leaves the work and output roots unchanged. Admission checks work/output paths, owned work entries, run-only readiness, and selected wheel inputs that could be removed by work-directory preparation. Safe paths containing `..` are canonicalized after lexical symlink checks. Once admitted, a new attempt directory is created before setup or execution, so even an early failure cannot mix previous scores, JUnit, or receipts into the attempt. The output root can contain earlier results or other caller-owned files; no marker, rotation, or automatic cleanup is required. Remove result directories when you no longer need them.

`--prepare-only` changes preparation state in the external work directory and creates no result directory. Successful preparation writes `installed.json`; an interrupted preparation removes readiness, so `--run-only` refuses it. Execution copies that receipt into its fresh result directory alongside JUnit and scenario files. CI uploads the output root recursively, including each attempt's diagnostics.

Read a failed command's `<scenario>/<command>.log`, then compare `<scenario>/requests.json` with the expected payload. The work directory retains both environments and uncopied DataDesigner diagnostics for inspection. Setup commands, CLI subprocesses, provider readiness, and the complete pytest process have timeouts. An empty test collection fails.

The runner does not remove the external work directory. Remove it after diagnosis to reclaim the installed environments and tokenizer cache.
