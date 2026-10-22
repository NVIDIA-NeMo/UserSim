# Local and CI validation

CI calls repository Make targets so failures can be reproduced locally. After
`make install-dev`, run `make validate-ci` to execute the native validation
bundle, including distribution checks.

| Target | Checks |
| --- | --- |
| `make check-all` | Static checks defined by the repository. |
| `make check-dependency-licenses` | Dependency license policy. |
| `make coverage` | Ordinary test suite with an 84% minimum total coverage. |
| `make smoke` | Offline configuration, registry, asset, and reporting self-check. |
| `make check-extensions` | Out-of-tree installed extension behavior. |
| `make validate-native` | The five targets above. |
| `make check-dist` | Build and validate archives, then check a clean wheel installation. |
| `make validate-ci` | `validate-native` plus `check-dist`. |

The distribution archives are produced by `check-dist`; extension checks build
a separate temporary wheel. CI also runs the test suite across Python 3.11
through 3.14; a local bundle checks the active
Python environment. Provider health checks run separately and need credentials.

## Coverage

The Python 3.12 CI matrix job runs `make coverage`; the other Python versions
run `make test`. Both test paths use the same ordinary suite. Coverage failures
fail the existing `Tests (Python 3.12)` check. The 84% floor lives in the Makefile
and applies to local runs as well as CI.

```bash
make coverage PYTEST_WORKERS=2
make coverage PYTEST_ARGS="--durations=25 --junitxml=junit.xml"
make coverage COV_XML=coverage-custom.xml
```

`coverage.xml` is the default XML output. `COV_XML` changes its location, and
`PYTEST_ARGS` passes additional pytest options through to the coverage command.
Create a custom output directory before writing there. A focused pytest
selection also limits measured execution; run the complete suite to reproduce
the CI coverage gate. `COV_FAIL_UNDER` is configurable for experiments, but
maintainers must not lower the committed floor to make a change pass.

CI uploads `coverage.xml` as `coverage-python-3.12` for 14 days, even when the
coverage command fails. Missing XML fails the upload step, and uploading a
report does not suppress a preceding test or coverage failure. If collection or
setup stops before XML is produced, inspect the failed step's logs.
