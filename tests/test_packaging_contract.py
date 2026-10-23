# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts that the rest of the suite cannot catch.

Everything else runs against the editable monorepo, where imports resolve from
the source tree and assets are found by walking up from the CWD. These tests
assert the properties that only matter once the code is *installed*: that the
plugin's dotted entry points resolve, and that every third-party import is
actually declared as a dependency.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The running Python's standard library, plus ``tomllib``, which is new in
#: 3.11: on 3.10 the package imports the declared ``tomli`` backport instead.
_STDLIB = sys.stdlib_module_names | {"tomllib"}


class TestPluginEntryPoints:
    """Data Designer resolves the simulator and evaluator by *dotted string*
    at load time. Nothing else in the suite imports usersim.engine.plugin,
    so a class rename or module move passes every other test and fails on the
    first real run. This is the cheapest guard in the repo."""

    @staticmethod
    def _plugins():
        from usersim.engine import plugin

        return [
            getattr(plugin, name)
            for name in dir(plugin)
            if not name.startswith("_") and type(getattr(plugin, name)).__name__ == "Plugin"
        ]

    def test_both_plugins_are_declared(self) -> None:
        assert len(self._plugins()) == 2, "expected simulator + evaluator plugins"

    @pytest.mark.parametrize("attr", ["impl_qualified_name", "config_qualified_name"])
    def test_dotted_names_resolve_to_real_objects(self, attr: str) -> None:
        for plug in self._plugins():
            dotted = getattr(plug, attr)
            module_path, _, symbol = dotted.rpartition(".")
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:  # pragma: no cover - the failure we guard
                pytest.fail(f"{dotted}: module {module_path!r} does not import ({exc})")
            assert hasattr(module, symbol), (
                f"{dotted}: {module_path!r} has no attribute {symbol!r}. A rename broke a plugin entry point."
            )

    def test_declared_entry_points_match_the_module(self) -> None:
        """The pyproject entry points and plugin.py must not drift apart."""
        pyproject = tomllib.load((_REPO_ROOT / "pyproject.toml").open("rb"))
        declared = pyproject["project"]["entry-points"]["data_designer.plugins"]
        assert declared, "no data_designer.plugins entry points declared"

        for _name, target in declared.items():
            module_path, _, symbol = target.partition(":")
            module = importlib.import_module(module_path)
            assert hasattr(module, symbol), f"{target}: missing {symbol!r}"

    def test_registry_discovers_both_plugins(self) -> None:
        """End to end: Data Designer's own registry must find them."""
        from data_designer.plugins.registry import PluginRegistry

        discovered = set(PluginRegistry()._plugins)
        assert {"conversation-simulator", "trajectory-evaluator"} <= discovered, (
            f"registry discovered {sorted(discovered)}"
        )


class TestDeclaredDependencies:
    """Third-party imports and declared dependencies match in both directions.
    An undeclared import resolves fine in the monorepo, where a sibling package
    happens to pull it in, and fails on a standalone install. A declared
    dependency nothing imports still installs, with everything it depends on,
    for every user."""

    #: Imported behind try/except ImportError on purpose; must stay optional.
    #: Undeclared on purpose: a guarded notebook probe, not a dependency.
    _OPTIONAL = {"IPython"}

    #: Distribution names differ from import names for a few packages.
    _IMPORT_TO_DISTRIBUTION = {
        "yaml": "pyyaml",
        "lingua": "lingua_language_detector",
        "dotenv": "python_dotenv",
    }

    @staticmethod
    def _imports(src_root: Path, own_package: str) -> set[str]:
        found: set[str] = set()
        for path in src_root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
        return {m for m in found if m not in _STDLIB and m != own_package}

    @staticmethod
    def _declared(pyproject: Path) -> set[str]:
        data = tomllib.load(pyproject.open("rb"))
        names = set()
        for spec in data["project"].get("dependencies", []):
            # "pydantic>=2.0", "typing_extensions>=4.0; python_version < '3.11'"
            head = spec.split(";")[0]
            for sep in (">=", "<=", "==", "~=", ">", "<", "["):
                head = head.split(sep)[0]
            names.add(head.strip().lower().replace("-", "_"))
        return names

    def test_plugin_declares_everything_it_imports(self) -> None:
        root = _REPO_ROOT
        imported = self._imports(root / "src", "usersim")
        declared = self._declared(root / "pyproject.toml")

        # Packages this distribution ships; they are not dependencies.
        first_party = {"usersim"}
        missing = {
            mod
            for mod in imported - self._OPTIONAL - first_party
            if self._IMPORT_TO_DISTRIBUTION.get(mod, mod).lower().replace("-", "_") not in declared
        }
        assert not missing, f"the distribution imports but does not declare: {sorted(missing)}"

    def test_plugin_imports_everything_it_declares(self) -> None:
        imported = {
            self._IMPORT_TO_DISTRIBUTION.get(mod, mod).lower().replace("-", "_")
            for mod in self._imports(_REPO_ROOT / "src", "usersim")
        }
        unused = self._declared(_REPO_ROOT / "pyproject.toml") - imported
        assert not unused, f"the distribution declares but never imports: {sorted(unused)}"

    def test_optional_imports_stay_guarded(self) -> None:
        """An undeclared import is only acceptable behind try/except."""
        src = _REPO_ROOT / "src"
        for mod in self._OPTIONAL:
            sites: list[tuple[Path, bool]] = []
            for path in src.rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                # Which import nodes sit inside a try block?
                in_try: set[int] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Try):
                        for child in ast.walk(node):
                            in_try.add(id(child))
                for node in ast.walk(tree):
                    names: list[str] = []
                    if isinstance(node, ast.Import):
                        names = [a.name.split(".")[0] for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module.split(".")[0]]
                    if mod in names:
                        sites.append((path, id(node) in in_try))

            assert sites, f"{mod} is listed optional but never imported"
            unguarded = [p for p, guarded in sites if not guarded]
            assert not unguarded, (
                f"{mod} must be imported inside try/except ImportError or be "
                f"declared; unguarded in: {[str(p.relative_to(_REPO_ROOT)) for p in unguarded]}"
            )


class TestProbeTemplate:
    """The probe scaffold is a contributor's first file; it must run.

    It rotted silently through the package rename: every import still named
    ``conversation_plugin``, a package that no longer exists, and it told the
    reader to copy itself to a directory that had moved. Nothing caught it
    because a ``.template`` is not collected by pytest and not linted.
    """

    TEMPLATE = _REPO_ROOT / "templates" / "probe" / "test_probe.py.template"

    #: The scaffold's fill-in-the-blank probe module. Not expected to import.
    PLACEHOLDER = "usersim.engine.probes.YOUR_PROBE"

    def test_template_parses(self) -> None:
        import ast

        ast.parse(self.TEMPLATE.read_text())

    def test_template_imports_resolve(self) -> None:
        import importlib
        import re

        src = self.TEMPLATE.read_text()
        modules = sorted(set(re.findall(r"^\s*(?:from|import)\s+(usersim[\w.]*)", src, re.M)))
        assert modules, "template should import from usersim"
        broken = []
        for module in modules:
            if module == self.PLACEHOLDER:
                continue
            try:
                importlib.import_module(module)
            except ImportError as e:
                broken.append(f"{module}: {e}")
        assert not broken, f"probe template imports do not resolve: {broken}"

    def test_template_names_a_real_destination(self) -> None:
        """The 'copy this to ...' path must exist, or step one fails."""
        import re

        src = self.TEMPLATE.read_text()
        match = re.search(r"Copy this to ``([^`]+)``", src)
        assert match, "template should say where to copy itself"
        destination = (_REPO_ROOT / match.group(1)).parent
        assert destination.is_dir(), f"template points at a missing dir: {destination}"

    def test_referenced_examples_exist(self) -> None:
        """It cites three probe tests as references; dead ones help nobody."""
        import re

        src = self.TEMPLATE.read_text()
        cited = set(re.findall(r"``(test_\w+)\.py``", src))
        assert cited, "template should cite worked examples"
        missing = [name for name in cited if not (_REPO_ROOT / "tests" / "engine" / "probes" / f"{name}.py").is_file()]
        assert not missing, f"template cites tests that do not exist: {sorted(missing)}"


class TestTypeMarker:
    """PEP 561. The marker is a single empty file, which makes it easy to lose
    in a build-config change and impossible to notice from the source tree,
    where annotations resolve regardless."""

    def test_marker_exists(self) -> None:
        assert (_REPO_ROOT / "src" / "usersim" / "py.typed").is_file()

    def test_marker_is_declared_as_package_data(self) -> None:
        with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
            data = tomllib.load(fh)
        package_data = data["tool"]["setuptools"]["package-data"]
        assert "py.typed" in package_data.get("usersim", []), (
            "src/usersim/py.typed exists but is not declared as package data, "
            "so the wheel ships without it and annotations are invisible to "
            "type checkers."
        )


class TestProbeAssetPreflight:
    """A selected probe whose bank is absent must fail before any model call.

    Without the pre-flight the failure surfaces from inside a loader partway
    through a run, after the other probes in the mix have already been billed
    for -- and ``--dry-run`` reports the plan and exits 0, which asserts the
    run is fine.
    """

    def test_missing_probe_directory_is_rejected(self, tmp_path: Path) -> None:
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        with pytest.raises(ConfigError) as exc:
            verify_probe_assets({"financial_services": 1.0}, ["en_US"], tmp_path)
        message = str(exc.value)
        assert "financial_services" in message
        assert "no asset directory" in message
        # The error must say what to do, not just what is wrong.
        assert "--probe-mix" in message and "--assets-dir" in message
        # A missing probe directory is a wrong assets tree, not a locale to ask for.
        assert "feature request" not in message

    def test_missing_locale_subtree_is_rejected(self) -> None:
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        with pytest.raises(ConfigError) as exc:
            verify_probe_assets({"financial_services": 1.0}, ["ja_JP"], None)
        message = str(exc.value)
        assert "ja_JP: no bank" in message
        # Naming the locales that can run turns a dead end into a next step.
        assert "en_US" in message.split("Locales that can run: ")[1].splitlines()[0]

    def test_a_locale_whose_bank_does_not_load_is_rejected(self, finance_assets_without_fr_tasks: Path) -> None:
        """A locale folder that exists is not enough: without tasks, every row aborts."""
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        with pytest.raises(ConfigError) as exc:
            verify_probe_assets({"financial_services": 1.0}, ["fr_FR"], finance_assets_without_fr_tasks)
        message = str(exc.value)
        assert "fr_FR: its bank does not load" in message and "tasks.yaml" in message
        assert message.split("Locales that can run: ")[1].splitlines()[0] == "en_US"
        verify_probe_assets({"financial_services": 1.0}, ["en_US"], finance_assets_without_fr_tasks)

    def test_india_variant_without_its_own_bank_uses_en_in(self) -> None:
        """A registered variant runs on its base locale's bank, as its probe and scorer resolve it."""
        from usersim.cli._pipeline import verify_probe_assets

        mix = {"financial_services": 1.0, "sov_ai_facts": 1.0, "sov_ai_dynamic": 1.0}
        verify_probe_assets(mix, ["bn_Beng_IN"], None)

    def test_india_variant_is_rejected_when_its_base_bank_is_absent(self, tmp_path: Path) -> None:
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        (tmp_path / "financial_services" / "en_US").mkdir(parents=True)
        (tmp_path / "financial_services" / "en_US" / "bank.yaml").write_text("{}")
        with pytest.raises(ConfigError) as exc:
            verify_probe_assets({"financial_services": 1.0}, ["bn_Beng_IN"], tmp_path)
        assert "bn_Beng_IN" in str(exc.value)

    def test_the_hindi_locales_run_finance_on_the_en_in_bank(self) -> None:
        """hi_Deva_IN ships a corpus without tasks and hi_Latn_IN no bank; both run on en_IN's."""
        from usersim.cli._pipeline import verify_probe_assets

        mix = {"financial_services": 1.0, "sov_ai_facts": 1.0, "sov_ai_dynamic": 1.0}
        verify_probe_assets(mix, ["hi_Deva_IN", "hi_Latn_IN"], None)

    def test_the_hindi_locales_are_rejected_when_the_en_in_bank_is_absent(self, tmp_path: Path) -> None:
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        (tmp_path / "financial_services" / "en_US").mkdir(parents=True)
        (tmp_path / "financial_services" / "en_US" / "bank.yaml").write_text("{}")
        with pytest.raises(ConfigError) as exc:
            verify_probe_assets({"financial_services": 1.0}, ["hi_Deva_IN", "hi_Latn_IN"], tmp_path)
        assert "hi_Deva_IN (through en_IN): no bank; hi_Latn_IN (through en_IN): no bank" in str(exc.value)

    def test_a_refusal_names_the_locales_that_run_on_another_bank(self) -> None:
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        with pytest.raises(ConfigError) as exc:
            verify_probe_assets({"financial_services": 1.0}, ["ko_KR"], None)
        can_run = str(exc.value).split("Locales that can run: ")[1].splitlines()[0]
        assert "en_US" in can_run
        assert can_run.endswith(", plus hi_Deva_IN, hi_Latn_IN and the India language variants through en_IN")

    def test_a_locale_without_a_bank_says_how_to_ask_for_one(self) -> None:
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        with pytest.raises(ConfigError) as exc:
            verify_probe_assets({"financial_services": 1.0}, ["ko_KR"], None)
        assert str(exc.value).split("\n\n", 1)[1].splitlines() == [
            "Drop the probe from --probe-mix or the locale from the run, or point --assets-dir at a tree that has them.",
            "To ask for a locale, open a feature request: "
            "https://github.com/NVIDIA-NeMo/UserSim/issues/new?template=feature-request.yml",
            "To build a financial_services bank yourself, see https://github.com/NVIDIA-NeMo/UserSim/blob/main/"
            "src/usersim/engine/probes/financial_services/README.md#authoring-a-new-locale",
        ]

    def test_the_refusal_links_point_at_files_in_this_repository(self) -> None:
        """The links are absolute, to work from an installed package; moving a file must not break them quietly."""
        from usersim.cli._pipeline import _BANK_AUTHORING_GUIDES, _FEATURE_REQUEST_URL

        template = _FEATURE_REQUEST_URL.split("?template=", 1)[1]
        assert (_REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / template).is_file()
        for url in _BANK_AUTHORING_GUIDES.values():
            path, anchor = url.removeprefix("https://github.com/NVIDIA-NeMo/UserSim/blob/main/").split("#")
            lines = (_REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
            headings = {line.lstrip("#").strip().lower().replace(" ", "-") for line in lines if line.startswith("#")}
            assert anchor in headings, f"{path} has no heading for #{anchor}"

    def test_every_missing_probe_reported_at_once(self, tmp_path: Path) -> None:
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import verify_probe_assets

        mix = {"financial_services": 0.5, "sov_ai_facts": 0.5}
        with pytest.raises(ConfigError) as exc:
            verify_probe_assets(mix, ["en_US"], tmp_path)
        message = str(exc.value)
        assert "financial_services" in message and "sov_ai_facts" in message

    def test_shipped_assets_satisfy_every_probe(self) -> None:
        """The banks that ship must satisfy their own probes for en_US."""
        import usersim.engine.generator  # noqa: F401  (populates registry)
        from usersim.cli._pipeline import verify_probe_assets
        from usersim.engine.core.probes import known_probes

        verify_probe_assets({p: 1.0 for p in known_probes()}, ["en_US"], None)


class TestSdistContents:
    def test_tests_are_pruned_from_the_sdist(self) -> None:
        """The suite runs from a checkout; a partial copy without conftest.py would not run."""
        directives = [line.split() for line in (_REPO_ROOT / "MANIFEST.in").read_text().splitlines()]
        assert ["prune", "tests"] in directives


class TestManifestProvenance:
    """A run manifest records the version of the installed ``usersim`` distribution.

    It is the distribution named ``usersim`` that counts, so stale metadata from
    another name that also claims the package cannot supply the version.
    """

    def test_engine_version_is_recorded(self) -> None:
        from usersim.engine.core.manifest import _own_version

        assert _own_version(), (
            "the manifest would record a null engine version; the lookup "
            "no longer resolves to an installed distribution"
        )

    def test_version_matches_the_installed_distribution(self) -> None:
        from importlib.metadata import version

        from usersim.engine.core.manifest import _own_version

        assert _own_version() == version("usersim")

    def test_package_and_cli_report_the_installed_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``pyproject.toml`` is the only place the version is written."""
        from importlib.metadata import version

        import usersim
        from usersim import cli

        assert usersim.__version__ == version("usersim")
        with pytest.raises(SystemExit) as exc:
            cli.main(["--version"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == f"usersim {version('usersim')}"

    def test_code_info_carries_both_versions(self) -> None:
        from usersim.engine.core.manifest import _detect_code_info

        info = _detect_code_info()
        assert info.conversation_plugin_version
        assert info.data_designer_version


class TestRuntimeDependencySurface:
    """A library must not force development tooling on every install.

    ``ipykernel`` was a runtime dependency, which pulled IPython, the Jupyter
    client and kernel stack, pyzmq, tornado and debugpy -- 23 packages -- to
    satisfy one *guarded* ``from IPython import get_ipython`` used to detect
    whether a run started in a notebook. Every one of those also enlarged the
    licence-review surface.
    """

    @staticmethod
    def _runtime_deps() -> set[str]:
        data = tomllib.load((_REPO_ROOT / "pyproject.toml").open("rb"))
        names = set()
        for spec in data["project"]["dependencies"]:
            head = spec.split(";")[0]
            for sep in (">=", "<=", "==", "~=", ">", "<", "["):
                head = head.split(sep)[0]
            names.add(head.strip().lower().replace("_", "-"))
        return names

    def test_notebook_tooling_is_not_a_runtime_dependency(self) -> None:
        forbidden = {
            "ipykernel",
            "ipython",
            "jupyter",
            "jupyterlab",
            "notebook",
            "jupyter-client",
            "jupyter-core",
            "nbconvert",
            "nbformat",
        }
        leaked = forbidden & self._runtime_deps()
        assert not leaked, (
            f"notebook tooling declared as a runtime dependency: {sorted(leaked)}. "
            "The dev extra carries the kernel; shipped code only touches IPython "
            "through a guarded import."
        )

    def test_test_tooling_is_not_a_runtime_dependency(self) -> None:
        """Ours, at least -- sqlfluff drags pytest in from upstream."""
        forbidden = {"pytest", "pytest-cov", "pytest-xdist", "pytest-env", "ruff"}
        leaked = forbidden & self._runtime_deps()
        assert not leaked, f"test tooling declared at runtime: {sorted(leaked)}"

    def test_ipython_stays_a_guarded_import(self) -> None:
        """The reason ipykernel is not needed: the import already degrades."""
        import ast

        source = (_REPO_ROOT / "src/usersim/engine/core/manifest.py").read_text()
        tree = ast.parse(source)
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    guarded.add(id(child))

        sites = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "IPython"]
        assert sites, "expected a guarded IPython import in manifest.py"
        assert all(id(n) in guarded for n in sites), (
            "an IPython import escaped its try/except; it would then need to be declared as a runtime dependency"
        )
