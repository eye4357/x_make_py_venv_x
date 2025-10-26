"""Multi-version Python environment manager for x_make_py_venv_x."""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypedDict, TypeVar, cast

from x_make_common_x.x_subprocess_utils_x import CommandError, run_command

LOGGER = logging.getLogger(__name__)


DEFAULT_AUTO_REQUIREMENT_FILES: tuple[str, ...] = (
    "requirements.txt",
    "x_0_make_all_x/requirements.txt",
)

_MAJOR_INDEX = 0
_MINOR_INDEX = 1
_PATCH_INDEX = 2


HashableT = TypeVar("HashableT", bound=Hashable)


def _dedupe_preserve_order(items: Iterable[HashableT]) -> list[HashableT]:
    seen: set[HashableT] = set()
    result: list[HashableT] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


class Tool(StrEnum):
    """Supported interpreter orchestration tools."""

    AUTO = "auto"
    UV = "uv"
    PYENV = "pyenv"
    PYLAUNCHER = "py"
    CURRENT = "current"

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


@dataclass(frozen=True)
class VersionRequest:
    """Parsed representation of a requested Python runtime."""

    raw: str
    major: int
    minor: int
    patch: int | None

    @classmethod
    def parse(cls, text: str) -> VersionRequest:
        parts = text.split(".")
        if not parts or not parts[0].isdigit():
            msg = f"Invalid version specifier: {text!r}"
            raise ValueError(msg)
        major = int(parts[_MAJOR_INDEX])
        minor = (
            int(parts[_MINOR_INDEX])
            if len(parts) > _MINOR_INDEX and parts[_MINOR_INDEX].isdigit()
            else 0
        )
        patch = (
            int(parts[_PATCH_INDEX])
            if len(parts) > _PATCH_INDEX and parts[_PATCH_INDEX].isdigit()
            else None
        )
        return cls(raw=text, major=major, minor=minor, patch=patch)

    @property
    def env_slug(self) -> str:
        if self.patch is not None:
            return f"{self.major}.{self.minor}.{self.patch}"
        return f"{self.major}.{self.minor}"

    @property
    def py_launcher_tag(self) -> str:
        return f"{self.major}.{self.minor}"

    @property
    def env_name(self) -> str:
        return f".venv-{self.env_slug}"

    @property
    def tox_env(self) -> str:
        return f"py{self.major}{self.minor}"


class EnvManager:
    """Coordinate interpreter availability and virtual environment creation."""

    def __init__(
        self,
        *,
        tool: Tool,
        project_root: Path,
        env_root: Path,
        dry_run: bool = False,
    ) -> None:
        self.tool = tool
        self.project_root = project_root
        self.env_root = env_root
        self.dry_run = dry_run

    def ensure_versions(
        self,
        versions: Sequence[VersionRequest],
        requirements: Sequence[Path],
        packages: Sequence[str],
        *,
        upgrade_pip: bool = True,
    ) -> list[Path]:
        created: list[Path] = []
        for version in versions:
            self._ensure_interpreter(version)
            env_path = self.env_root / version.env_name
            if self._ensure_environment(version, env_path):
                created.append(env_path)
                self._ensure_pip(env_path, upgrade=upgrade_pip)
            if requirements:
                self._install_requirements(env_path, requirements)
            if packages:
                self._install_packages(env_path, packages)
        return created

    def _ensure_interpreter(self, version: VersionRequest) -> None:
        if self.tool is Tool.UV:
            uv_executable = _resolve_uv_executable()
            if uv_executable is None:
                msg = "uv is not available on PATH after installation"
                raise RuntimeError(msg)
            self._run_command(
                [uv_executable, "python", "install", version.raw],
                f"Installing Python {version.raw} via uv",
            )
        elif self.tool is Tool.PYENV:
            self._run_command(
                ["pyenv", "install", "-s", version.raw],
                f"Ensuring Python {version.raw} with pyenv",
            )
        elif self.tool is Tool.PYLAUNCHER:
            launcher = shutil.which("py")
            if launcher is None:
                msg = "Python launcher 'py' not found."
                raise RuntimeError(msg)
            self._run_command(
                [launcher, f"-{version.py_launcher_tag}", "-V"],
                f"Checking availability of Python {version.py_launcher_tag}",
            )
        elif self.tool is Tool.CURRENT:
            LOGGER.info(
                "Using current interpreter at %s for Python %s",
                sys.executable,
                version.raw,
            )
        else:  # Tool.AUTO should never reach here
            msg = f"Unhandled tool: {self.tool}"
            raise RuntimeError(msg)

    def _ensure_environment(self, version: VersionRequest, env_path: Path) -> bool:
        if env_path.exists():
            LOGGER.info("Environment already exists at %s", env_path)
            return False
        if self.dry_run:
            LOGGER.info("[dry-run] Would create %s", env_path)
            return False
        env_path.parent.mkdir(parents=True, exist_ok=True)
        if self.tool is Tool.UV:
            uv_executable = _resolve_uv_executable()
            if uv_executable is None:
                msg = "uv is not available on PATH after installation"
                raise RuntimeError(msg)
            self._run_command(
                [uv_executable, "venv", str(env_path), "--python", version.raw],
                f"Creating {env_path.name} via uv",
            )
        elif self.tool is Tool.PYENV:
            env = os.environ.copy()
            env["PYENV_VERSION"] = version.raw
            self._run_command(
                ["pyenv", "exec", "python", "-m", "venv", str(env_path)],
                f"Creating {env_path.name} via pyenv",
                env=env,
            )
        elif self.tool is Tool.PYLAUNCHER:
            launcher = shutil.which("py")
            if launcher is None:
                msg = "Python launcher 'py' not found."
                raise RuntimeError(msg)
            self._run_command(
                [launcher, f"-{version.py_launcher_tag}", "-m", "venv", str(env_path)],
                f"Creating {env_path.name} via py launcher",
            )
        elif self.tool is Tool.CURRENT:
            self._run_command(
                [sys.executable, "-m", "venv", str(env_path)],
                f"Creating {env_path.name} with current interpreter",
            )
        else:
            msg = f"Unhandled tool: {self.tool}"
            raise RuntimeError(msg)
        LOGGER.info("Created environment at %s", env_path)
        return True

    def _python_binary(self, env_path: Path) -> Path:
        python_name = "python.exe" if os.name == "nt" else "python"
        bin_dir = env_path / ("Scripts" if os.name == "nt" else "bin")
        return bin_dir / python_name

    def _ensure_pip(self, env_path: Path, *, upgrade: bool) -> None:
        python_bin = self._python_binary(env_path)
        if not python_bin.exists():
            msg = f"Interpreter not found inside {env_path}"
            raise RuntimeError(msg)
        self._run_command(
            [str(python_bin), "-m", "ensurepip", "--upgrade"],
            f"Bootstrapping pip in {env_path.name}",
        )
        if upgrade:
            self._run_command(
                [str(python_bin), "-m", "pip", "install", "--upgrade", "pip"],
                f"Upgrading pip in {env_path.name}",
            )

    def _install_requirements(
        self,
        env_path: Path,
        requirement_files: Sequence[Path],
    ) -> None:
        python_bin = self._python_binary(env_path)
        if not python_bin.exists():
            msg = f"Interpreter not found inside {env_path}"
            raise RuntimeError(msg)
        for requirement in requirement_files:
            if not requirement.exists():
                LOGGER.warning("Requirement file %s missing; skipping", requirement)
                continue
            self._run_command(
                [str(python_bin), "-m", "pip", "install", "-r", str(requirement)],
                f"Installing dependencies from {requirement} into {env_path.name}",
            )

    def _install_packages(self, env_path: Path, packages: Sequence[str]) -> None:
        if not packages:
            return
        python_bin = self._python_binary(env_path)
        if not python_bin.exists():
            msg = f"Interpreter not found inside {env_path}"
            raise RuntimeError(msg)
        self._run_command(
            [str(python_bin), "-m", "pip", "install", *packages],
            f"Installing packages {', '.join(packages)} into {env_path.name}",
        )

    def _run_command(
        self,
        command: Sequence[str],
        reason: str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        LOGGER.info(reason)
        LOGGER.debug("Command: %s", " ".join(command))
        if self.dry_run:
            LOGGER.info("[dry-run] Skipped execution")
            return
        try:
            run_command(command, env=dict(env) if env else None)
        except CommandError as exc:
            msg = f"Command failed ({reason}): {exc}"
            raise RuntimeError(msg) from exc


def detect_tool(
    preference: str,
    *,
    bootstrap_uv: bool = False,
    dry_run: bool = False,
) -> Tool:
    desired = Tool(preference)
    if desired is not Tool.AUTO:
        _ensure_tool_available(desired, bootstrap_uv=bootstrap_uv, dry_run=dry_run)
        if _tool_available(desired):
            return desired
        msg = f"Requested tool '{desired.value}' is not available on PATH"
        raise RuntimeError(msg)
    for candidate in (Tool.UV, Tool.PYENV, Tool.PYLAUNCHER, Tool.CURRENT):
        _ensure_tool_available(candidate, bootstrap_uv=bootstrap_uv, dry_run=dry_run)
        if _tool_available(candidate):
            return candidate
    msg = "No supported Python management tools detected"
    raise RuntimeError(msg)


def _tool_available(tool: Tool) -> bool:
    if tool is Tool.UV:
        return _resolve_uv_executable() is not None
    if tool is Tool.PYENV:
        return shutil.which("pyenv") is not None
    if tool is Tool.PYLAUNCHER:
        return shutil.which("py") is not None
    return tool is Tool.CURRENT


def _ensure_tool_available(
    tool: Tool,
    *,
    bootstrap_uv: bool,
    dry_run: bool,
) -> None:
    if tool is Tool.UV and _resolve_uv_executable() is None and bootstrap_uv:
        if dry_run:
            LOGGER.info("[dry-run] Would install uv via pip")
            return
        LOGGER.info("Installing uv via pip to provision interpreters")
        try:
            subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pip", "install", "--upgrade", "uv"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            msg = "Unable to install uv; install it manually or disable --bootstrap-uv"
            raise RuntimeError(msg) from exc


def _resolve_uv_executable() -> str | None:
    candidate = shutil.which("uv")
    if candidate:
        return candidate
    local = Path(sys.executable).with_name("uv.exe" if os.name == "nt" else "uv")
    if local.exists():
        return str(local)
    return None


def write_python_version(project_root: Path, version: VersionRequest) -> None:
    target = project_root / ".python-version"
    target.write_text(f"{version.raw}\n", encoding="utf-8")
    LOGGER.info("Pinned .python-version to %s", version.raw)


def update_tox_ini(
    project_root: Path,
    versions: Sequence[VersionRequest],
    *,
    tox_path: Path,
) -> None:
    resolved_tox_path = tox_path if tox_path.is_absolute() else project_root / tox_path
    config = configparser.ConfigParser()
    if resolved_tox_path.exists():
        config.read(resolved_tox_path, encoding="utf-8")
    if "tox" not in config:
        config["tox"] = {}
    env_names = ", ".join(version.tox_env for version in versions)
    config["tox"]["envlist"] = env_names
    for version in versions:
        section = f"testenv:{version.tox_env}"
        if section not in config:
            config[section] = {}
        config[section].setdefault(
            "basepython", f"python{version.major}.{version.minor}"
        )
    resolved_tox_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_tox_path.open("w", encoding="utf-8") as handle:
        config.write(handle)
    LOGGER.info("Updated %s with envlist=%s", resolved_tox_path, env_names)


def parse_versions(items: Iterable[str]) -> list[VersionRequest]:
    seen: set[str] = set()
    parsed: list[VersionRequest] = []
    for item in items:
        version = VersionRequest.parse(item)
        if version.raw in seen:
            continue
        seen.add(version.raw)
        parsed.append(version)
    if not parsed:
        msg = "At least one Python version must be provided"
        raise ValueError(msg)
    return parsed


def _configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


def _resolve_roots(project_root_arg: str, env_root_arg: str) -> tuple[Path, Path]:
    project_root = Path(project_root_arg).resolve()
    env_root = Path(env_root_arg)
    if not env_root.is_absolute():
        env_root = project_root / env_root
    env_root.mkdir(parents=True, exist_ok=True)
    return project_root, env_root


def _normalize_requirement_path(project_root: Path, raw: str) -> Path:
    candidate_path = Path(raw)
    if not candidate_path.is_absolute():
        candidate_path = project_root / candidate_path
    return candidate_path


def _collect_requirements(
    *,
    project_root: Path,
    explicit: Sequence[str],
    default_candidates: Sequence[str],
    include_auto: bool,
) -> list[Path]:
    requirements = [
        _normalize_requirement_path(project_root, item) for item in explicit if item
    ]
    requirements = _dedupe_preserve_order(requirements)
    if include_auto and not requirements:
        candidates = list(DEFAULT_AUTO_REQUIREMENT_FILES)
        if default_candidates:
            candidates.extend(default_candidates)
        for candidate in _dedupe_preserve_order(candidates):
            candidate_path = _normalize_requirement_path(project_root, candidate)
            if candidate_path.exists():
                LOGGER.info(
                    "Auto-including requirements file at %s",
                    candidate_path,
                )
                requirements.append(candidate_path)
    return _dedupe_preserve_order(requirements)


def _collect_packages(packages: Sequence[str]) -> list[str]:
    filtered = [pkg for pkg in packages if pkg]
    return _dedupe_preserve_order(filtered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision multiple Python interpreters and virtual environments.",
    )
    parser.add_argument(
        "versions",
        metavar="VERSION",
        nargs="+",
        help="Python versions to provision (e.g. 3.12.6 3.11)",
    )
    parser.add_argument(
        "--tool",
        choices=Tool.choices(),
        default=Tool.AUTO.value,
        help="Interpreter manager to use (default: auto-detect)",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Project root for generated metadata",
    )
    parser.add_argument(
        "--env-root",
        default=".",
        help="Directory where environments should be created",
    )
    parser.add_argument(
        "--requirements",
        action="append",
        help="Requirement files to install into each environment",
    )
    parser.add_argument(
        "--default-requirements",
        action="append",
        help="Candidate requirement files to auto-include when present",
    )
    parser.add_argument(
        "--package",
        dest="packages",
        action="append",
        help="Additional packages to install into each environment",
    )
    parser.add_argument(
        "--update-tox",
        action="store_true",
        help="Synchronize tox.ini with the requested versions",
    )
    parser.add_argument(
        "--tox-path",
        default="tox.ini",
        help="Path to tox.ini (default: %(default)s)",
    )
    parser.add_argument(
        "--write-python-version",
        action="store_true",
        help="Write .python-version pinned to the first version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing them",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--bootstrap-uv",
        action="store_true",
        help="Automatically install uv with pip if it is not available",
    )
    parser.add_argument(
        "--no-auto-requirements",
        action="store_true",
        help="Disable automatic inclusion of requirements.txt when present",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Skip pip --upgrade step after bootstrapping",
    )
    return parser


@dataclass(frozen=True)
class CLIArguments:
    versions: list[str]
    tool: str
    project_root: str
    env_root: str
    requirements: list[str]
    default_requirements: list[str]
    packages: list[str]
    bootstrap_uv: bool
    dry_run: bool
    no_auto_requirements: bool
    skip_pip_upgrade: bool
    verbose: bool
    write_python_version: bool
    update_tox: bool
    tox_path: str


class _ParsedNamespace(TypedDict):
    versions: list[str]
    tool: str
    project_root: str
    env_root: str
    requirements: list[str] | None
    default_requirements: list[str] | None
    packages: list[str] | None
    bootstrap_uv: bool
    dry_run: bool
    no_auto_requirements: bool
    skip_pip_upgrade: bool
    verbose: bool
    write_python_version: bool
    update_tox: bool
    tox_path: str


def _parse_cli_arguments(argv: Sequence[str] | None) -> CLIArguments:
    parser = build_parser()
    namespace = parser.parse_args(argv)

    raw = cast("_ParsedNamespace", vars(namespace))

    versions = [str(item) for item in raw["versions"]]
    requirements_seq = raw["requirements"]
    default_requirements_seq = raw["default_requirements"]
    packages_seq = raw["packages"]

    requirements = [str(item) for item in requirements_seq] if requirements_seq else []
    default_requirements = (
        [str(item) for item in default_requirements_seq]
        if default_requirements_seq
        else []
    )
    packages = [str(item) for item in packages_seq] if packages_seq else []

    return CLIArguments(
        versions=versions,
        tool=str(raw["tool"]),
        project_root=str(raw["project_root"]),
        env_root=str(raw["env_root"]),
        requirements=requirements,
        default_requirements=default_requirements,
        packages=packages,
        bootstrap_uv=bool(raw["bootstrap_uv"]),
        dry_run=bool(raw["dry_run"]),
        no_auto_requirements=bool(raw["no_auto_requirements"]),
        skip_pip_upgrade=bool(raw["skip_pip_upgrade"]),
        verbose=bool(raw["verbose"]),
        write_python_version=bool(raw["write_python_version"]),
        update_tox=bool(raw["update_tox"]),
        tox_path=str(raw["tox_path"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_arguments(argv)

    _configure_logging(verbose=args.verbose)

    versions = parse_versions(args.versions)
    tool = detect_tool(
        args.tool,
        bootstrap_uv=args.bootstrap_uv,
        dry_run=args.dry_run,
    )

    project_root, env_root = _resolve_roots(args.project_root, args.env_root)
    requirements = _collect_requirements(
        project_root=project_root,
        explicit=args.requirements,
        default_candidates=args.default_requirements,
        include_auto=not args.no_auto_requirements,
    )
    packages = _collect_packages(args.packages)

    manager = EnvManager(
        tool=tool,
        project_root=project_root,
        env_root=env_root,
        dry_run=args.dry_run,
    )
    created = manager.ensure_versions(
        versions,
        requirements,
        packages,
        upgrade_pip=not args.skip_pip_upgrade,
    )

    if args.write_python_version:
        write_python_version(project_root, versions[0])
    if args.update_tox:
        update_tox_ini(project_root, versions, tox_path=Path(args.tox_path))

    LOGGER.info("Provisioned %d environment(s)", len(created))
    return 0


if __name__ == "__main__":
    sys.exit(main())
