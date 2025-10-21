"""Multi-version Python environment manager for x_make_py_venv_x."""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Tool(Enum):
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
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
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
            logging.info(
                "Using current interpreter at %s for Python %s",
                sys.executable,
                version.raw,
            )
        else:  # Tool.AUTO should never reach here
            msg = f"Unhandled tool: {self.tool}"
            raise RuntimeError(msg)

    def _ensure_environment(self, version: VersionRequest, env_path: Path) -> bool:
        if env_path.exists():
            logging.info("Environment already exists at %s", env_path)
            return False
        if self.dry_run:
            logging.info("[dry-run] Would create %s", env_path)
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
        logging.info("Created environment at %s", env_path)
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
                logging.warning("Requirement file %s missing; skipping", requirement)
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
        env: dict[str, str] | None = None,
    ) -> None:
        logging.info(reason)
        logging.debug("Command: %s", " ".join(command))
        if self.dry_run:
            logging.info("[dry-run] Skipped execution")
            return
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as exc:
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
    if tool is Tool.CURRENT:
        return True
    return False


def _ensure_tool_available(
    tool: Tool,
    *,
    bootstrap_uv: bool,
    dry_run: bool,
) -> None:
    if tool is Tool.UV and _resolve_uv_executable() is None and bootstrap_uv:
        if dry_run:
            logging.info("[dry-run] Would install uv via pip")
            return
        logging.info("Installing uv via pip to provision interpreters")
        try:
            subprocess.run(
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
    logging.info("Pinned .python-version to %s", version.raw)


def update_tox_ini(
    project_root: Path,
    versions: Sequence[VersionRequest],
    *,
    tox_path: Path,
) -> None:
    config = configparser.ConfigParser()
    if tox_path.exists():
        config.read(tox_path, encoding="utf-8")
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
    tox_path.parent.mkdir(parents=True, exist_ok=True)
    with tox_path.open("w", encoding="utf-8") as handle:
        config.write(handle)
    logging.info("Updated %s with envlist=%s", tox_path, env_names)


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
        default=[],
        help="Requirement files to install into each environment",
    )
    parser.add_argument(
        "--default-requirements",
        action="append",
        default=["requirements.txt", "x_0_make_all_x/requirements.txt"],
        help="Candidate requirement files to auto-include when present",
    )
    parser.add_argument(
        "--package",
        dest="packages",
        action="append",
        default=[],
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    versions = parse_versions(args.versions)
    tool = detect_tool(
        args.tool,
        bootstrap_uv=args.bootstrap_uv,
        dry_run=args.dry_run,
    )
    project_root = Path(args.project_root).resolve()
    env_root = Path(args.env_root)
    if not env_root.is_absolute():
        env_root = project_root / env_root
    env_root.mkdir(parents=True, exist_ok=True)
    requirement_args = list(args.requirements)
    requirements = [
        (Path(path) if Path(path).is_absolute() else project_root / path)
        for path in requirement_args
    ]
    default_requirement_candidates = list(args.default_requirements or [])
    if not args.no_auto_requirements and not requirements:
        for candidate in default_requirement_candidates:
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = project_root / candidate
            if candidate_path.exists():
                logging.info(
                    "Auto-including requirements file at %s",
                    candidate_path,
                )
                requirements.append(candidate_path)
    requirements = list(dict.fromkeys(requirements))
    package_args = list(args.packages or [])
    packages = list(dict.fromkeys(package_args))

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
        tox_path = Path(args.tox_path)
        if not tox_path.is_absolute():
            tox_path = project_root / tox_path
        update_tox_ini(project_root, versions, tox_path=tox_path)

    logging.info("Provisioned %d environment(s)", len(created))
    return 0


if __name__ == "__main__":
    sys.exit(main())
