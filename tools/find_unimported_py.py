#!/usr/bin/env python3
"""
List Python modules in this repo that are not referenced by explicit imports.

By default this script:
- Scans all *.py files under the repo root
- Parses Python AST for `import ...` and `from ... import ...`
- Optionally scans *.sh files for `python -m <module>` patterns (enabled by default)

This is a heuristic for cleanup / deprecation work. It will *not* catch:
- Dynamic imports (importlib, __import__, eval)
- Attribute-based usage after `import package`
- Entry-points invoked directly as scripts (python path/to/file.py)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
    "deprecated",
    "depricated",
}


@dataclass(frozen=True)
class PyFile:
    path: Path
    module: str


def _iter_files(root: Path, *, suffix: str, exclude_dirs: set[str]) -> Iterator[Path]:
    for p in root.rglob(f"*{suffix}"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        yield p


def _is_pkg_dir(d: Path) -> bool:
    return d.is_dir() and (d / "__init__.py").exists()


def _compute_module_name(root: Path, py_path: Path) -> str:
    rel = py_path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # drop .py
    return ".".join(parts)


def _read_text(path: Path) -> str:
    # Best-effort; we only need to parse imports.
    return path.read_text(encoding="utf-8", errors="ignore")


def _resolve_relative(module_of_file: str, level: int, module: str | None) -> str | None:
    """
    Resolve `from ...` imports to an absolute dotted module name, using the
    importing file's module path as context.
    """
    if level <= 0:
        return module
    base = module_of_file.split(".")[:-1]  # package of file
    drop = max(0, level - 1)
    if drop > len(base):
        return None
    prefix = base[: len(base) - drop]
    if module:
        return ".".join(prefix + module.split("."))
    return ".".join(prefix) if prefix else None


def _imports_from_ast(tree: ast.AST, *, module_of_file: str) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    refs.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            abs_mod = _resolve_relative(module_of_file, int(node.level or 0), node.module)
            if abs_mod:
                refs.add(abs_mod)
                # Heuristic: "from pkg import submod" likely references pkg.submod.
                for alias in node.names:
                    if alias.name and alias.name != "*":
                        refs.add(f"{abs_mod}.{alias.name}")
    return refs


# Roughly match: python [flags...] -m module.name
# Works for: "python -u -m pkg.mod", "conda run ... python -u -m pkg.mod", etc.
# Important: do not let the flag-consumer eat the "-m" token.
PY_M_RE = re.compile(r"\bpython\S*\b(?:\s+-(?!m\b)[^\s]+)*\s+-m\s+([A-Za-z0-9_\.]+)\b")


def _imports_from_shell(text: str) -> set[str]:
    return {m.group(1) for m in PY_M_RE.finditer(text)}


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_importable_module_name(name: str) -> bool:
    if not name:
        return False
    return all(_IDENT_RE.match(part) is not None for part in name.split("."))


def _load_py_files(root: Path, exclude_dirs: set[str]) -> list[PyFile]:
    out: list[PyFile] = []
    for p in _iter_files(root, suffix=".py", exclude_dirs=exclude_dirs):
        out.append(PyFile(path=p, module=_compute_module_name(root, p)))
    return out


def _load_packages(root: Path, py_files: Iterable[PyFile]) -> set[str]:
    # A directory is a package if it has __init__.py. Compute importable package names.
    pkgs: set[str] = set()
    for pf in py_files:
        rel = pf.path.relative_to(root)
        d = root
        parts: list[str] = []
        for part in rel.parts[:-1]:
            d = d / part
            parts.append(part)
            if _is_pkg_dir(d):
                pkgs.add(".".join(parts))
    return pkgs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root to scan (default: .)")
    ap.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude (repeatable). Matches any path component.",
    )
    ap.add_argument(
        "--no-shell",
        action="store_true",
        help="Do not scan *.sh files for `python -m ...` module references.",
    )
    ap.add_argument(
        "--print-referenced",
        action="store_true",
        help="Also print the referenced import/module set (debug).",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS) | {str(x) for x in args.exclude_dir}

    py_files_all = _load_py_files(root, exclude_dirs)
    py_files: list[PyFile] = []
    non_importable: list[PyFile] = []
    for pf in py_files_all:
        if _is_importable_module_name(pf.module):
            py_files.append(pf)
        else:
            non_importable.append(pf)
    pkgs = _load_packages(root, py_files)

    referenced: set[str] = set()
    for pf in py_files:
        try:
            tree = ast.parse(_read_text(pf.path), filename=str(pf.path))
        except SyntaxError:
            continue
        referenced |= _imports_from_ast(tree, module_of_file=pf.module)

    if not args.no_shell:
        for sh in _iter_files(root, suffix=".sh", exclude_dirs=exclude_dirs):
            referenced |= _imports_from_shell(_read_text(sh))

    # Candidate modules we might consider "importable".
    candidates: set[str] = {pf.module for pf in py_files} | pkgs

    # Mark a candidate as referenced if it's imported explicitly.
    unreferenced = sorted(candidates - referenced)

    print(f"root: {root}")
    print(f"python_files: {len(py_files_all)}")
    print(f"python_files_importable: {len(py_files)}")
    print(f"python_files_non_importable: {len(non_importable)}")
    print(f"packages: {len(pkgs)}")
    print(f"referenced_imports: {len(referenced)}")
    print(f"unreferenced_modules: {len(unreferenced)}")

    if args.print_referenced:
        print("\n# referenced")
        for m in sorted(referenced):
            print(m)

    print("\n# unreferenced (explicit-import heuristic)")
    for m in unreferenced:
        print(m)

    if non_importable:
        print("\n# non-importable module paths (not valid dotted identifiers)")
        for pf in sorted(non_importable, key=lambda x: str(x.path)):
            rel = pf.path.relative_to(root)
            print(f"{rel} -> {pf.module}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
