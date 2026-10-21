"""Locate a byte-identical *pinned* TRAIN trajectory, without touching TEST.

A fresh Git checkout does not contain reference_split/train_xref.csv:
reference_split is gitignored to keep original private research data out
of the source repository. Reuse a verified local copy if one exists
in a nearby project/worktree or a previous evaluation's byte-for-byte
copied cad_eval_source_xref.csv.

Never synthesize TRAIN from unrelated tracked XREFs, never select a
file based on filename alone, never read TEST candidates, and never
overwrite a present file even if its checksum is wrong.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

from training_initialization import TRAIN_REFERENCE_SHA256


# Finite-depth, named TRAIN-only file paths. No unbounded recursive
# filesystem scan; do not scan disks outside the nearby project tree.
LOCAL_CANDIDATE_PATTERNS = (
    "reference_split/train_xref.csv",
    "train_xref.csv",
    "nmpc_casadi/reference_split/train_xref.csv",
    "*/reference_split/train_xref.csv",
    "*/nmpc_casadi/reference_split/train_xref.csv",
    "*/*/nmpc_casadi/reference_split/train_xref.csv",
    "*/*/*/nmpc_casadi/reference_split/train_xref.csv",
    "reference_split/_rl_validated_train_copy.csv",
    "results_rl/*/cad_eval_source_xref.csv",
    "nmpc_casadi/results_rl/*/cad_eval_source_xref.csv",
    "*/nmpc_casadi/results_rl/*/cad_eval_source_xref.csv",
    "*/results_rl/*/cad_eval_source_xref.csv",
)
MAX_CANDIDATES = 500


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_search_roots(dest):
    """Search sibling checkouts and limited ancestors, never whole drive."""
    nm=dest.parent.parent
    # Include current NMPC directory, checkout, enclosing project, Code,
    # but not arbitrary file roots (D:\, C:\ or user home).
    return [nm,*list(nm.parents)[:3]]


def candidate_files(roots):
    """Deterministic fixed-depth path search; never match 'test_xref.csv'."""
    seen=set()
    for root in roots:
        folder=Path(root).expanduser()
        if not folder.is_dir():
            continue
        for pattern in LOCAL_CANDIDATE_PATTERNS:
            for possible in sorted(folder.glob(pattern)):
                # Check the exact names before reading any content; saved
                # byte-identical TRAIN copy has a distinct trusted filename.
                if not possible.is_file() or possible.suffix.lower()!=".csv":
                    continue
                canonical=possible.resolve()
                if canonical in seen:
                    continue
                seen.add(canonical)
                if len(seen)>MAX_CANDIDATES:
                    raise RuntimeError("Too many local TRAIN candidates; set TRAIN_CSV_SOURCE explicitly")
                yield canonical


def resolve_train_reference(destination, *, source=None, search_roots=None,
                            expected_sha256=TRAIN_REFERENCE_SHA256):
    """Return an existing or safely restored checksum-pinned local TRAIN CSV.

    If source is given, it is the user's centrally configured local path,
    NOT a web URL. When a pinned local source is found, copy exact bytes
    into reference_split/train_xref.csv. This is a single trajectory
    folder, not a new project-level output folder.
    """
    dest=Path(destination).expanduser().resolve(strict=False)
    if dest.exists():
        if not dest.is_file() or dest.suffix.lower()!=".csv":
            raise ValueError(f"TRAIN source is not a CSV file: {dest}")
        current=file_sha256(dest)
        if current != expected_sha256:
            raise ValueError(
                f"Existing TRAIN checksum mismatch (will NOT overwrite): {dest}\n"
                f"Expected SHA256: {expected_sha256}\nObserved: {current}")
        return dest

    if source is not None:
        candidate=Path(source).expanduser().resolve(strict=False)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Configured TRAIN_CSV_SOURCE does not exist: {candidate}")
        candidates=[candidate]
    else:
        roots=_default_search_roots(dest) if search_roots is None else search_roots
        candidates=candidate_files(roots)

    checked=0
    for candidate in candidates:
        if candidate==dest:
            continue
        checked+=1
        if file_sha256(candidate)!=expected_sha256:
            if source is not None:
                raise ValueError(
                    f"TRAIN_CSV_SOURCE has incorrect SHA256: {candidate}\n"
                    f"Expected {expected_sha256}")
            continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        # Copy and verify the complete bytes BEFORE atomic installation.
        temp=dest.with_name("._train_xref_verified_copy.tmp")
        if temp.exists() or dest.exists():
            raise FileExistsError(
                f"Existing input or staging path will not be overwritten: {temp}")
        try:
            shutil.copy2(candidate,temp)
            if file_sha256(temp)!=expected_sha256:
                raise RuntimeError("Copied TRAIN bytes changed; refused")
            if dest.exists():
                raise FileExistsError(f"TRAIN input appeared concurrently: {dest}")
            os.replace(temp,dest)
        finally:
            if temp.exists():
                temp.unlink()
        print(json.dumps({
            "train_reference_status":"RESTORED_EXACT_PINNED_LOCAL_COPY",
            "source":str(candidate), "destination":str(dest),
            "sha256":expected_sha256,
            "not_test":True,
        }),flush=True)
        return dest

    raise FileNotFoundError(
        f"Missing pinned TRAIN trajectory: {dest}\n"
        f"No byte-identical TRAIN CSV was found in {checked} nearby candidates. "
        "Because reference_split/ is ignored by Git, git pull cannot restore it.\n"
        "Copy the ORIGINAL train_xref.csv from your previous training "
        "checkout into reference_split/, OR set TRAIN_CSV_SOURCE to its "
        "existing absolute file path in experiment_config.py. "
        f"Required SHA256: {expected_sha256}. "
        "Do not rename xref_states_smoothed.csv or use TEST.")
