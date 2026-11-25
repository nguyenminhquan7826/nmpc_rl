"""Named, checksum-pinned 8 m references for the train-and-evaluation branch.

'pinned' preserves the original legacy TRAIN checksum. The two new references
are exploratory geometric tests, not screened or certified stable controllers.
"""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from training_initialization import TRAIN_REFERENCE_SHA256

ROOT = Path(__file__).resolve().parent
TRACK_SHA256 = {
    "pinned": TRAIN_REFERENCE_SHA256,
    "spiral_8m": "e8e7fe2eea66fac80753197163f11ef1698c669367a01bbe2da254017538fc4a",
    "zigzag_8m": "d6135506e5b55ab4168e9b9a912b0b86cf30fc8c61ad4c305c8d037944ad2db5",
}
NEW_TRACKS = ("spiral_8m", "zigzag_8m")
TRACK_CHOICES = ("pinned",) + NEW_TRACKS

def file_digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def track_path(name):
    if name not in NEW_TRACKS:
        raise ValueError(f"Not a bundled new trajectory: {name!r}")
    return ROOT / "trajectory_references" / f"xref_{name}.csv"

def resolve_named_track(name):
    if name not in NEW_TRACKS:
        raise ValueError(f"Choose a new track from {NEW_TRACKS}")
    path = track_path(name).resolve(strict=True)
    digest = file_digest(path)
    if digest != TRACK_SHA256[name]:
        raise ValueError(f"Reference checksum mismatch for {name}: {digest}")
    return path, digest

def identify_checkpoint_train(config):
    """Accept legacy checkpoints and explicitly registered named-trajectory checkpoints."""
    digest = config.get("train_sha256")
    name = config.get("train_trajectory", "pinned")
    if name not in TRACK_SHA256 or digest != TRACK_SHA256[name]:
        raise ValueError(
            "Checkpoint TRAIN reference is not checksum-registered: "
            f"trajectory={name!r}, sha256={digest!r}"
        )
    return name, digest
