"""Safe restoration of user-owned pinned TRAIN without TEST or guessing."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiment_config as cfg
from train_reference_locator import (
    candidate_files, file_sha256, resolve_train_reference,
)


class PinnedTrainLocatorTests(unittest.TestCase):
    def test_original_valid_train_kept_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"reference_split"/"train_xref.csv"
            target.parent.mkdir()
            payload=b"index,s_m\n0,0.0\n1,0.2\n"
            target.write_bytes(payload)
            digest=hashlib.sha256(payload).hexdigest()
            result=resolve_train_reference(
                target,search_roots=(),expected_sha256=digest)
            self.assertEqual(result,target.resolve())
            self.assertEqual(target.read_bytes(),payload)

    def test_restore_exact_sha_from_nearby_sibling_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)/"Code"/"a_project"
            dest=project/"nmpc_rl"/"nmpc_casadi"/"reference_split"/"train_xref.csv"
            original=project/"nmpc_casadi"/"reference_split"/"train_xref.csv"
            original.parent.mkdir(parents=True)
            payload=b"known original TRAIN content\n"
            original.write_bytes(payload)
            digest=hashlib.sha256(payload).hexdigest()
            restored=resolve_train_reference(
                dest,search_roots=[project],expected_sha256=digest)
            self.assertEqual(restored,dest.resolve())
            self.assertEqual(restored.read_bytes(),original.read_bytes())
            self.assertEqual(file_sha256(restored),digest)
            self.assertTrue(original.exists())
            self.assertFalse((dest.parent/"._train_xref_verified_copy.tmp").exists())

    def test_restore_from_old_evaluation_source_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout=Path(tmp)/"nmpc_casadi"
            dest=checkout/"reference_split"/"train_xref.csv"
            copy=checkout/"results_rl"/"gamma090_one_episode_seed0"/"cad_eval_source_xref.csv"
            copy.parent.mkdir(parents=True)
            payload=b"exact pinned train clone\n"
            copy.write_bytes(payload)
            digest=hashlib.sha256(payload).hexdigest()
            actual=resolve_train_reference(
                dest,search_roots=[checkout],expected_sha256=digest)
            self.assertEqual(actual.read_bytes(),payload)
            self.assertEqual(copy.read_bytes(),payload)

    def test_wrong_sha_candidates_do_not_become_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            target=root/"nmpc_casadi"/"reference_split"/"train_xref.csv"
            fake=root/"nmpc_casadi"/"train_xref.csv"
            fake.parent.mkdir(parents=True)
            fake.write_text("some other CSV",encoding="utf8")
            expected=hashlib.sha256(b"true TRAIN").hexdigest()
            with self.assertRaisesRegex(FileNotFoundError,"Missing pinned TRAIN"):
                resolve_train_reference(target,search_roots=[root],
                                        expected_sha256=expected)
            self.assertFalse(target.exists())

    def test_existing_wrong_sha_not_overwritten_even_if_valid_copy_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            target=root/"nmpc_casadi"/"reference_split"/"train_xref.csv"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"wrong original")
            source=root/"exact.csv"
            source.write_bytes(b"valid training bytes")
            expected=file_sha256(source)
            with self.assertRaisesRegex(ValueError,"will NOT overwrite"):
                resolve_train_reference(target,source=source,
                                        expected_sha256=expected)
            self.assertEqual(target.read_bytes(),b"wrong original")

    def test_explicit_other_local_source_and_wrong_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            target=root/"reference_split"/"train_xref.csv"
            source=root/"other_copy.csv"
            source.write_bytes(b"trusted")
            good_sha=file_sha256(source)
            self.assertEqual(
                resolve_train_reference(target,source=source,
                                        expected_sha256=good_sha),target.resolve())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/"other_copy.csv"
            src.write_bytes(b"not valid")
            with self.assertRaisesRegex(ValueError,"incorrect SHA256"):
                resolve_train_reference(
                    root/"reference_split"/"train_xref.csv",
                    source=src,expected_sha256=hashlib.sha256(b"valid").hexdigest())

    def test_never_scan_test_xref_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            test=root/"nmpc_casadi"/"reference_split"/"test_xref.csv"
            test.parent.mkdir(parents=True)
            test.write_bytes(b"irrelevant test")
            self.assertNotIn(test.resolve(),list(candidate_files([root])))

    def test_runner_resolves_before_long_training_with_code_config(self):
        import run_experiment as runner
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"my_old_train.csv"
            source.write_bytes(b"verified local bytes")
            target=root/"reference_split"/"train_xref.csv"
            with (patch.object(cfg,"TRAIN_CSV",target),
                  patch.object(cfg,"TRAIN_CSV_SOURCE",source),
                  patch.object(cfg,"RESULTS_DIR",root/"results"),
                  patch.object(cfg,"IMAGES_DIR",root/"images"),
                  patch("train_reference_locator.TRAIN_REFERENCE_SHA256",
                        hashlib.sha256(source.read_bytes()).hexdigest()),
                  patch("run_experiment.verify_train_reference",
                        return_value=(target,hashlib.sha256(source.read_bytes()).hexdigest()))):
                # Default sha argument is bound at module definition; replace
                # the helper with a wrapper forwarding synthetic test digest.
                real=resolve_train_reference
                with patch("run_experiment.resolve_train_reference",
                           side_effect=lambda dest,source:real(
                               dest,source=source,
                               expected_sha256=file_sha256(source))):
                    epochs,digest=runner._validate_config()
                self.assertEqual(len(epochs),5)
                self.assertEqual(target.read_bytes(),source.read_bytes())
                self.assertEqual(digest,file_sha256(source))


if __name__=="__main__":
    unittest.main()
