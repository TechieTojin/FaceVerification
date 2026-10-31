"""Template encryption, legacy detection, dimension/model validation, metric maths."""
from __future__ import annotations

import numpy as np
import pytest
from cryptography.fernet import Fernet

from backend.config import Settings, settings
from backend.services.biometric import templates as T
from backend.services.biometric.providers import NEAR_IDENTITY_DISTANCE, fake_identity_vector


def test_roundtrip_is_normalised_float32():
    v = np.arange(1, 513, dtype=np.float64)
    blob = T.encrypt_vectors([v], "ArcFace")
    assert v.tobytes() not in blob  # ciphertext, not plaintext
    dec = T.decrypt_vectors(blob)
    assert dec.model == "ArcFace" and dec.dim == 512 and dec.vectors.dtype == np.float32
    assert np.linalg.norm(dec.vector) == pytest.approx(1.0, abs=1e-5)


def test_legacy_v21_blob_is_refused():
    legacy = T._get_fernet().encrypt(np.ones(512, dtype=np.float64).tobytes())
    with pytest.raises(T.LegacyTemplateError):
        T.decrypt_vectors(legacy)


def test_wrong_key_cannot_decrypt():
    blob = Fernet(Fernet.generate_key()).encrypt(b"FVT2xxxx")
    with pytest.raises(T.TemplateDecryptError):
        T.decrypt_vectors(blob)


def test_dimension_mismatch_is_rejected_not_truncated():
    with pytest.raises(T.TemplateMismatchError):
        T.cosine_distance(np.ones(512), np.ones(128))


def test_model_mismatch_is_rejected():
    tmpl = T.decrypt_vectors(T.encrypt_vectors([np.ones(512)], "Facenet512"))
    with pytest.raises(T.TemplateMismatchError):
        T.compare_to_template(np.ones(512), tmpl, "ArcFace")


def test_zero_and_nan_vectors_rejected():
    with pytest.raises(T.TemplateMismatchError):
        T.l2_normalise(np.zeros(512))
    with pytest.raises(T.TemplateMismatchError):
        T.l2_normalise(np.array([np.nan] * 512))


def test_cosine_distance_examples():
    assert T.cosine_distance(np.array([1.0, 2, 2]), np.array([2.0, 1, 2])) == pytest.approx(1 - 8 / 9)
    assert T.cosine_distance(np.array([1.0, 0, 0]), np.array([0.0, 1, 0])) == pytest.approx(1.0)
    assert T.cosine_distance(np.array([1.0, 0]), np.array([-1.0, 0])) == pytest.approx(2.0)


def test_fake_near_identities_have_documented_distances():
    base = fake_identity_vector(1)
    for code, dist in NEAR_IDENTITY_DISTANCE.items():
        assert T.cosine_distance(base, fake_identity_vector(code)) == pytest.approx(dist, abs=1e-6)


def test_build_template_mean_and_consistency():
    samples = [fake_identity_vector(1), fake_identity_vector(13)]
    built = T.build_template(samples)
    assert built.sample_count == 2
    assert built.consistency == pytest.approx(0.20, abs=1e-6)
    assert np.linalg.norm(built.vector) == pytest.approx(1.0, abs=1e-5)


def test_missing_encryption_key_raises():
    s = Settings.model_construct(secret_key="x" * 40, embedding_encryption_key="")
    with pytest.raises(RuntimeError, match="not set"):
        s.get_fernet_key()


def test_malformed_encryption_key_raises():
    s = Settings.model_construct(secret_key="x" * 40, embedding_encryption_key="short")
    with pytest.raises(RuntimeError, match="invalid length"):
        s.get_fernet_key()


def test_fake_provider_refused_outside_test_env(monkeypatch):
    with pytest.raises(ValueError, match="only permitted"):
        Settings(secret_key="x" * 40, inference_provider="fake", app_env="production", _env_file=None)


def test_low_bcrypt_rounds_refused_outside_test_env():
    with pytest.raises(ValueError, match="BCRYPT_ROUNDS"):
        Settings(secret_key="x" * 40, bcrypt_rounds=4, app_env="development", inference_provider="deepface",
                 _env_file=None)


def test_settings_require_secret_key():
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(secret_key="", _env_file=None)
    assert settings.app_env == "test"
