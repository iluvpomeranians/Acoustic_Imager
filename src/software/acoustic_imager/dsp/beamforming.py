# dsp/beamforming.py
from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, pinv, eig


def music_spectrum(
    R: np.ndarray,
    angles: np.ndarray,
    f_signal: float,
    n_sources: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    speed_sound: float,
) -> np.ndarray:
    """
    Vectorized MUSIC spectrum over all angles (no Python loop).
    Same output semantics: float32, normalized to max=1.

    NOTE: Signature matches your main code usage, except we pass x_coords/y_coords/speed_sound
          as explicit inputs instead of relying on globals.
    """
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]

    En = eigvecs[:, n_sources:]
    Pn = En @ En.conj().T

    theta = np.deg2rad(angles).astype(np.float32)
    cth = np.cos(theta).astype(np.float32)
    sth = np.sin(theta).astype(np.float32)

    proj = (x_coords[:, None] * cth[None, :] +
            y_coords[:, None] * sth[None, :]).astype(np.float32)

    k = (2.0 * np.pi * float(f_signal) / float(speed_sound))
    A = np.exp(1j * k * proj).astype(np.complex64)

    PA = Pn @ A
    denom = np.einsum("ma,ma->a", A.conj(), PA).real
    denom = np.maximum(denom, 1e-12)

    spec = (1.0 / denom).astype(np.float32)

    m = float(spec.max()) if spec.size else 1.0
    spec /= (m + 1e-12)
    return spec


def esprit_estimate(
    R: np.ndarray,
    f_signal: float,
    n_sources: int,
    pitch: float,
    speed_sound: float,
) -> np.ndarray:
    """
    ESPRIT estimate. Signature matches your logic but avoids globals by taking pitch/speed_sound.
    """
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]

    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)

    if pitch == 0:
        return np.zeros(n_sources)

    val = -(psi * float(speed_sound)) / (2 * np.pi * float(f_signal) * float(pitch))
    val = np.clip(np.real(val), -1.0, 1.0)
    theta = np.arcsin(val)
    return np.degrees(theta)
