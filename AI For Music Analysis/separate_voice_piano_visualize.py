#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip3 install -U demucs librosa soundfile matplotlib numpy
# 

"""
Input:  mp4 / m4a / common audio file
Output:
  1. extracted WAV from the input media
  2. Demucs-separated vocals.wav and piano.wav
  3. waveform comparison figure
  4. spectrogram comparison figure
  5. RMS energy comparison figure

Example:
  python separate_voice_piano_visualize.py input.m4a --out results
  python separate_voice_piano_visualize.py input.mp4 --out results --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


MODEL_NAME = "htdemucs_6s"
TARGET_SR = 44100
TARGET_STEMS = ("vocals", "piano")

# Visualization colors
LINE_COLORS = ("red", "pink", "blue")
SPECTROGRAM_CMAP = "bwr_r"  # red-white-blue

def check_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Cannot find `{name}` in PATH. Please install it first."
        )


def check_python_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(
            f"Cannot import Python module `{name}`. "
            f"Install it with: pip install -U {name}"
        )


def safe_stem(path: Path) -> str:
    """
    Produce a filesystem-safe, stable stem name.
    This avoids problems caused by Chinese characters, spaces, or punctuation
    in the original filename when locating Demucs output folders.
    """
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    if not base:
        base = "audio"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def run_command(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")

    if proc.stdout.strip():
        print(proc.stdout[-3000:])


def extract_audio_to_wav(input_path: Path, wav_path: Path, sr: int = TARGET_SR) -> None:
    """
    Convert mp4/m4a/video container to a clean 44.1 kHz stereo WAV.
    -vn discards video streams.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(sr),
        str(wav_path),
    ]
    run_command(cmd)


def run_demucs(
    wav_path: Path,
    demucs_root: Path,
    device: str = "auto",
    segment: float | None = None,
    shifts: int = 1,
    overlap: float = 0.25,
) -> Path:
    """
    Run Demucs htdemucs_6s.
    Output folder should be:
      demucs_root / MODEL_NAME / wav_path.stem
    """
    demucs_root.mkdir(parents=True, exist_ok=True)

    expected_track_dir = demucs_root / MODEL_NAME / wav_path.stem
    if expected_track_dir.exists():
        shutil.rmtree(expected_track_dir)

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        MODEL_NAME,
        "-o",
        str(demucs_root),
        "--float32",
        "--clip-mode",
        "clamp",
        "--shifts",
        str(shifts),
        "--overlap",
        str(overlap),
    ]

    if device != "auto":
        cmd += ["-d", device]

    if segment is not None:
        cmd += ["--segment", str(segment)]

    cmd.append(str(wav_path))
    run_command(cmd)

    if expected_track_dir.exists():
        return expected_track_dir

    # Fallback: search for a folder containing the expected stems.
    candidates = list((demucs_root / MODEL_NAME).glob("*/vocals.wav"))
    for vocals_path in candidates:
        candidate_dir = vocals_path.parent
        if (candidate_dir / "piano.wav").exists():
            return candidate_dir

    raise FileNotFoundError(
        f"Cannot locate Demucs output folder under {demucs_root / MODEL_NAME}"
    )


def load_mono(path: Path, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32. Stereo is downmixed only for visualization.
    The actual Demucs output files remain stereo WAV files on disk.
    """
    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    y = y.astype(np.float32)
    return y, loaded_sr


def align_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Truncate all signals to the same length for fair plotting.
    """
    min_len = min(len(y) for y in signals.values())
    return {name: y[:min_len] for name, y in signals.items()}


def plot_waveforms(
    signals: Dict[str, np.ndarray],
    sr: int,
    out_path: Path,
    title: str,
) -> None:
    signals = align_signals(signals)
    names = list(signals.keys())
    n = len(names)

    max_abs = max(float(np.max(np.abs(signals[name]))) for name in names)
    max_abs = max(max_abs, 1e-6)

    fig, axes = plt.subplots(n, 1, figsize=(16, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, names)):
        y = signals[name]
        t = np.arange(len(y)) / sr
        ax.plot(
            t,
            y,
            linewidth=0.5,
            color=LINE_COLORS[idx % len(LINE_COLORS)],
        )
        ax.set_title(name)
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)

def plot_spectrograms(
    signals: Dict[str, np.ndarray],
    sr: int,
    out_path: Path,
    title: str,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> None:
    signals = align_signals(signals)
    names = list(signals.keys())
    n = len(names)

    fig, axes = plt.subplots(n, 1, figsize=(16, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        y = signals[name]
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(stft)
        db = librosa.amplitude_to_db(mag, ref=np.max)

        librosa.display.specshow(
            db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="log",
            ax=ax,
            cmap=SPECTROGRAM_CMAP,
        )
        ax.set_title(name)
        ax.set_ylabel("Frequency")

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)

def rms_envelope(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    return rms.astype(np.float32)


def plot_rms_comparison(
    signals: Dict[str, np.ndarray],
    sr: int,
    out_path: Path,
    title: str,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> None:
    signals = align_signals(signals)

    fig, ax = plt.subplots(figsize=(16, 5))

    for idx, (name, y) in enumerate(signals.items()):
        rms = rms_envelope(y, frame_length=frame_length, hop_length=hop_length)
        t = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=hop_length,
        )
        ax.plot(
            t,
            rms,
            linewidth=1.0,
            label=name,
            color=LINE_COLORS[idx % len(LINE_COLORS)],
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate vocals and piano from mp4/m4a and visualize waveforms/spectrograms."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input mp4/m4a/audio file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("voice_piano_results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Demucs device. Use cuda for NVIDIA GPU, mps for Apple Silicon if supported.",
    )
    parser.add_argument(
        "--segment",
        type=float,
        default=None,
        help="Optional Demucs segment length in seconds. Useful when GPU memory is insufficient.",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Number of random shifts for Demucs inference. Higher may improve quality but is slower.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between Demucs windows.",
    )

    args = parser.parse_args()

    input_path: Path = args.input.expanduser().resolve()
    out_dir: Path = args.out.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    check_executable("ffmpeg")
    check_python_module("demucs")

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    demucs_root = out_dir / "separated"

    name = safe_stem(input_path)
    extracted_wav = work_dir / f"{name}.wav"

    print(f"\n[1/4] Extracting audio to WAV: {extracted_wav}")
    extract_audio_to_wav(input_path, extracted_wav, sr=TARGET_SR)

    print(f"\n[2/4] Running Demucs model: {MODEL_NAME}")
    track_dir = run_demucs(
        extracted_wav,
        demucs_root=demucs_root,
        device=args.device,
        segment=args.segment,
        shifts=args.shifts,
        overlap=args.overlap,
    )

    vocals_path = track_dir / "vocals.wav"
    piano_path = track_dir / "piano.wav"

    missing = [str(p) for p in (vocals_path, piano_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Demucs did not produce the expected stems:\n" + "\n".join(missing)
        )

    print(f"\n[3/4] Loading audio for visualization")
    original, sr = load_mono(extracted_wav, sr=TARGET_SR)
    vocals, _ = load_mono(vocals_path, sr=TARGET_SR)
    piano, _ = load_mono(piano_path, sr=TARGET_SR)

    signals = align_signals(
        {
            "Original mixture": original,
            "Separated vocals": vocals,
            "Separated piano": piano,
        }
    )

    original = signals["Original mixture"]
    vocals = signals["Separated vocals"]
    piano = signals["Separated piano"]
    vocals_plus_piano = vocals + piano

    wave_png = out_dir / "waveform_original_vs_vocals_vs_piano.png"
    spec_png = out_dir / "spectrogram_original_vs_vocals_vs_piano.png"
    rms_png = out_dir / "rms_energy_original_vs_vocals_vs_piano.png"
    overlay_png = out_dir / "waveform_original_vs_vocals_plus_piano.png"

    print(f"\n[4/4] Saving visualization figures")
    plot_waveforms(
        signals,
        sr=sr,
        out_path=wave_png,
        title="Waveform comparison: original mixture vs separated vocals vs separated piano",
    )

    plot_spectrograms(
        signals,
        sr=sr,
        out_path=spec_png,
        title="Log-frequency spectrogram comparison: original mixture vs separated vocals vs separated piano",
    )

    plot_rms_comparison(
        signals,
        sr=sr,
        out_path=rms_png,
        title="RMS energy comparison: original mixture vs separated vocals vs separated piano",
    )

    plot_waveforms(
        {
            "Original mixture": original,
            "Vocals + piano only": vocals_plus_piano,
        },
        sr=sr,
        out_path=overlay_png,
        title="Waveform comparison: original mixture vs vocals + piano only",
    )

    corr_vp = safe_corr(original, vocals_plus_piano)

    print("\nDone.")
    print(f"Extracted WAV:       {extracted_wav}")
    print(f"Demucs output dir:   {track_dir}")
    print(f"Vocals stem:         {vocals_path}")
    print(f"Piano stem:          {piano_path}")
    print(f"Waveform figure:     {wave_png}")
    print(f"Spectrogram figure:  {spec_png}")
    print(f"RMS figure:          {rms_png}")
    print(f"Overlay figure:      {overlay_png}")
    print(f"Corr(original, vocals + piano): {corr_vp:.4f}")
    print(
        "\nNote: original ≠ vocals + piano in general, because the full Demucs "
        "6-stem model may also assign energy to drums, bass, guitar, and other."
    )


if __name__ == "__main__":
    main()