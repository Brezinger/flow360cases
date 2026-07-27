from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def pressure_to_wav(
    input_file: str,
    output_file: str,
    target_fs: int = 48_000,
    pressure_column: int = 1,
    time_column: int = 0,
    common_gain: float | None = None,
) -> dict:
    """
    Convert an acoustic pressure time history to a WAV file.

    Input:
        Text file containing time [s] and acoustic/total pressure [Pa].

    common_gain:
        Optional fixed conversion factor from Pa to digital amplitude.
        Use the same value for several files when relative loudness
        between simulations must be retained.
    """

    data = np.loadtxt(input_file)

    if data.ndim != 2 or data.shape[1] <= max(time_column, pressure_column):
        raise ValueError("Input file does not contain the requested columns.")

    t = np.asarray(data[:, time_column], dtype=float)
    p = np.asarray(data[:, pressure_column], dtype=float)

    if len(t) < 10:
        raise ValueError("The pressure signal is too short.")

    dt_values = np.diff(t)
    dt = float(np.median(dt_values))

    if dt <= 0:
        raise ValueError("Time values must be strictly increasing.")

    # Warn if the samples are substantially nonuniform.
    relative_dt_variation = np.max(np.abs(dt_values - dt)) / dt
    if relative_dt_variation > 1e-3:
        raise ValueError(
            "Time samples are not sufficiently uniform. "
            "Interpolate onto a uniform time grid first."
        )

    fs_cfd = 1.0 / dt

    # Remove mean and slow linear drift.
    p_acoustic = signal.detrend(p, type="linear")

    # Rational approximation of the resampling ratio.
    ratio = Fraction(target_fs / fs_cfd).limit_denominator(10_000)
    up = ratio.numerator
    down = ratio.denominator

    # Polyphase resampling includes anti-aliasing filtering.
    p_audio = signal.resample_poly(p_acoustic, up, down)

    # Recalculate the effective sample rate after rational approximation.
    effective_fs = fs_cfd * up / down

    # Apply a short fade to prevent clicks during playback.
    fade_duration = 0.02
    fade_samples = min(
        int(fade_duration * effective_fs),
        len(p_audio) // 2,
    )

    if fade_samples > 1:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        p_audio[:fade_samples] *= fade_in
        p_audio[-fade_samples:] *= fade_out

    peak_pressure = float(np.max(np.abs(p_audio)))
    rms_pressure = float(np.sqrt(np.mean(p_audio**2)))

    if peak_pressure == 0:
        raise ValueError("The acoustic pressure signal is zero.")

    # SPL of the original simulated pressure.
    p_ref = 20e-6
    spl = 20.0 * np.log10(rms_pressure / p_ref)

    if common_gain is None:
        gain = 0.95 / peak_pressure
    else:
        if common_gain <= 0:
            raise ValueError("common_gain must be positive.")
        gain = common_gain

    audio = gain * p_audio

    if np.max(np.abs(audio)) > 1.0:
        raise ValueError(
            "The selected common gain causes clipping. "
            "Use a smaller common_gain."
        )

    audio_long = np.tile(audio, 20)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        output_path,
        audio_long,
        int(round(effective_fs)),
        subtype="PCM_24",
    )

    return {
        "CFD sample rate [Hz]": fs_cfd,
        "WAV sample rate [Hz]": effective_fs,
        "duration [s]": len(audio_long) / effective_fs,
        "RMS pressure [Pa]": rms_pressure,
        "peak pressure [Pa]": peak_pressure,
        "simulated SPL [dB re 20 µPa]": spl,
        "digital gain [1/Pa]": gain,
    }


if __name__ == "__main__":
    information = pressure_to_wav(
        input_file="observer_pressure.txt",
        output_file="propeller_noise.wav",
        target_fs=48_000,
    )

    for name, value in information.items():
        print(f"{name}: {value}")