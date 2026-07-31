from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import soundfile as sf
except ImportError:
    sf = None


RESULTS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = (
    RESULTS_DIR
    / "aeroacoustics"
    / "case-cf29e49e-da21-43ad-80e2-61a6f97bec80_results_total_acoustics_v3.csv"
)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "audio"

AIR_SPECIFIC_HEAT_RATIO = 1.4
AIR_GAS_CONSTANT_J_KG_K = 287.05287

# Local Lifter4B Flow360 reference quantities.
REFERENCE_TEMPERATURE_K = 303.275
REFERENCE_DENSITY_KG_M3 = 1.064099
L_GRID_UNIT_M = 0.001
RPM = 1063.677
BLADE_COUNT = 4
LOOP_CROSSFADE_BPF_PERIODS = 2.0
FINAL_FADE_DURATION_S = 0.02

FLOW360_OBSERVER_NUMBERING_OFFSET = -1


def speed_of_sound_m_s(temperature_k: float) -> float:
    return float(np.sqrt(AIR_SPECIFIC_HEAT_RATIO * AIR_GAS_CONSTANT_J_KG_K * temperature_k))


def pressure_scale_pa(reference_density_kg_m3: float, temperature_k: float) -> float:
    return reference_density_kg_m3 * speed_of_sound_m_s(temperature_k) ** 2


def _csv_observer_index(user_observer_index: int, observer_offset: int) -> int:
    return user_observer_index + observer_offset


def _read_flow360_observer_pressure(
    input_file: Path,
    observer: int,
    observer_offset: int,
    reference_density_kg_m3: float,
    temperature_k: float,
    l_grid_unit_m: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    csv_observer = _csv_observer_index(observer, observer_offset)
    pressure_column = f"observer_{csv_observer}_pressure"

    time_values: list[float] = []
    pressure_values: list[float] = []

    with input_file.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"{input_file} has no CSV header.")
        fieldnames = [field.strip() for field in reader.fieldnames]
        if "time" not in fieldnames:
            raise ValueError(f"{input_file} has no 'time' column.")
        if pressure_column not in fieldnames:
            raise ValueError(f"{input_file} has no '{pressure_column}' column.")

        for row in reader:
            time_values.append(float(row["time"]))
            pressure_values.append(float(row[pressure_column]))

    if len(time_values) < 10:
        raise ValueError("The pressure signal is too short.")

    a = speed_of_sound_m_s(temperature_k)
    t = np.asarray(time_values, dtype=float) * (l_grid_unit_m / a)
    p = np.asarray(pressure_values, dtype=float) * pressure_scale_pa(
        reference_density_kg_m3,
        temperature_k,
    )
    return t, p, pressure_column


def _trim_zero_signal_edges(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonzero_indices = np.where(p != 0.0)[0]
    if nonzero_indices.size == 0:
        return t, p
    start = int(nonzero_indices[0])
    stop = int(nonzero_indices[-1]) + 1
    return t[start:stop], p[start:stop]


def _trim_to_integer_revolutions(
    signal_values: np.ndarray,
    sample_rate_hz: float,
    rpm: float,
) -> tuple[np.ndarray, int, int]:
    if rpm <= 0:
        raise ValueError("rpm must be positive.")

    revolution_count = max(1, int(round(len(signal_values) / sample_rate_hz * rpm / 60.0)))
    target_sample_count = int(round(revolution_count * sample_rate_hz * 60.0 / rpm))
    target_sample_count = min(target_sample_count, len(signal_values))
    if target_sample_count <= 0:
        raise ValueError("The computed revolution-aligned signal length is invalid.")

    return signal_values[:target_sample_count], revolution_count, target_sample_count


def _make_loopable_signal(
    signal_values: np.ndarray,
    sample_rate_hz: float,
    rpm: float,
    blade_count: int,
    crossfade_bpf_periods: float,
) -> tuple[np.ndarray, int]:
    if blade_count <= 0:
        raise ValueError("blade_count must be positive.")
    if crossfade_bpf_periods < 0:
        raise ValueError("crossfade_bpf_periods must be non-negative.")

    signal_values = np.asarray(signal_values, dtype=float)
    if crossfade_bpf_periods == 0:
        return signal_values, 0

    bpf_hz = blade_count * rpm / 60.0
    crossfade_samples = int(round(crossfade_bpf_periods * sample_rate_hz / bpf_hz))
    crossfade_samples = min(crossfade_samples, len(signal_values) // 4)
    if crossfade_samples < 2:
        return signal_values, 0

    fade_in = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, crossfade_samples))
    fade_out = 1.0 - fade_in
    loopable_signal = signal_values.copy()
    loopable_signal[:crossfade_samples] = (
        fade_out * signal_values[-crossfade_samples:]
        + fade_in * signal_values[:crossfade_samples]
    )
    return loopable_signal[:-crossfade_samples], crossfade_samples


def _apply_endpoint_fades(
    audio: np.ndarray,
    sample_rate_hz: float,
    fade_duration_s: float,
) -> tuple[np.ndarray, int]:
    if fade_duration_s < 0:
        raise ValueError("fade_duration_s must be non-negative.")

    fade_samples = min(int(round(fade_duration_s * sample_rate_hz)), len(audio) // 2)
    if fade_samples < 2:
        return audio, 0

    faded_audio = audio.copy()
    faded_audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
    faded_audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return faded_audio, fade_samples


def _write_wav(output_path: Path, audio: np.ndarray, sample_rate_hz: int) -> str:
    if sf is not None:
        sf.write(output_path, audio, sample_rate_hz, subtype="PCM_24")
        return "PCM_24 via soundfile"

    wavfile.write(output_path, sample_rate_hz, np.asarray(audio, dtype=np.float32))
    return "float32 via scipy.io.wavfile"


def _pressure_to_audio(
    t: np.ndarray,
    p: np.ndarray,
    target_fs: int,
    common_gain: float | None,
    repeat_count: int,
    rpm: float,
    blade_count: int,
    loop_crossfade_bpf_periods: float,
    final_fade_duration_s: float,
) -> tuple[np.ndarray, int, dict[str, float]]:
    t, p = _trim_zero_signal_edges(np.asarray(t, dtype=float), np.asarray(p, dtype=float))

    if len(t) < 10:
        raise ValueError("The pressure signal is too short after trimming zero edges.")

    dt_values = np.diff(t)
    dt = float(np.median(dt_values))

    if dt <= 0:
        raise ValueError("Time values must be strictly increasing.")

    relative_dt_variation = np.max(np.abs(dt_values - dt)) / dt
    if relative_dt_variation > 1e-3:
        raise ValueError(
            "Time samples are not sufficiently uniform. "
            "Interpolate onto a uniform time grid first."
        )

    fs_cfd = 1.0 / dt
    p_acoustic = signal.detrend(p, type="linear")

    ratio = Fraction(target_fs / fs_cfd).limit_denominator(10_000)
    up = ratio.numerator
    down = ratio.denominator
    p_audio = signal.resample_poly(p_acoustic, up, down)
    effective_fs = fs_cfd * up / down

    p_audio, revolution_count, revolution_aligned_samples = _trim_to_integer_revolutions(
        p_audio,
        sample_rate_hz=effective_fs,
        rpm=rpm,
    )
    p_audio, loop_crossfade_samples = _make_loopable_signal(
        p_audio,
        sample_rate_hz=effective_fs,
        rpm=rpm,
        blade_count=blade_count,
        crossfade_bpf_periods=loop_crossfade_bpf_periods,
    )

    peak_pressure = float(np.max(np.abs(p_audio)))
    rms_pressure = float(np.sqrt(np.mean(p_audio**2)))

    if peak_pressure == 0:
        raise ValueError("The acoustic pressure signal is zero.")

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

    audio_long = np.tile(audio, repeat_count)
    audio_long, final_fade_samples = _apply_endpoint_fades(
        audio_long,
        sample_rate_hz=effective_fs,
        fade_duration_s=final_fade_duration_s,
    )
    p_ref = 20e-6
    spl = 20.0 * np.log10(rms_pressure / p_ref)

    return audio_long, int(round(effective_fs)), {
        "CFD sample rate [Hz]": fs_cfd,
        "WAV sample rate [Hz]": effective_fs,
        "duration [s]": len(audio_long) / effective_fs,
        "RMS pressure [Pa]": rms_pressure,
        "peak pressure [Pa]": peak_pressure,
        "simulated SPL [dB re 20 uPa]": spl,
        "digital gain [1/Pa]": gain,
        "loop revolutions": revolution_count,
        "revolution-aligned samples": revolution_aligned_samples,
        "loop samples after crossfade": len(p_audio),
        "loop crossfade samples": loop_crossfade_samples,
        "loop crossfade duration [s]": loop_crossfade_samples / effective_fs,
        "final endpoint fade samples": final_fade_samples,
    }


def pressure_to_wav(
    input_file: str | Path,
    output_file: str | Path,
    observer: int,
    target_fs: int = 48_000,
    common_gain: float | None = None,
    repeat_count: int = 20,
    observer_offset: int = FLOW360_OBSERVER_NUMBERING_OFFSET,
    reference_density_kg_m3: float = REFERENCE_DENSITY_KG_M3,
    temperature_k: float = REFERENCE_TEMPERATURE_K,
    l_grid_unit_m: float = L_GRID_UNIT_M,
    rpm: float = RPM,
    blade_count: int = BLADE_COUNT,
    loop_crossfade_bpf_periods: float = LOOP_CROSSFADE_BPF_PERIODS,
    final_fade_duration_s: float = FINAL_FADE_DURATION_S,
) -> dict[str, float | str]:
    input_path = Path(input_file)
    output_path = Path(output_file)

    t, p, pressure_column = _read_flow360_observer_pressure(
        input_path,
        observer=observer,
        observer_offset=observer_offset,
        reference_density_kg_m3=reference_density_kg_m3,
        temperature_k=temperature_k,
        l_grid_unit_m=l_grid_unit_m,
    )
    audio, effective_fs, information = _pressure_to_audio(
        t,
        p,
        target_fs=target_fs,
        common_gain=common_gain,
        repeat_count=repeat_count,
        rpm=rpm,
        blade_count=blade_count,
        loop_crossfade_bpf_periods=loop_crossfade_bpf_periods,
        final_fade_duration_s=final_fade_duration_s,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav_encoding = _write_wav(output_path, audio, effective_fs)

    return {
        "input file": str(input_path),
        "output file": str(output_path),
        "observer": observer,
        "pressure column": pressure_column,
        "wav encoding": wav_encoding,
        **information,
    }


def _parse_observers(value: str) -> tuple[int, ...]:
    observers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not observers:
        raise argparse.ArgumentTypeError("Provide at least one observer.")
    return observers


def _output_file_for_observer(output_dir: Path, input_file: Path, observer: int) -> Path:
    return output_dir / f"{input_file.stem}_observer_{observer}.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WAV audio from Flow360 aeroacoustic CSV observer data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--observers",
        type=_parse_observers,
        default=(1,),
        help="Comma-separated observer numbers. Default maps observer 1 to CSV observer_0.",
    )
    parser.add_argument(
        "--csv-observer-offset",
        type=int,
        default=FLOW360_OBSERVER_NUMBERING_OFFSET,
        help="Offset from user observer number to CSV observer index. Default maps observer 1 to observer_0.",
    )
    parser.add_argument("--target-fs", type=int, default=48_000)
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--common-gain", type=float, default=None)
    parser.add_argument("--rho", type=float, default=REFERENCE_DENSITY_KG_M3)
    parser.add_argument("--temperature-k", type=float, default=REFERENCE_TEMPERATURE_K)
    parser.add_argument("--l-grid-unit", type=float, default=L_GRID_UNIT_M)
    parser.add_argument("--rpm", type=float, default=RPM)
    parser.add_argument("--blade-count", type=int, default=BLADE_COUNT)
    parser.add_argument(
        "--loop-crossfade-bpf-periods",
        type=float,
        default=LOOP_CROSSFADE_BPF_PERIODS,
        help="Cyclic loop crossfade length in blade-passing periods.",
    )
    parser.add_argument(
        "--final-fade-duration",
        type=float,
        default=FINAL_FADE_DURATION_S,
        help="Fade duration in seconds applied only to the start/end of the final WAV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for observer in args.observers:
        output_file = _output_file_for_observer(args.output_dir, args.input, observer)
        information = pressure_to_wav(
            input_file=args.input,
            output_file=output_file,
            observer=observer,
            target_fs=args.target_fs,
            common_gain=args.common_gain,
            repeat_count=args.repeat_count,
            observer_offset=args.csv_observer_offset,
            reference_density_kg_m3=args.rho,
            temperature_k=args.temperature_k,
            l_grid_unit_m=args.l_grid_unit,
            rpm=args.rpm,
            blade_count=args.blade_count,
            loop_crossfade_bpf_periods=args.loop_crossfade_bpf_periods,
            final_fade_duration_s=args.final_fade_duration,
        )
        print(f"Wrote {output_file}")
        for name, value in information.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
