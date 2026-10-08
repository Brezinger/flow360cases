# Mathematical audit of plot_aeroacoustics.py — after corrections

Updated: **9 October 2026**. Corrections and plot regeneration were completed on
8 October 2026 on branch `fix/aeroacoustic-math-consistency`.

**Conclusion:** the identified implementation defects have been corrected and
the affected plot sets have been regenerated. Additive spectral powers use
window-energy-normalized PSD × frequency-bin width. Third-octave calculations
use complete IEC base-10 bands, and measurement comparisons use matching band
coverage and weighting. Remaining limitations involve finite-record resolution,
CFD convergence, stationary-tone assumptions and the measurement/calibration
basis. They do not justify another universal dB correction.

This is the audit of the corrected implementation. The original investigation,
including its earlier numerical examples and code line references, is preserved
in [the pre-correction audit](mathematical_audit_pre_corrections_2026-10-08.md).
Historical defects described there are not the current calculation.

Reviewed implementation: [plot_aeroacoustics.py](../plot_aeroacoustics.py), with
[calibration](../correct_lifter4b_acoustics.py) and
[regeneration](../regenerate_aeroacoustic_plots.py). Function names below refer to
the current code; obsolete line numbers have been removed.

Workspace script SHA-256, checked on 9 October 2026:
`d5298807a98a0b592609c3dec438aaaf6e5e01488ff83cc3851c9ff975f2b96e`.

## Scope and verification status

The review covers dimensionalization, CSV validation, source combination,
observer/surface time selection, Welch normalization, A-weighting, third-octave
bands, OASPL, harmonic extraction, PSD/amplitude displays, spectrograms,
measurement loading and calibration.

The correction work completed with **69 acoustic regression tests passing**.
They cover known RMS tones, off-bin and jointly fitted harmonics, Parseval power
normalization, complete IEC bands, PSD/ENBW scaling, missing bands, Nyquist
support, measurement weighting and bandwidth, independent physical references,
selection origins, coherent zero edges, surface alignment, spectrogram power
and single-frame rendering. All seven real source files passed timestamp
validation. These are the recorded implementation-validation results from
8 October; the 9 October update changes the documentation.

The original audit additionally checked 32 window/record/segment Parseval cases
and 35 real observer records. Those historical results remain in
`audit_results.json`; they have not been relabelled as fresh post-correction
results. Updated real-data checks and regenerated manifests provide the
post-correction evidence below.

## Corrected mathematical definitions and plot labels

Let `Gpp(f)` be the one-sided pressure PSD in Pa²/Hz and `df` the FFT-bin width.
The additive bin power is:

```text
q_k = Gpp(f_k) × df                         [Pa²]
L_band = 10 log10(sum(q_k in band) / p0²)  [dB]
p0 = 20 µPa
```

A-weighting multiplies bin power by `10**(A(f_k)/10)` before summation.
No extra one-sided factor of two is applied; SciPy already incorporates it.

Changing Welch to `scaling="density"` makes its direct output a **PSD**.
Multiplying by `df` and integrating gives mean-square pressure and therefore
**SPL/OASPL**, not PSD.

| Output | Current quantity and appropriate label |
|---|---|
| PSD spectrum | Pressure PSD, dB re (20 µPa)²/Hz; A-weighting stated explicitly when applied |
| Narrowband amplitude spectrum | Amplitude spectrum, dB re 20 µPa; retains Welch `scaling="spectrum"` |
| Third-octave spectrum | Integrated third-octave band SPL, dB or dBA |
| Overall level | OASPL, dB or dBA, with its frequency coverage identified |
| Harmonic plots | Coherent stationary sinusoidal-fit RMS levels, dB |
| Spectrogram | Frequency on the y-axis, elapsed seconds on the x-axis, pressure PSD on the colour scale |

The earlier Hann summation error was a normalization error: summing
amplitude-normalized `spectrum` bins overestimates additive power by a factor
of 1.5 for the periodic Hann window, or **1.7609 dB**, for identical samples and
bins. That correction is implemented. Subsequent changes to bands, selections
and estimators have separate, frequency-dependent effects.

The foundational calculations verified by the original investigation are retained:

- Acoustic pressure is dimensionalized as CSV pressure × rho × a², and time as
  CSV time × L_grid/a for these air cases. The force-coefficient reference
  velocity is retained for reporting, not used as the acoustic pressure scale. [1]
- A-weighting is applied to power before integration and normalized to 0 dB at
  1 kHz.
- BPF is blade count × RPM/60; the inspected studies have four blades.
- Total pressure equals thickness plus loading in the inspected observer arrays;
  coherent pressure sums preserve phase/interference.
- Revolution averaging uses the dimensional sample rate and RPM. No major
  period-length error was identified in the inspected files.

## Status of the audit findings

### 1. Complete upper third-octave band — corrected

Functions: `third_octave_spectrum_at_centers_db`,
`third_octave_spectrum_db`, and the shared calibration band calculation.

Bands are integrated from the full additive spectrum before limiting the
displayed centre-frequency range. The nominal 10 kHz band includes its complete
upper half rather than stopping at 10000 Hz. Its current IEC edges are
**8912.509–11220.185 Hz**.

POC2 step4 observer 3 gives **48.3530 dBA** for this full band, agreeing with
independent direct PSD integration. The original audit's 45.4162/48.3065 dBA
example used the old binary edges and is retained only in the archived audit.

Full-band Nyquist support is checked. An unsupported upper half-band is not
silently accepted as a complete band.

### 2. Simulation/test OASPL bandwidth — corrected

Functions: `measurement_matched_oaspl_db`,
`_measurement_matched_oaspl_values`, and the measurement comparison renderers.

Two quantities are now explicitly distinguished:

- **Literal OASPL:** FFT bins in 40–10000 Hz.
- **Measurement-matched OASPL:** the energy sum over the same complete
  third-octave bands supplied by the measurement workbook, nominally
  50, 63, …, 10000 Hz.

For the 24 consecutive IEC bands, the latter covers approximately
**44.668–11220.185 Hz**, subject to the available FFT-bin grid. Their energy sum
is not expected to equal literal 40–10000 Hz OASPL.

Measurement plots, signed simulation/test differences and exported comparison
CSVs use the matched quantity. Their filenames contain
`measurement_matched`. The workbook's reported overall is checked against
its band-energy sum. Calibration uses the same complete bands.

For POC2 step4 observer 3, literal A-weighted OASPL is **61.0585 dBA** and
measurement-matched OASPL is **61.1622 dBA**. This is a coverage difference,
not a blanket acoustic correction.

### 3. IEC band centres and edges — implemented

Functions: `_third_octave_centers`, `exact_third_octave_centers`,
`third_octave_band_edges`, and the calibration's shared conversion.

The old base-2 definitions have been replaced:

```text
Exact centre: f_c = 1000 × 10**(m/10), with integer m
Lower edge:   f_l = f_c / 10**0.05
Upper edge:   f_u = f_c × 10**0.05
```

Preferred nominal labels are mapped to exact centres; for example, **63 Hz**
maps to **63.0957 Hz**. Nonstandard labels and duplicate mapped measurement bands
are rejected. Plotting, matched comparisons and calibration share these
definitions. [3]

The measurement analyser's actual convention remains unverified. IEC base-10
is explicitly an **assumption for the workbook's nominal labels**. Implementing
IEC frequency definitions does not make FFT-bin integration a certified IEC
filter bank, especially for short records with strong leakage.

### 4. Empty bands, silence and invalid power — corrected

Functions: `_db_from_power`, full-band integration and spectral renderers.

- A band containing no FFT bins is **unavailable (NaN)** and produces a
  resolution diagnostic.
- Plotted curves retain NaN gaps instead of connecting across missing bands.
- True zero power has level **−infinity**, not an artificial finite floor.
- Negative/infinite power is rejected; NaN is reserved for unavailable results.
- OASPL axis limits handle zero or cancelling surface contributions without
  trying to use infinite bounds.

The current Pusher step3 and combined continuation still have no FFT bins in
the nominal **63 Hz** band for all five observers. This is an unresolved-data
limitation, now represented correctly, rather than a remaining fake-level bug.

### 5. Pusher physical metadata — corrected

Functions: `_timing_for_file` and `_reference_values_for_file`.

| Parameter | Current Pusher step3/step3B setting |
|---|---|
| RPM | 2186 |
| CFD steps/revolution | 1756 |
| Density | 1.1839124829 kg/m³ |
| Temperature | 298.15 K |
| Grid-unit length | 0.001 m |
| Reference velocity retained for reporting | 242.6524 m/s |

Per-step counts are also configured for steps 0, 1, 2 and 4. Explicit CLI
overrides remain available; an unknown Pusher step requires explicit timing
information rather than silently using the old default.

User observer ids 1–5 map to CSV ids 0–4 at **20°, 45°, 90°, 135°, 160°**.
The current Pusher study has observer radius **1000 ft = 304.8 m**, reported in
the analysis manifest. Historical files still require their own historical
case settings if those differ from the present preset.

### 6. Independent references before combining sources — corrected

Functions: `make_plots`, `combine_acoustic_datasets` and the regeneration loader.

Every source resolves its own physical references before compatibility checks.
The first source's inferred values no longer masquerade as explicit overrides
for every subsequent source. User-supplied overrides remain distinct from
automatic inference.

Combination checks reference quantities, observer columns, sampling interval,
time-grid alignment and active-window coverage. Duplicate source paths are
rejected by the plotting pipeline. Pressures are summed **coherently at original
timestamps**; independently calculated SPLs are not added.

The inspected Pusher continuation has no active overlap: step3B begins one
acoustic sample after step3 ends. Its pressure sum therefore acts as a
continuation. Other files must still be checked for duplicate/cumulative content
or genuinely separate contributions before combining them.

### 7. Common revolution-selection origin — corrected

Functions: `prepare_acoustic_dataset`, loading and plotting dispatch.

Single-file, combined and skipped processing now select revolutions relative to
each observer's common active pressure/thickness/loading interval, **after any
requested initial skip**. Absolute timestamps are retained.

Each revolution boundary is converted independently, avoiding accumulated
rounding drift. For CFD-duration skips, the first retained acoustic sample is
at or after the requested cut, allowing CSV interval-rounding tolerance.
For the current Pusher, **150 CFD steps correspond to 75 acoustic samples**.

### 8. Consistent total/surface time windows — corrected

Functions: `_apply_total_selection`, `_observer_pressure`, surface loading and
coherent surface-pair summation.

The total's selected physical timestamps are applied to corresponding surfaces.
The spectral helpers preserve valid zero edges and internal zeros in these
selected arrays; they do not independently trim each component and thereby
change its FFT duration.

Padding detection uses a common observer interval across total, thickness and
loading pressure. Low-level spectral helpers treat their input arrays as
authoritative: direct callers must pass the intended analysis interval.

Combined total files without matching combined surface inputs do not generate
a misleading uncombined surface analysis.

### 9. Input and frequency-support validation — implemented

Functions: `AcousticDataset.__post_init__`, `_validated_time_spacing`,
`_validate_spectral_input`, `_nperseg` and frequency/band validation.

The pipeline rejects nonfinite samples, invalid reference quantities,
nonincreasing/irregular timestamps, inconsistent component shapes, invalid
segment sizes, unsupported tones and incomplete band support above Nyquist.
Spectral analysis requires at least eight samples. Empty configured frequency
intervals are rejected rather than reported as valid integrated levels.

Timestamp interval variation up to **0.1%** is accepted as CSV rounding.
The inspected real files vary by approximately **0.009–0.025%**.
No silent interpolation/resampling of missing or irregular samples is performed.

### Additional finding: measurement weighting mismatch — corrected

Functions: `MeasurementSpectrum`, `render_third_octave_plots`,
measurement loading and `refresh_spectral_displays`.

Measurement spectra carry explicit A/Z weighting and band metadata. Renderers
reject incompatible weighting, and the regeneration helper overlays the
workbook's A-weighted data only on A-weighted spectra. The regenerated
unweighted POC2/Lifter4B spectra no longer contain the erroneous black dBA curve.

An A-weighted band cannot generally be converted exactly to an unweighted band
by subtracting only its centre-frequency correction: the within-band spectrum
is unknown. No such conversion is claimed.

### 10. Off-bin tone extraction — corrected, with estimator assumptions

Functions: `fitted_tone_spl_db`, `tone_spl_db` and BPF component renderers.

Known-frequency cosine/sine regression with a DC term replaces nearest-bin
tone extraction. Joint fitting is used for plotted harmonics, including BPF
components, and per-case fitted target frequencies are displayed. The earlier
Hann scalloping examples describe the replaced estimator.

The fit estimates a **stationary coherent tone** over the selected record.
It does not recover all frequency drift, modulation or neighbouring broadband
power. Short records and poorly separated target frequencies still limit the
physical interpretation; ill-conditioned fits are rejected.

### 11. Overlapping harmonic neighbourhoods — corrected in the plots

Functions: `harmonic_tone_spl_db`, `render_harmonic_content_plots` and BPF renderers.

Harmonic plots now show jointly fitted coherent RMS amplitudes, instead of
labelling overlapping third-octave neighbourhood powers as isolated harmonics.
A ninth harmonic is therefore not automatically attributed to a tenth-harmonic
band merely because that band includes it.

The explicitly named `harmonic_third_octave_spl_db` helper remains available
for legitimate band-neighbourhood analysis, now with IEC-width edges. Such
neighbourhoods still overlap and must not be summed as independent harmonic
powers. They are not the estimator used by the corrected harmonic plots.

### 12. Broadband comparison bandwidth — addressed with separate PSD plots

Functions: `spl_spectrum_db`, `psd_spectrum_db`,
`render_psd_plots` and `spectral_analysis_metadata`.

Amplitude spectra are retained for tone inspection. For a periodic Hann window:

```text
Amplitude-normalized spectrum = PSD × ENBW
ENBW = 1.5 × df
```

Their broadband floor therefore still depends on analysis bandwidth; this is
part of the retained amplitude-spectrum convention, not an uncorrected
normalization error. Separate `psd_spectrum*.png` outputs provide density
comparisons with explicit per-Hz units. Manifests report df, ENBW, segment size
and Welch segment count. Display frequency limits are separate from the literal
OASPL integration interval.

PSD removes the deterministic bandwidth factor from broadband comparisons;
different resolutions, leakage and statistical variability can still affect
individual plotted estimates.

### 13. Spectrogram resolution and units — corrected, with a tradeoff

Functions: `_spectrogram_spl_db` and `render_spectrogram_plots`.

The default window is **one propeller revolution**, not a fixed 512 samples.
`--spectrogram-window-s` provides a physical-duration override. Spectrograms use
`scaling="density"`, remove the global mean once, and use `detrend=False` for
individual frames to avoid suppressing slow content through repeated local
mean removal.

The colour scale is pressure PSD, time is elapsed seconds, and the title and
manifest report actual frequency resolution and frame count. Explicit cell
edges render a single frame with finite time width.

For the four-blade studies, one revolution contains four BPF cycles. This
improves BPF resolution, but short records may yield only a few frames. Longer
windows trade transient localization for frequency resolution; the plot does
not create temporal information absent from the input.

## Remaining limitations and follow-up work

### 14. Short-record resolution and statistical confidence — remains open

Default `--nperseg All` produces one full-record periodogram. Fifty-percent
overlap does not create extra segments in that case. Current examples span
approximately 2.87 revolutions for POC2 step4 observer 3, 1.86 for Lifter4B step3,
and 2.00 for Pusher step3.

Some bands contain only one or two bins or none. FFT leakage and assignment of
bins by their centre frequencies remain finite-record approximations, even with
correct IEC edges and normalization. The Hann result is a window-weighted
power estimate, not necessarily the uniform record RMS of a transient.

**Follow-up:** obtain longer stationary records and check convergence against
duration, window and Welch segment length. Integer `--nperseg` settings allow
averaging when sufficient data exist, at the cost of frequency resolution.
Zero padding/interpolation alone does not add independent information.
Manifests now disclose these resolution limits rather than hiding them.

### CFD temporal/spatial convergence — remains open

Pusher step3 has approximately **6.40 CFD steps** and **3.20 acoustic samples**
per 10 kHz period. Both time scales are reported explicitly.

Nyquist support is necessary but does not establish solver or source-resolution
accuracy. Postprocessing cannot recover physical attenuation, dispersion or
unresolved noise caused by insufficient CFD time/spatial resolution.
**Follow-up:** assess timestep, acoustic-output sampling, mesh and record-length
convergence over the frequency range used for conclusions.

### 15. Measurement geometry and calibration transfer — remains conditional

The Pusher observer radius is recorded correctly as 304.8 m. The POC2 comparison
and calibration still use workbook measurements **as reported**. The workbook's
distance/propagation normalization and analyser filter convention have not been
established by this plotting pipeline.

The separate microphone-coordinate file suggests radius around 15.45 in its
coordinate units. This is not sufficient evidence to assume metres or apply a
distance correction. No unverified spherical-spreading or atmospheric adjustment
has been added.

Lifter4B calibration remains:

```text
L_Lifter,calibrated = L_Lifter,CFD + L_POC2,measured − L_POC2,CFD
```

The relation is valid as a per-band multiplicative energy correction, followed
by an energy sum. Its physical transfer across geometries, RPMs and propagation
conditions is an engineering assumption, not independent validation.

**Follow-up:** establish measurement units, radius, normalization, propagation
conditions and analyser convention before interpreting absolute simulation/test
differences or the transferred estimate as predictions at a specified distance.

## Post-correction numerical examples

These values use the corrected selection and shared mathematical helpers:

| Dataset / observer | Literal OASPL, 40–10000 Hz | Measurement-matched OASPL | Full IEC 10 kHz band |
|---|---|---|---|
| POC2 step4 / 3 | 65.1281 dB / 61.0585 dBA | 61.1622 dBA | 48.3530 dBA |
| Pusher step3 / 3 | 75.7232 dB / 65.4922 dBA | Not evaluated for this study | Not listed here |
| Pusher step3 + step3B, skip 150 CFD / 3 | 75.7180 dB / 65.4909 dBA | Not evaluated for this study | Not listed here |

The last two rows do not imply that the nominal 63 Hz band is resolved. Their
literal OASPL sums available FFT-bin powers over the specified interval, whereas
a complete measurement-band comparison requires the requested bands to be
resolved and supported.

## Regenerated outputs and provenance

The completed regeneration includes A-weighted and unweighted outputs for:

- POC2 step3/step4 comparison and step4 simulation/test overlays.
- Lifter4B step3, step4 and POC2 measurement overlays.
- Pusher step3 alone and step3 + step3B after skipping 150 CFD steps.
- The available POC2/Lifter4B/Q40LT comparison.
- POC2-calibrated Lifter4B plots in `Lifter4B_corrected`.

Weighted/unweighted spectra are in their corresponding subfolders.
Measurement-matched overall plots and comparison CSVs identify their basis in
their names/labels. Surface analyses use matching selected time intervals where
surface inputs exist. Previous PDF reports and source snapshots retain the
versions described when they were produced.

Reproduction/provenance files:

- [Exact regeneration commands](../aeroacoustic_regeneration_commands.txt).
- [Regeneration summary](../aeroacoustic_regeneration_summary.json).
- `analysis_manifest.json` in each plot directory: physical references,
  selections, df, ENBW, segment count, band bin counts and sampling limits.
- [Implementation details and CLI examples](../aeroacoustic_fixes.md).
- [Regeneration notes](../aeroacoustic_regeneration_README.md).

The correction work is recorded in commits `b569492`, `31fc71d`, `4ba9bda`,
`1db2a37`, `79abb0d` and `9a9f022`.

## Reproduction of the corrected analysis

From the repository root, with the configured project interpreter:

```powershell
& 'C:/git/flow360cases/venv/Scripts/python.exe' -m unittest discover -s tests -p '*aeroacoustic*.py'
& 'C:/git/flow360cases/venv/Scripts/python.exe' DUC/results_postprocessing/regenerate_aeroacoustic_plots.py
```

The first command runs the current regression suite. The second regenerates
plots and their numerical exports; it requires the recorded CSV/workbook inputs
and the F: source for the optional Q40LT comparison.

The original `verify_math.py`, `audit_results.json`, `input_hashes.json` and
`verification.log` are **pre-correction diagnostic artifacts**. The diagnostic
script imports the live production code and writes `audit_results.json`;
running it unchanged is not a like-for-like reproduction of the archived
investigation and is not the validation command for this corrected audit.
Retain those historical results separately from the regenerated manifests.

## Primary references

[1] [Flow360 reference scaling](https://docs.flexcompute.com/projects/flow360/en/stable/user_guide/NonDimensionalization/Introduction/Introduction.html)
and [acoustic output/padding documentation](https://docs.flexcompute.com/projects/flow360/en/latest/user_guide/NonDimensionalization/NonDimOutputs/NonDimOutputs.html).

[2] [SciPy Welch documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
and [SciPy spectrogram documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html).

[3] [IEC 61260-1:2014 preview, clauses 5.2–5.4](https://preview.sist.si/sist-preview/13383/3c4ae3e762b540cc8111744cb8f0ae8e/IEC-61260-1-2014.pdf).

The references support definitions and normalization conventions. Local code,
regression tests and regenerated results establish the implementation status;
sampling and measurement-basis limitations remain as stated above.
