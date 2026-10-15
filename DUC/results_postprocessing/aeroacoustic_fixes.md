# Aeroacoustic audit fixes

Branch `fix/aeroacoustic-math-consistency`, implemented 8 October 2026.

The pressure/time reference scaling and one-sided PSD × df normalization were
verified. The following changes address the separate audit findings; they do not
apply another universal dB shift.

| Finding | Corrected behavior |
|---|---|
| Hann amplitude-spectrum bins summed as power | Integrate window-energy-normalized PSD × df; retain amplitude spectra separately. |
| Truncated top third-octave band | Integrate complete bands before limiting displayed centres. The IEC 10 kHz band spans 8912.509–11220.185 Hz. |
| Binary versus modern IEC definitions | Share base-10 centres and edges in plotting and calibration. This is FFT-bin integration using IEC frequency definitions, not a certified IEC filter bank. |
| Different CFD/test OASPL bandwidths | Retain literal 40–10000 Hz OASPL and calculate a separate overall from the exact complete measurement bands for comparisons and CSV errors. |
| Fake levels for empty bands | No-bin bands are NaN gaps. True zero power is −∞. Invalid negative/infinite power is rejected. |
| A-weighted measurements on unweighted plots | Carry A/Z metadata, validate direct renderers, and restrict the workbook overlay to A-weighted plots. |
| Wrong automatic Pusher references | Infer 2186 RPM, current per-step counts, 298.15 K, 1.1839124829 kg/m³ and 0.001 m grid unit. Step3B shares step3's 1756 CFD steps/revolution. |
| First source's metadata applied to combined sources | Resolve each source independently, then require compatible physical references and time grids. Reject duplicate source paths. |
| Inconsistent revolution origin | Select from each observer's common active pressure/thickness/loading window after any initial skip. Convert each boundary independently to avoid rounding drift. |
| Different surface/total cuts | Apply the total's physical time bounds to every surface; retain valid zero edges and internal zeros. |
| CFD skip rounding ambiguity | Retain the first acoustic sample at or after the requested duration, allowing CSV interval rounding. Current Pusher: 150 CFD steps = 75 acoustic samples. |
| Missing validation | Reject nonfinite data/references, irregular or nonincreasing timestamps, invalid segment sizes and insufficient Nyquist support. CSV interval rounding within 0.1% is accepted; resampling must be explicit. |
| Nearest-bin tone scalloping | Fit cosine/sine and DC terms at known physical frequencies. Display per-case frequencies. |
| Overlapping neighbourhoods presented as isolated harmonics | Fit harmonics jointly and plot their coherent stationary RMS amplitudes. The explicit band-neighbourhood helper retains its documented overlap limitation. |
| Broadband amplitude levels depend on bandwidth | Add PSD plots in dB re (20 µPa)²/Hz; disclose df, ENBW and averaging. Amplitude spectra remain available for tone inspection. |
| Spectrogram display | Restore the historical 512-sample, 50%-overlap SPL view with per-window mean removal, normalized time, and 0–120 dB colors. Explicit physical windows remain available for PSD diagnostics. |

Low-level spectral helpers treat supplied arrays as authoritative. The dataset
pipeline clips Flow360 padding once using a common observer window. Arrays passed
directly to spectral helpers should already represent the intended time interval.

Pusher user observer ids 1–5 map to CSV ids 0–4 at 20°, 45°, 90°, 135°, 160° and
1000 ft (304.8 m). No invented propagation adjustment is applied. The workbook's
analyser convention is assumed to be IEC base-10; its distance and normalization
basis remain unknown. Lifter4B calibration remains the per-band relation
`L_Lifter + L_measured_POC2 − L_CFD_POC2`, followed by an energy sum, conditional
on the measurement basis and transfer across operating points/geometries.

## Verification and reproduction

Tests cover known RMS/off-bin tones, jointly fitted harmonics, Parseval power,
IEC edges and the full upper band, PSD/ENBW, missing bands, Nyquist errors,
measurement weighting and bandwidth, independent references, common cuts and
coherent zero edges, surface alignment, spectrogram BPF power and single-frame
rendering. All seven real source files pass timestamp validation.

Independent real-data checks for POC2 step4 observer 3 give 65.1281 dB /
61.0585 dBA for literal 40–10000 Hz, 61.1622 dBA for the matched complete
measurement bands, and 48.3530 dBA for the complete IEC 10 kHz band. Pusher step3
observer 3 gives 75.7232 dB / 65.4922 dBA; combined step3 + step3B after the
150-CFD-step cut gives 75.7180 dB / 65.4909 dBA. Its nominal 63 Hz band remains
unresolved, correctly reported as NaN.

From the repository root, with the configured interpreter:

```powershell
& 'C:\git\flow360cases\venv\Scripts\python.exe' -m unittest discover -s tests -p '*aeroacoustic*.py'
& 'C:\git\flow360cases\venv\Scripts\python.exe' DUC/results_postprocessing/regenerate_aeroacoustic_plots.py
```

Exact per-study commands are in `aeroacoustic_regeneration_commands.txt` and
sources/settings in `aeroacoustic_regeneration_summary.json`. Per-study
`analysis_manifest.json` files report physical references and resolution.
Existing reports retain their historical source snapshots. The
[current mathematical audit](aeroacoustic_math_audit/mathematical_audit.md) describes
the corrected implementation; the
[original audit](aeroacoustic_math_audit/mathematical_audit_pre_corrections_2026-10-08.md)
is preserved separately.

## CLI examples

Pusher step3 alone, no skipped steps:

```powershell
& 'C:\git\flow360cases\venv\Scripts\python.exe' DUC/results_postprocessing/plot_aeroacoustics.py `
  --csv 'DUC/results_postprocessing/Pusher/aeroacoustics/step3/case-ed80b35a-78a0-413a-a8a5-5c158fb87127_total_acoustics_v3.csv' `
  --output-dir 'DUC/results_postprocessing/Pusher/aeroacoustic_plots/step3' --a-weighting
```

Pusher continuation, skipping the first 150 CFD steps:

```powershell
& 'C:\git\flow360cases\venv\Scripts\python.exe' DUC/results_postprocessing/plot_aeroacoustics.py `
  --csv 'DUC/results_postprocessing/Pusher/aeroacoustics/step3/case-ed80b35a-78a0-413a-a8a5-5c158fb87127_total_acoustics_v3.csv' `
  --csv 'DUC/results_postprocessing/Pusher/aeroacoustics/step3B/total_acoustics_v3.csv' `
  --combine-csv --n-timesteps-skip 150 --skip-unit cfd `
  --output-dir 'DUC/results_postprocessing/Pusher/aeroacoustic_plots/step3B' --a-weighting
```

`--no-a-weighting` selects unweighted spectral displays. OASPL calculations
always report both literal unweighted and A-weighted values. `--revolutions
zero:1-3` selects the second and third revolutions after the initial cut.
`--nperseg All` is the default full-record periodogram; an integer enables shorter
Welch segments and averaging when enough samples exist. Spectrograms alone default
to the historical 512-sample window, 50% overlap, spectrum scaling, per-window
mean removal, time fraction axis, and 0–120 dB color range. `--spectrogram-window-s
0.06` opts into a 60 ms physical-window PSD view independently of RPM and sample rate.
Pusher spectrograms use 78 samples per window with 50% overlap, giving 44 frames
for the current retained records, matching Lifter4B's normalized frame width.
This affects only spectrograms; the shorter window gives coarser frequency bins
(about 410 Hz). All other spectra and acoustic calculations retain their settings.

## Remaining physical and statistical limits

Current records span only a few revolutions. Some low bands contain one or two
bins or none. A full-record Hann periodogram has no ensemble averaging and gives
a window-weighted power estimate. FFT leakage and assigning bins by their centres
remain finite-record approximations. Longer stationary records and
duration/window/segment convergence checks are necessary; interpolation or zero
padding does not add independent frequency information.

Pusher step3 has about 6.4 CFD steps and 3.2 acoustic samples per 10 kHz period.
Nyquist support does not establish CFD time/spatial accuracy. Postprocessing
cannot recover physical attenuation, dispersion or unresolved source noise.

Sinusoidal fits estimate coherent stationary tones at specified frequencies;
modulation and broadband power remain in residuals. Spectrogram windows trade
time localization for frequency resolution. The transferred POC2 correction is
an engineering calibration, not independent validation of Lifter4B.
