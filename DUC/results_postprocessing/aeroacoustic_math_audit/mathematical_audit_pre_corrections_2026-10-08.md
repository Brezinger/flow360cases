# Mathematical audit of plot_aeroacoustics.py

Date: 08 October 2026. This is an investigation, not an implementation change.
The production script and existing plot sets were left unchanged.

**Conclusion:** the corrected additive spectral-power normalization is sound.
Other problems remain in band definitions and comparison coverage, data selection,
automatic case metadata, tone estimation, resolution reporting and validation.
Some are confirmed defects; others are limitations of the chosen analysis.
They should not be treated as another universal dB correction.

Audited script SHA-256:
`7a08c0d0433680018439acbe699573577d2cdcf38cc5b76ad634dcda3e9930b0`.
Code line numbers below refer to this version.

## Scope and verification

All mathematical paths in the script were read, including dimensionalization,
CSV loading, continuation sums, transient/revolution cuts, PSD and amplitude
spectra, A-weighting, overall and third-octave levels, tone/harmonic extraction,
surface sums, revolution averaging, spectrograms, measurement loading and CLI
routing. The related Lifter4B calibration and regeneration scripts were inspected
to establish how the outputs are actually used.

The 19 existing acoustic tests pass. Additional independent checks cover:

- A known 1 Pa RMS, 1 kHz sinusoid: 93.979400 dB, recovered by the OASPL calculation.
- 32 full-band Parseval cases: odd/even lengths, Hann, Hamming, rectangular and
  flat-top windows, full records and multiple overlapping Welch segments. Maximum
  relative error is approximately 8.4e-16.
- Real POC2 step3/step4 and Lifter4B step3/step4, Pusher step3, the short step3B
  continuation alone, and step3 + step3B after excluding 150 CFD steps: 35
  observer records in total, each checked against Hann-weighted time-domain power.
- Known off-bin tones, white noise, band-edge leakage, harmonic overlap,
  spectrogram resolution, invalid sampling and inconsistent selection paths.
- The original measurement workbook: its reported overall values equal the
  energy sum of its 24 supplied A-weighted bands to numerical precision.

Machine-readable results are in `audit_results.json`; `verify_math.py` reproduces
the calculations. The standalone step3B record is deliberately included as a
short-record diagnostic, not as a sufficient standalone acoustic assessment.

## Confirmed implementation and comparison defects

### 1. The top plotted third-octave band is incomplete

Code: `_band_limited_mean_square_spectrum` at line 627;
`third_octave_spectrum_db` at line 701;
`render_third_octave_plots` at line 1237.

The script first discards all FFT bins above 10000 Hz. It then forms the band
whose binary centre is 10079.368 Hz and whose full edges are approximately
8979.696–11313.708 Hz. The plotted point therefore contains only its lower part.
The related `correct_lifter4b_acoustics.py:cfd_band_levels` integrates the full
band instead. These two outputs disagree even for the same input pressure.

Recalculating the same band with its full width gives the following increases
over the plotted A-weighted point. This isolates the cutoff issue; both versions
here retain the current binary convention.

| Data | Observer(s) | Full minus plotted top-band level |
|---|---|---|
| POC2 step4 | 1–5 | +1.94 to +2.89 dBA |
| Lifter4B step4 | 1–5 | +2.74 to +4.61 dBA |
| Pusher step3 | 1–5 | +1.44 to +3.64 dBA |
| Pusher combined, skip 150 CFD steps | 1–5 | +1.44 to +2.96 dBA |

For example, POC2 observer 3 has 45.4162 dBA in the plotted partial top band and
48.3065 dBA in the full band. Lifter4B observer 1 has 52.5629 versus 57.1649 dBA.

**Mitigation:** distinguish a requested range of *band centre labels* from an
integration-frequency cutoff. Full third-octave bands should be evaluated before
applying a display limit. If a partial band is intentional, label it as partial
and compare it only with identically truncated measurement data. Share the band
definition between plotting and calibration.

### 2. Simulation and measured overall levels do not use identical bandwidths

Code: `oaspl_a_weighted_db` at line 638;
`load_poc2x_1000rpm_measurement_data` at line 1130;
`print_oaspl_differences` at line 760.

Simulation OASPL uses bins between 40 and 10000 Hz. The measurement workbook's
reported OASPL is exactly the energy sum of the supplied bands with nominal
centres 50, 63, …, 10000 Hz. A complete top band extends above 10000 Hz; the
bottom band begins above 40 Hz. Thus the printed simulation-minus-measurement
error combines a physical discrepancy with unequal integration coverage.

Using full binary bands with the workbook's centre labels, POC2 observer 3 gives
61.1691 dBA instead of the script's 61.0585 dBA: +0.1106 dB. Lifter4B observer 1
changes from 63.4039 to 64.0310 dBA: +0.6271 dB. This is a coverage comparison,
not a proposed global correction.

The displayed bands also omit the bins between 40 and 44.194 Hz, explaining why
their energy sum need not equal the script's OASPL even with the top band left
partial. There is no displayed nominal 40 Hz band.

**Mitigation:** retain clearly labelled 40–10000 Hz OASPL if desired, and compute
a separate measurement-matched overall level from the exact same complete bands
as the measurement. Use that quantity for the comparison error and calibration.

### 3. Current bands use a different convention from modern IEC bands

Code: `_third_octave_centers` at line 684 and `_third_octave_bands` at line 695;
also `correct_lifter4b_acoustics.py:nominal_to_exact_centers`.

The script uses centres `1000 * 2**(m/3)` and edge factor `2**(1/6)`.
IEC 61260-1:2014 specifies base-10 frequency ratios: for third-octave bands,
centres are `1000 * 10**(m/10)` and the edge factor is `10**0.05`.
Binary bands are a meaningful historical convention, but they should not be
silently equated with current IEC bands in a measurement overlay. [3]

The difference in centre frequency is small, but hard assignment of a few FFT
bins can amplify it considerably. Keeping the same pressure, PSD and A-weighting:

| Real-data example | Band label | Base-10 minus binary full-band level |
|---|---|---|
| Lifter4B step3, observer 4 | 160 Hz | +8.2724 dBA |
| Lifter4B step4, observer 4 | 160 Hz | +5.4168 dBA |
| POC2 step4, observer 1 | 100 Hz | +3.5259 dBA |

In the first example, the binary band contains FFT bins at 149.247, 158.575 and
167.902 Hz. The base-10 band also contains the bin at 177.230 Hz. Its contribution
accounts for the difference. Other bands decrease when a bin moves out.

**Mitigation:** obtain the measurement analyser's band convention and filter
settings, then implement that convention consistently. For current IEC
comparisons, use base-10 centres/edges. This alone does not make short-record,
hard-bin integration a standards-compliant filter bank or remove leakage.

### 4. Empty bands are reported as finite acoustic levels

Code: `_db_from_power` at line 553;
`third_octave_spectrum_db` at line 701;
finite-value filtering at line 1255 and in surface-band plotting.

Zero power is replaced by the smallest positive floating-point number, producing
approximately **−2982.547 dB**. A band with no available FFT bins consequently
receives a finite level and passes `np.isfinite` plotting filters. Zero signal and
negative power receive the same value, hiding the distinction between silence,
missing resolution and invalid power.

This affects the current Pusher data: the binary 62.5 Hz band contains no FFT
bins for **all five observers**, both in step3 and in the processed continuation.
The available bin spacing is about 18.2 Hz, wider than that band's 14.47 Hz width.

**Mitigation:** report no-bin bands as unavailable, with their bin counts. Do not
connect curves through unavailable bands. Treat mathematical zero as zero power
or −infinity in dB, and reject negative/nonfinite powers when appropriate. A
display floor can be applied separately and explicitly.

### 5. Pusher metadata defaults are wrong without explicit overrides

Code: `_timing_for_file` at line 890 and `_reference_values_for_file` at line 909.

Pusher paths have angle handling but no physical-reference or RPM preset. A
plain step3 invocation therefore receives Lifter/old-case defaults:

| Parameter | Inferred default | Pusher step3 value |
|---|---|---|
| RPM | 1063.677 | 2186 |
| CFD steps/revolution | 3840 | 1756 |
| Density, kg/m³ | 1.064099 | 1.1839124829 |
| Temperature, K | 303.275 | 298.15 |

The pressure scale is then 0.7787 dB too low and the frequency axis is 0.8558%
too high. For real Pusher observer 3, the net OASPL error is −0.7793 dB unweighted
and −0.7327 dBA. The requested BPF also shifts from 145.733 to 70.912 Hz, and CFD
skip durations and revolution selections use the wrong RPM/step information.

**Existing regenerated Pusher plots used explicit correct overrides**, so this
finding does not mean those outputs used the wrong metadata.

**Mitigation:** add an explicit study/step configuration, preferably checked
against the actual case metadata, and refuse unconfigured cases. Keep CLI
overrides available. Historical files should use their historical case settings,
not automatically inherit a later edited simulation configuration.

### 6. Combined-file loading can conceal incompatible physical references

Code: `make_plots` at lines 1960–1997; compatibility checks at line 394.

`make_plots` resolves the first file's RPM and references, then passes those
resolved numbers as explicit overrides when loading every other file. The
combiner's matching-reference check consequently sees identical values even when
the per-file inferred settings should differ.

A captured call using a POC2 path followed by a Q40LT path passes the POC2
1000 RPM, 303.275 K, 1.064099 kg/m³ and 0.001 m grid unit to both files.
Q40LT's own preset is 1002 RPM, 283.275 K, 1.149 kg/m³ and a 1 m grid unit.
The actual timestamp-grid check may reject some such combinations, but it is
not a reliable substitute for checking the genuine references before overriding.

**Mitigation:** preserve the user's overrides separately from inferred values,
resolve every source independently, and validate compatibility on those results.
Mixed physical units need dimensionalization before alignment; incompatible
operating points should not silently be merged.

### 7. Revolution selections change origin between processing paths

Code: `_slice_revolutions` at line 266;
`prepare_acoustic_dataset` at line 451;
loading/selection dispatch in `make_plots` at line 1986.

Ordinary `--revolutions` selection starts at the first CSV row, including padded
arrival time. The combined/skip path starts at each observer's first active
sample, after any requested skip. The same range can therefore select different
physical data merely by enabling a different CLI path.

A controlled record with 700 leading padded samples and 1000 samples/revolution
retains 1300 active samples with ordinary `zero:0-2`, versus 2000 active samples
with the prepared path. This can change duration, tone bins, leakage and levels.

**Mitigation:** define one selection origin and apply it to all paths. Prefer an
explicit valid observer interval followed by the same skip and revolution cut.
Preserve absolute timestamps alongside elapsed plotting time.

### 8. Initial cuts are not propagated to surface-resolved data

Code: `load_surface_resolved_datasets` at line 493;
`prepare_acoustic_dataset` at line 451;
surface time-history and pair plotting calls in `make_plots`.

Prepared total pressures have their initial interval zeroed, but surface files
are reloaded without that skip. A revolution range, when present, is also
re-applied relative to the CSV origin instead of the prepared observer interval.
Surface-pair and total plots can consequently describe different time windows.

On Lifter4B step4, observer 3, requesting a 150-acoustic-sample cut leaves the
reloaded blade1 signal at 11521 samples rather than 11371. Its uncut OASPL differs
from the correctly cut surface signal by +0.2372 dB in this example.

**Mitigation:** retain an explicit selection mask/absolute time interval from the
total preprocessing and apply it to each corresponding surface dataset before
computing surface levels or sums. Independently rebasing surface time histories
should also be avoided when relative phase is being interpreted.

### 9. Missing validation can silently change the quantity being calculated

Code: `AcousticDataset.sample_spacing_s` at line 168;
`load_acoustic_dataset` at line 286;
`tone_spl_db` at line 660;
band limiting at line 627.

Single-file processing infers sampling rate from the mean timestamp difference
without checking finite values, monotonicity, uniformity, missing samples or
positive physical references. A nonmonotonic synthetic time sequence is accepted
and processed by the FFT. Multi-file combination has stronger grid checks.

A requested tone above Nyquist is silently replaced by the nearest available
positive bin: a 20000 Hz request at 32000 Hz sampling returns 16000 Hz.
Similarly, a 40–10000 Hz OASPL request with 8000 Hz sampling is accepted although
only frequencies through 4000 Hz exist. No warning identifies the incomplete
bandwidth. Very small configured `NPERSEG` values are not explicitly validated.

The inspected real input files have finite selected pressures, increasing
timestamps and Nyquist frequencies above the requested integration interval.
Their timestamp deviations are small: approximately 0.009–0.025% of the mean
step, consistent with CSV precision. No major sampling-grid error was found
in these files.

**Mitigation:** validate samples and references once at loading; distinguish
roundoff from real irregular sampling; reject or explicitly resample missing
intervals. Require valid tone/band support and report every incomplete band.

### Additional confirmed issue: saved unweighted measurement overlays use A-weighted test data

Code: `regenerate_aeroacoustic_plots.py:refresh_spectral_displays`, lines 100–105;
the called renderer is `plot_aeroacoustics.py:render_third_octave_plots`, line 1237.

The refresh helper passes the workbook's A-weighted measurement spectra to the
renderer for both weighting modes. The normal plotting CLI passes measurements
only in A-weighted mode, but this helper bypasses that safeguard. The renderer
does not carry or validate weighting metadata in `MeasurementSpectrum`.

This is present in a saved result, not only a possible future error. In
`POC2/aeroacoustic_plots/step4_measurements_comparison/unweighted/third_octave_spectrum_observer_3.png`,
the black measurement point near nominal 63 Hz is about 36.30, matching the
workbook's **36.300887 dBA**, whereas the blue CFD curve is unweighted and the
axis is labelled dB. This alone introduces a large frequency-dependent comparison
error: A-weighting near 63 Hz is approximately −26.2 dB.

The same refresh-helper path covers the saved Lifter4B measurement-overlay
directory. A-weighted overlays and simulation OASPL calculations are not affected
by this particular plotting error.

**Mitigation:** suppress these measurement overlays for unweighted plots unless
true unweighted measurements are supplied. Add an explicit weighting field and
validation to measurement data and plotting. Do not claim that subtracting the
centre-frequency A correction exactly recovers an unweighted band from the
A-weighted band alone; its within-band spectral distribution is unknown.

## Important analysis limitations and interpretation inconsistencies

### 10. Nearest-bin tone levels suffer off-bin amplitude loss

Code: `tone_spl_db` at line 660 and BPF component plots at line 1522.

Taking one amplitude-normalized Hann FFT bin is exact only under suitable
stationary, bin-centred tone conditions. A unit-RMS tone halfway between bins
is reported 1.4236 dB low. With POC2 observer 3's actual record lengths/sample
rates and known BPF, a controlled unit-RMS BPF tone is 1.322 dB low for step3
and 1.333 dB low for step4. These are isolated numerical losses, not inferred
errors in the amplitudes of the actual mixed CFD signal.

Different observers can have different retained lengths and hence different
frequency grids and scalloping losses. Actual nonstationarity, broadband noise
or overlapping tones add further uncertainty.

**Mitigation:** estimate tones using a calibrated frequency-aware sinusoidal fit
or integrate a resolved, non-overlapping tone neighbourhood with correct power
normalization. Report the target frequency, actual bin frequency, spacing and
method. A whole third-octave band is not necessarily an isolated-tone estimator.

### 11. Harmonic third-octave neighbourhoods overlap and include other tones

Code: `harmonic_third_octave_spl_db` at line 670;
`render_harmonic_content_plots` at line 1601.

The windows centred at `h * BPF` are mathematically valid frequency-band sums,
but they are not isolated harmonic powers. Adjacent windows overlap starting
with harmonics 4 and 5. The tenth window extends from 8.909 to 11.225 BPF and
contains the ninth and eleventh harmonics as well as the tenth.

A synthetic signal containing only a 1 Pa RMS ninth harmonic gives **93.9794 dB**
in the plotted tenth-harmonic neighbourhood despite having no tenth-harmonic tone.
These neighbourhood levels must not be summed as independent harmonic powers.

**Mitigation:** label them as third-octave neighbourhood levels, or calculate
isolated harmonic powers in non-overlapping intervals or with a multi-tone fit.
Do not remove noise-floor contributions without a separate justified model.

### 12. Narrowband broadband levels depend on analysis bandwidth

Code: `NPERSEG="All"` at line 94;
`spl_spectrum_db` at line 645 and spectrum comparison at line 1194.

The current amplitude-normalized narrowband convention is useful for tones,
but its broadband floor is proportional to equivalent noise bandwidth:
`S = G * ENBW = G * 1.5 * df` for the periodic Hann window. Unequal record lengths
therefore introduce a broadband-level offset even for identical physical PSD.

The same stationary white-noise record, evaluated with 1024 and 8192 samples per
segment, gives broadband amplitude levels differing by 9.063 dB. The expected
bandwidth difference is 10*log10(8) = 9.031 dB. PSD levels differ by only 0.032 dB
and band-limited OASPL by 0.035 dB in this realization.

POC2 step4 observer 3 has df = 5.805 Hz; Pusher step3 observer 3 has df = 18.206 Hz.
Equal broadband PSDs would consequently appear about **4.96 dB apart** in the
current amplitude-spectrum display. That is a comparison convention effect,
not evidence of different source noise.

**Mitigation:** retain amplitude spectra for tone inspection, but provide PSD
plots with explicit dB re p0²/Hz for broadband comparisons. Alternatively impose
identical physical resolution bandwidth. Record df, ENBW and segment settings
with every comparison. The earlier 1.76 dB integration fix does not solve this.

### 13. Spectrogram settings cannot resolve the POC2 BPF

Code: `SPECTROGRAM_NPERSEG=512` at line 77;
`_spectrogram_spl_db` at line 1856;
`render_spectrogram_plots` at line 1878.

At POC2 step4 sampling, the window lasts 3.840 ms and df is 260.416 Hz. The BPF
is 66.667 Hz, below the first positive frequency bin. Per-segment mean removal
further suppresses such slowly varying content. A controlled unit-RMS POC2 BPF
tone loses 12.42 dB of mean full-frequency power with these short, detrended
segments. The spectrogram cannot be read as a resolved history of that tone.

`scaling="spectrum"` is an amplitude normalization here despite `mode="psd"`;
the displayed values are not density or additive bin power. Calling the colour
scale simply "SPL" leaves the bandwidth implicit. The time axis is divided by
the final window-centre time, not the full record duration, and hides seconds.

**Mitigation:** select spectrogram duration from the desired physical frequency
resolution and transient timescale, not a fixed number of samples for every
case. Label PSD or amplitude normalization and bandwidth. Use seconds when
comparing transients. Longer windows improve frequency resolution while reducing
time localization; these short records cannot provide both arbitrarily.

### 14. One full-record periodogram has limited statistical information

With `NPERSEG="All"`, 50% overlap does not create multiple segments. There is no
Welch ensemble averaging. POC2 step4 observer 3 spans about 2.87 revolutions,
Lifter4B step3 about 1.86, and Pusher step3 about 2.00. Lowest bands contain only
one or two bins, sometimes none. Hard assignment of all a bin's power according
to its centre is sensitive to leakage and band boundaries.

A controlled 71 Hz tone lies above the binary 62.5 Hz band's 70.154 Hz upper edge.
With df = 20 Hz, the algorithm nevertheless puts substantial power into that
band: 90.50 dB versus 93.98 dB for the whole tone. This demonstrates finite-record
leakage/resolution, not an error in the new PSD normalization.

The full-record Hann result is a window-weighted mean square, not guaranteed to
equal the unweighted record RMS for a transient. In the main real-data examples
the Hann-versus-rectangular 40–10000 Hz differences are generally modest; the
short standalone continuation differs more. There is no universal correction
from one windowed estimator to another.

**Mitigation:** collect longer stationary data, perform duration/segment/window
convergence checks, disclose band resolution and averaging, and separate early
transients deliberately. If an equal-time-weighted record RMS is wanted, compute
and label that quantity separately. More interpolation or FFT zero padding alone
does not create independent frequency information.

### 15. Measurement geometry and calibration transfer remain conditional

The plotting script does not store or validate observer radius, source position,
microphone coordinates, propagation environment, measurement normalization or
analyser bandwidth. Local CFD observer positions are around 100 m; the microphone
coordinate file suggests a radius around 15.45 in its coordinate units. The
notebook explicitly labels those as workbook units. The workbook's normalization
basis is not established by the acoustic plotting code.

If those microphone coordinates are metres and the supplied levels are raw at
that distance, simple far-field spherical spreading alone would imply roughly
16.2 dB between 15.45 and 100 m. That is **conditional**, not a demonstrated missing
correction: the workbook may already represent a normalized basis, and flow,
ground, directivity and atmospheric effects require separate treatment.

The related Lifter4B correction, `L_Lifter + L_measured_POC2 - L_CFD_POC2`, is
mathematically valid as a multiplicative transfer of band energies. It assumes
that the discrepancy transfers across geometry/operating-point differences.
It is an engineering calibration, not an independent validation. Band-definition
and coverage mismatches must be resolved before interpreting that transfer.

**Mitigation:** attach explicit geometry/measurement metadata and establish the
comparison basis before applying any distance or propagation adjustment.

## Checks that did not identify a new error

- `PSD * df` gives the correct additive one-sided bin powers. No extra factor of
  two is needed: SciPy already folds negative-frequency power into positive bins.
- A-weighting is correctly applied to *power* using `10**(A/10)` before summation.
  Its amplitude response uses `20*log10`, and normalization is 0 dB at 1 kHz.
- Pressure dimensionalization uses rho*a², appropriate for these air cases using
  `AerospaceCondition`. The configured force-coefficient reference velocity is
  correctly excluded from acoustic pressure scaling. Time uses L_grid/a. [1]
- BPF = blade count * RPM / 60 is correct, and these setups have four blades.
- Total pressure equals thickness plus loading exactly for the inspected selected
  observer arrays. Coherent surface-pressure sums preserve interference; adding
  independently calculated surface SPLs would not preserve it.
- The actual Pusher step3B signals start one acoustic sample after step3 ends for
  observers 1–5. Their active windows do not overlap. The current timestamp-aligned
  pressure sum therefore acts as a continuation without double-counting these
  signals. The 150-CFD-step cut corresponds to approximately 75 acoustic samples.
  Other files must still be checked for duplicate/cumulative content before summing.
- Revolution averaging in the current rendering function uses samples per
  revolution inferred from dimensional sampling and RPM before dividing into
  complete revolutions. No major period-length error was found for these files.

Nyquist support is a sampling requirement, not proof of CFD temporal accuracy.
For Pusher step3, approximately 6.40 solver steps and 3.20 acoustic samples occur
per 10 kHz period. Postprocessing cannot recover any physical attenuation or
dispersion caused by inadequate CFD time or spatial resolution.

## Recommended correction order

1. Define common, explicit physical metadata and observer selection for every
   study/file; resolve sources independently before combining them. Remove the
   incompatible A-weighted measurement overlays from unweighted plots.
2. Unify full third-octave band definitions with the measurement convention and
   compute a separate measurement-matched overall level. Handle unresolved bands.
3. Unify time selection across single/combined and surface/total paths; add input
   and frequency-support validation.
4. Improve tone extraction and separate isolated harmonics from overlapping
   third-octave neighbourhoods.
5. Use bandwidth-independent PSD comparisons and physically chosen spectrogram
   settings; document statistical and CFD resolution limits.
6. Resolve measurement geometry/propagation basis before interpreting calibration
   as a prediction at a specified distance.

No blanket level adjustment should be applied for these findings. They affect
different outputs and conditions, sometimes with opposite signs.

## Reproduction

From the repository root, using the configured project interpreter:

```powershell
& 'C:\git\flow360cases\venv\Scripts\python.exe' `
  'DUC\results_postprocessing\aeroacoustic_math_audit\verify_math.py'

& 'C:\git\flow360cases\venv\Scripts\python.exe' -m unittest discover `
  -s tests -p 'test_plot_aeroacoustics*.py'
```

The audit script writes only `audit_results.json` beside itself. The measurement
workbook path is taken from the existing calibration script. Versions used:
Python 3.11.9, NumPy 2.4.4, SciPy 1.16.3. It verifies that the production plotting
script hash is unchanged during execution. Code changes are required before
regenerating outputs for the identified defects.

## Primary references

[1] [Flow360 reference scaling](https://docs.flexcompute.com/projects/flow360/en/stable/user_guide/NonDimensionalization/Introduction/Introduction.html)
and [acoustic output/padding documentation](https://docs.flexcompute.com/projects/flow360/en/latest/user_guide/NonDimensionalization/NonDimOutputs/NonDimOutputs.html).

[2] [SciPy Welch documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
and [SciPy spectrogram documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html).

[3] [IEC 61260-1:2014 preview, clauses 5.2–5.4](https://preview.sist.si/sist-preview/13383/3c4ae3e762b540cc8111744cb8f0ae8e/IEC-61260-1-2014.pdf).

These references support definitions and normalization conventions; the numerical
findings above were calculated independently from the local code and data.
