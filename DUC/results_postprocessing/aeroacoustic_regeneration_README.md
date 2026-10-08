# Corrected acoustic plot regeneration

The integrated CFD levels use Welch PSD times frequency-bin width. The original
Hann spectrum-summation correction is retained. Additional audit fixes use full
IEC base-10 bands, measurement-matched OASPL, common observer/component time
windows, joint harmonic tone fits, PSD comparisons and density spectrograms versus
seconds. Narrowband amplitude spectra remain separate. Details and CLI examples
are in [aeroacoustic_fixes.md](aeroacoustic_fixes.md).

Regenerated studies: POC2 step3/step4 comparison and step4 measurement overlays;
Lifter4B step4, step3, and POC2 measurement overlays; Pusher step3 and the
step3+step3B continuation with the first 150 CFD steps (75 acoustic samples)
excluded; and the POC2/Lifter4B/Q40LT comparison. Weighted and unweighted plots
are in their corresponding subfolders. Existing flat-layout copies are refreshed.
Surface plots are generated where surface-resolved CSVs exist.

The POC2-calibrated Lifter4B plots are in Lifter4B_corrected. The common CFD
normalization correction cancels in Lifter CFD + measurement - POC2 CFD,
so that normalization fix alone leaves the transferred estimate unchanged.
The subsequent complete IEC-band and selection fixes can change it by different
amounts in different bands. Measurement values are unchanged. IEC base-10 is an
assumed analyser convention; measurement distance and propagation normalization
remain unverified. See Lifter4B_corrected/README.md.

The workbook's A-weighted measurements appear only on A-weighted spectral plots.
Measured OASPL is compared with the sum of the same full measurement bands;
those figures and CSV exports contain `measurement_matched` in their names.
Literal 40–10000 Hz OASPL remains a separate reported quantity. Missing FFT bands
are gaps rather than fabricated levels. New `psd_spectrum*.png` plots use density
units for broadband comparison. Each `analysis_manifest.json` reports df, ENBW,
segment and band bin counts, and both CFD and acoustic sample counts per period.

Commands: aeroacoustic_regeneration_commands.txt
Inputs, normalization and numerical calibration verification:
aeroacoustic_regeneration_summary.json
Per-study output logs: regeneration_a_weighted.log / regeneration_unweighted.log
Pusher continuation numerical comparison: Pusher/aeroacoustic_plots/step3B/comparison_with_step3.csv

To regenerate the suite, run the configured project Python interpreter on
regenerate_aeroacoustic_plots.py from this directory or the repository root.
The optional Q40LT comparison requires its recorded F: input path.

These fixes do not establish solver, acoustic-sampling or record-length
convergence. Longer stationary records and convergence checks remain necessary.
The earlier PDF reports and source snapshots describe previous versions and are
retained as historical documents. The
[mathematical audit](aeroacoustic_math_audit/mathematical_audit.md) now describes the
corrected implementation; its original version is archived separately.
