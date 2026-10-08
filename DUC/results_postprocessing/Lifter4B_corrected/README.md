# Lifter4B POC2-calibrated acoustic estimate

Source measurement: `C:\OneDrive\OneDrive - Achleitner Aerospace GmbH\Achleitner Aerospace GmbH Allgemein - General\01_Projekte\27_DUC_Noise\POC_2x\SPL DUC_POC2_2X RPM_SWEEP_SBD_ASSESSMENT dBA.xlsx`; 1000 RPM, A-weighted third-octave bands,
50–10,000 Hz, observers 2–4 (90°, 105°, 135°). Source CFD: `C:\git\flow360cases\DUC\results_postprocessing\POC2\aeroacoustics\step4\case-fb40c043-c99a-4cce-95cf-de9630dba98d_total_acoustics_v3.csv` and
`C:\git\flow360cases\DUC\results_postprocessing\Lifter4B\aeroacoustics\step4\case-cf29e49e-da21-43ad-80e2-61a6f97bec80_results_total_acoustics_v3.csv`. The 24 bands use IEC base-10 exact third-octave edges; reported nominal
frequencies are retained in the tables. CFD pressures are dimensionalized and
Welch bin powers are computed as PSD times frequency-bin width, with window-energy
normalization, as in `plot_aeroacoustics.py`. OASPL is recomputed
from the 24 A-weighted band energies, not corrected as a single dB value.
The top band is centered at 10 kHz and includes its full upper half-band.
IEC base-10 bands are assumed for the measurement's nominal labels; the workbook
does not establish the analyser filter convention. FFT-bin integration is not
equivalent to a standards-certified filter bank, especially for short records.

The measured levels are used *as reported* by the workbook, matching the
existing `plot_aeroacoustics.py --measurement-excel ... --a-weighting` comparison.
No distance or atmospheric-absorption adjustment is applied. The separate
microphone-position file gives a radius around 15.45 in its coordinate units;
those units and the workbook's normalization basis have not been established.
The POC2 CFD setup specifies observers around 100 m; therefore this transfer
is conditional on the workbook already representing the intended comparison
basis. Its levels should not be interpreted as a validated prediction at a
specified distance without resolving that basis.

For every observer and third-octave band: `correction = POC2 measurement as
reported - POC2 CFD`; `Lifter4B estimate = Lifter4B CFD + correction`.
The same relation is equivalent to multiplying band mean-square pressures by
the POC2 measured/CFD power ratio. Tones and broadband were not separated,
because the measurement source provides only third-octave levels. POC2 CFD is
at 1000 RPM and Lifter4B CFD at 1063.68 RPM; the calibrated
estimate assumes the POC2 model discrepancy transfers across this operating
point difference. It is an engineering estimate, not an independent validation.

Run again with:

`python correct_lifter4b_acoustics.py --measurement-excel "C:\OneDrive\OneDrive - Achleitner Aerospace GmbH\Achleitner Aerospace GmbH Allgemein - General\01_Projekte\27_DUC_Noise\POC_2x\SPL DUC_POC2_2X RPM_SWEEP_SBD_ASSESSMENT dBA.xlsx"`
