# POC2 pressure-history processing report

Second technical report: **Acoustic signal processing — POC2 worked example**.
The PDF uses the Achleitner Aerospace report template and real POC2 step4 data.

The analysis follows dimensional pressure/time conversion, clipping of padded
zeros, mean removal, a periodic Hann window, the FFT, the amplitude-normalized
narrowband spectrum, PSD integration for third-octave bands, and band-limited
unweighted and A-weighted OASPL. It reproduces the former erroneous summation
and the corrected calculation using matching spectral settings. The correction
is checked independently against Hann-weighted time-domain power.

## Input and analysis settings

- Case: `fb40c043-c99a-4cce-95cf-de9630dba98d`, POC2 step4.
- Complete original CSV: `input/POC2/aeroacoustics/step4/case-fb40c043-c99a-4cce-95cf-de9630dba98d_total_acoustics_v3.csv`.
- CSV SHA-256: `4162c972e135dd09d2d4783618e8da99d0daf25f9fb144bda6bdfad4ad124dc4`.
- Detailed example: plot observer 3 / CSV `observer_2_pressure`, 105 degrees.
- All five observers are included in the OASPL comparison.
- 1000 RPM, four blades; density 1.064099 kg/m3, temperature 303.275 K,
  grid length unit 0.001 m; pressure reference is density times sound speed squared.
- Full retained record, one periodic Hann-windowed segment, no skipped samples
  or FFT zero padding. Integration interval: 40–10000 Hz.
- Exact numerical results, reference quantities, hashes and software versions:
  `analysis_results.json`.

## Rebuild

Extract the complete archive and run commands from the extracted report folder.
Dependencies used: Python 3.11.9, NumPy 2.4.4, SciPy 1.16.3 and Matplotlib 3.10.7.
XeLaTeX is needed to compile the report; the template's OpenSans fonts and artwork
are included. The font license is retained in `fonts/`.

In the original project, the PowerShell commands are:

```powershell
& 'C:\git\flow360cases\venv\Scripts\python.exe' '.\build_figures.py'
xelatex -interaction=nonstopmode -halt-on-error poc2_signal_processing.tex
xelatex -interaction=nonstopmode -halt-on-error poc2_signal_processing.tex
```

For another machine, replace the Python executable with an interpreter containing
the dependencies above. The builder uses the preserved plotting-script snapshot.
It uses the original project CSV if available, otherwise the included CSV. Both
copies have the hash recorded above. No production source is modified.

## Included artifacts

- `poc2_signal_processing.pdf`: finished report.
- `poc2_signal_processing.tex`, `numbers.tex`, `observer_table.tex`, template,
  fonts and assets: editable report sources.
- `build_figures.py`: reproducible calculations and five scientific figures.
- `figures/`: vector PDFs and PNG previews of every report figure.
- `observer_oaspl.csv`: original and corrected overall levels for all observers.
- `third_octave_observer3.csv`: original and corrected band levels and powers.
- `poc2_observer3_processing.npz`: actual retained time and dimensional pressure,
  demeaned pressure, Hann weights, frequency bins, PSD, amplitude spectrum and
  additive bin powers. Spectral arrays are linear, not decibels.
- `source_snapshot/plot_aeroacoustics.py`: corrected script used for reproduction.
- `source_snapshot/spl_spectra_calculator.py`: historical script containing the
  former Hann-spectrum summation error.
- `input/`: complete source CSV.

The archive excludes temporary LaTeX files and PDF layout-check images.
