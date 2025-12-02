import pandas as pd
import os

# === Input file ===
dir = "C:/Nextcloud/Freigaben/GBT/derivatives_RevB (corrected)"

os.chdir(dir)

files = ["GBT_parametric_study.csv", "GBT_y_rotation_1.0rad_s.csv"]

for filename in files:

    # === Read CSV ===
    df = pd.read_csv(filename)
    df_copy = df.copy()

    # === Multiply the last 8 columns by 2 ===
    df_copy.iloc[:, -8:] = df_copy.iloc[:, -8:] * 2

    df_copy.loc[:, ["CMx", "CMz"]] = 0.0  # Set the entire 'CMx' and 'CMz' rows to 0.0

    # === Save new file with "_RevB" suffix ===
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_RevB{ext}"
    df_copy.to_csv(new_filename, index=False)

    print(f"Saved updated file: {new_filename}")