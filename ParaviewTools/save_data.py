from paraview.simple import *
import os

# Find the original surfaces reader
surface_reader = FindSource("surfaces.vtu")

if surface_reader is None:
    raise RuntimeError("Could not find pipeline object 'surfaces.vtu'.")

# Get the filename of the loaded VTU
surface_file = surface_reader.FileName.GetData()[0]

# FileName can be either a string or a list depending on the reader
if isinstance(surface_file, (list, tuple)):
    surface_file = surface_file[0]

output_dir = os.path.dirname(surface_file)

print("Output directory:")
print(output_dir)

# -----------------------------
# Settings
# -----------------------------

objects_to_save = [
    "Extract wing1",
    "Extract wing2",
    "Extract wing3",
    "Extract wing4",
    "Extract stab1",
    "Extract stab2",
    "Extract stab3",
    "Extract stab4",
]

arrays_to_save = [
    "PatchID"
]

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Export loop
# -----------------------------
for obj_name in objects_to_save:
    source = FindSource(obj_name)

    if source is None:
        print(f"WARNING: Could not find pipeline object: {obj_name}")
        continue

    print(f"Saving {obj_name}...")

    patch_id_only = PassArrays(
        registrationName=f"{obj_name} PatchID only",
        Input=source,
    )
    patch_id_only.PointDataArrays = arrays_to_save
    patch_id_only.CellDataArrays = arrays_to_save

    # Make sure data is updated
    UpdatePipeline(proxy=patch_id_only)

    # Optional: safe filename

    filename = obj_name.replace("Extract ", "") + "_data.csv"

    filepath = os.path.join(output_dir, filename)

    SaveData(
        filepath,
        proxy=patch_id_only,
        Precision=6,
        UseScientificNotation=1,
        AddMetaData=0,
        ChooseArraysToWrite=1,
        PointDataArrays=arrays_to_save,
        CellDataArrays=arrays_to_save,
    )

print("Done.")
