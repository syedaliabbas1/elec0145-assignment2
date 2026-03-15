import os

# Define all class names — edit these if you change your classes
classes = [
    "01_BMW_M3_2023",
    "02_Porsche_911_2024",
    "03_Mercedes_AMG_C63_2023",
    "04_Audi_RS6_2023",
    "05_Ferrari_SF90_2022",
    "06_Mercedes_Maybach_S580_2022",
    "07_Lamborghini_Urus_2022",
    "08_AlfaRomeo_Giulia_QV_2022",
    "09_RangeRover_Sport_2023",
    "10_Tesla_ModelS_Plaid_2022"
]

# Define full directory structure
splits = ["raw", "train", "val", "test"]

dirs_to_create = [
    "results",
    "models",
]

# Create top-level dirs
for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}/")

# Create dataset split folders with class subfolders
for split in splits:
    for cls in classes:
        path = os.path.join("dataset", split, cls)
        os.makedirs(path, exist_ok=True)
    print(f"Created: dataset/{split}/ with {len(classes)} class folders")

print("\nProject structure ready.")