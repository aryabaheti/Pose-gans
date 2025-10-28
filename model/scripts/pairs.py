import os
import csv

# Path to your folder containing heatmap images
folder_path = r"C:\Users\Arya Baheti\Desktop\test\data\keypoints\heatmaps"

# List all files
files = sorted(os.listdir(folder_path))

# Store matches in a dictionary
matches = {}

for file in files:
    if not file.endswith(".png"):  # skip non-image files
        continue

    # Extract ID part — first part before underscore
    file_id = file.split("_")[0]

    if file_id not in matches:
        matches[file_id] = []

    matches[file_id].append(file)

# Write matches to CSV
csv_path = r"C:\Users\Arya Baheti\Desktop\test\data\splits\train.csv"
with open(csv_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # Write header
    writer.writerow(["File_ID", "Files..."])

    # Write each group
    for file_id, file_list in matches.items():
        row = [file_id] + file_list  # first column = file_id, rest = filenames
        writer.writerow(row)

print(f"✅ CSV saved at: {csv_path}")
