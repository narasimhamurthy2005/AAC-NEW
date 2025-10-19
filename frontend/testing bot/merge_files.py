import os

folder = "scraped_pages"
output_file = "combined_data.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as infile:
                outfile.write(f"\n--- {file} ---\n")
                outfile.write(infile.read() + "\n")

print(f"✅ Combined data saved as {output_file}")
