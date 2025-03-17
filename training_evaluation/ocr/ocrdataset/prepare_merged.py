import random

# File paths
file1 = "dataset_test/nepali_digits.txt"
file2 = "dataset_test/nepali_words.txt"
output_file = "dataset_test/data.txt"

# Read lines from both files
with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
    lines = f1.readlines() + f2.readlines()

# Shuffle the lines randomly
random.shuffle(lines)

# Write shuffled lines to the new file
with open(output_file, "w", encoding="utf-8") as f_out:
    f_out.writelines(lines)
