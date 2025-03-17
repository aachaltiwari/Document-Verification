import os
import random
from datasets import load_dataset


ds = load_dataset("ashokpoudel/nepali-english-translation-dataset")


with open('nepali.txt', 'w', encoding='utf-8') as f:
    for i in range(len(ds['train'])):
        f.write(ds['train'][i+50000+50000]['np'] + '\n')
        if i+1 == 50000+50000+15000:
            break


with open('nepali.txt', 'r', encoding='utf-8') as f:
    with open('words.txt', 'w', encoding='utf-8') as w:
        for line in f:
            words = line.split()
            for word in words:
                w.write(word + '\n')


with open("words.txt", "r", encoding="utf-8") as file:
    words = [line.strip() for line in file.readlines()]


with open("dataset_test/nepali_words.txt", "w", encoding="utf-8") as output_file:
    i = 0
    written_lines = 0
    while i < len(words) and written_lines < 15000:
        num_lines = random.choice([1, 1, 1, 2, 2, 3])
        if i + num_lines <= len(words):
            combined = " ".join(words[i:i + num_lines])
            output_file.write(combined + "\n")
            written_lines += 1
            i += num_lines
        else:
            break

# delete the nepali.txt and words.txt files
os.remove('nepali.txt')
os.remove('words.txt')