import random

nepali_characters = [
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", 
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", 
    "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", 
    "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ", "ज्ञ"
]
nepali_symbols = [    "ं", "ँ", "्",
    "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ"]


def generate_nepali_word():
    word_length = random.randint(2, 4)
    word = ""
    last_was_symbol = False
    
    for _ in range(word_length):
        word += random.choice(nepali_characters)
        
        if not last_was_symbol and random.choice([True, False,  False,  False]):
            word += random.choice(nepali_symbols)
            last_was_symbol = True
        else:
            last_was_symbol = False

    return word

# Generate 5000 random Nepali words
nepali_words = [generate_nepali_word() for _ in range(5000)]

###################################################################################


# Define Nepali digits
nepali_digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']


# Define formats
formats = [
    "१५-०८-४५-४५६२३", "४५-४५६२३", "१५-०८-४५", "०८-४५",
    "साल: २०४५ महिना: ०२ गते: २८", "साल: २०४५", "२०४५ महिना", "महिना: ०२ गते: २८", "गते: २८", "महिना: ०२",
    "उ. म. न. पा.- १२", "गा. वि. स.- ९", "न. पा.- ५", "{anynepaliword} न. पा.- ५", 
    "{anynepaliword} गा. वि. स.- ९", "{anynepaliword} उ. म. न. पा.- १२", 
    "न. पा.- ५, {anynepaliword}", "५, {anynepaliword}", "गा. वि. स.- ९, {anynepaliword}",
    "स.- ९, {anynepaliword}", "९, {anynepaliword}", "वडा नं.: ९", "नं.: ९", "४५६२३", "६४३२", "१२३", "४५", "९"
]

# Function to replace Nepali numbers and words
def generate_random_text(format):
    text = format
    # Replace Nepali digits
    for digit in nepali_digits:
        text = text.replace(digit, random.choice(nepali_digits))
    
    # Replace any word placeholders with random Nepali words
    text = text.replace("{anynepaliword}", random.choice(nepali_words))
    
    return text



dataset = []
for _ in range(5000):
    format = random.choice(formats)
    dataset.append(generate_random_text(format))

# Save to .txt file
with open("dataset_test/nepali_digits.txt", "w", encoding="utf-8") as file:
    for line in dataset:
        file.write(line + "\n")

print("Dataset generated successfully!")

