import re
from fuzzywuzzy import fuzz
import json


def find_similarity_location(text, query, threshold=50):
    best_match = None
    best_similarity = 0
    best_start = None
    best_end = None

    # Convert text and query to Unicode if not already
    text = text.strip()
    query = query.strip()

    # Check for exact match first
    start_idx = text.find(query)
    if start_idx != -1:
        end_idx = start_idx + len(query)
        return query, 100, start_idx, end_idx

    # Use fuzzy matching to find approximate matches
    # Split text by non-word characters while preserving Unicode
    words = re.findall(r'\S+', text)  # Find sequences of non-whitespace characters
    for word in words:
        similarity = fuzz.ratio(word, query)
        if similarity >= threshold and similarity > best_similarity:
            best_match = word
            best_similarity = similarity
            best_start = text.find(best_match)
            best_end = best_start + len(best_match)

    if best_match:
        return best_match, best_similarity, best_start, best_end
    else:
        return None, 0, None, None


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # Specify encoding
        data = json.load(file)
    return data


def extract_result(data, text_output):
  results = {}
  for key, value in data.items():
      match_text, similarity, start, end = find_similarity_location(text_output, value)
      if match_text:
          results[key] = {
              "value": value,
              "match": match_text,
              "similarity": f"{similarity}%",
              "start": start,
              "end": end
          }
      else:
          results[key] = {
              "value": value,
              "match": None,
              "similarity": "0%",
              "start": None,
              "end": None
          }

  return results