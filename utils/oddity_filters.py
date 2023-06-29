# We use the same DataFrame from the previous steps: final_df

# 1. Filter rows where 'text' has only one word
single_word_df = final_df[final_df['text'].str.split().str.len() == 1]

# 2. Case insensitive filter for 'text' that contains the word 'nei'
nei_in_text_df = single_word_df[single_word_df['text'].str.contains(r'\bnei\b', case=False, na=False, regex=True)]

# 3. Case insensitive filter for 'wav2vec' that does NOT contain the word 'nei' or 'tnei'
nei_df = nei_in_text_df[~nei_in_text_df['wav2vec'].str.contains(r'\b(nei|tnei)\b', case=False, na=False, regex=True)]

# Loop over the DataFrame and print the desired fields
for idx, row in nei_df.iterrows():
    print(f"ID: {row['id']}\nText: {row['text']}\nWav2Vec: {row['wav2vec']}\n")

# Print number of matches
print(f'Number of matches: {len(nei_df)}')

# Save filtered DataFrame to a tab-separated file
# nei_df.to_csv('nei.tsv', sep='\t', index=False)


# We use the same DataFrame from the previous steps: final_df

# 1. Filter rows where 'text' has more than one word
single_word_df = final_df[final_df['text'].str.split().str.len() == 1]

# 2. Case insensitive filter for 'text' that contains the substring 'nei'
nei_in_text_df = single_word_df[single_word_df['text'].str.contains('nei', case=False)]

# 3. Case insensitive filter for 'wav2vec' that contains the substring 'nei'
nei_df = nei_in_text_df[nei_in_text_df['wav2vec'].str.contains('nei', case=False)]

# Print number of matches
print(f'Number of matches: {len(nei_df)}')

# Loop over the DataFrame and print the desired fields
for idx, row in nei_df.iterrows():
    print(f"ID: {row['id']}\nText: {row['text']}\nWav2Vec: {row['wav2vec']}\n")

# Save filtered DataFrame to a tab-separated file
# nei_df.to_csv('nei.tsv', sep='\t', index=False)



import re

# Define the pattern for 'ja' or 'tja' as standalone words
pattern_text = re.compile(r'\b(ja|tja)\b', re.IGNORECASE)

# For wav2vec we just look for 'ja' as a substring (not standalone word)
pattern_wav2vec = re.compile('ja', re.IGNORECASE)

# Apply the pattern over each element of 'text' and 'wav2vec' columns
text_contains = final_df['text'].apply(lambda x: bool(pattern_text.search(x)))
wav2vec_not_contains = final_df['wav2vec'].apply(lambda x: not bool(pattern_wav2vec.search(x)))

# Add condition for 'text' to be a single word
text_single_word = final_df['text'].apply(lambda x: len(x.split()) == 1)

# Combine all conditions using the bitwise operator & (and)
filtered_df = final_df[text_contains & wav2vec_not_contains & text_single_word]

# Loop over the filtered DataFrame, print the entries and count them
count = 0
for index, row in filtered_df.iterrows():
    print(f"ID: {row['id']}\nText: {row['text']}\nWav2Vec: {row['wav2vec']}\n---\n")
    count += 1

# Print the count after going through the entire DataFrame
print(f"Number of rows where 'text' contains 'ja' or 'tja' (standalone words), 'wav2vec' does not contain 'ja' as a substring, and 'text' is a single word: {count}")

# Save filtered DataFrame to a tab-separated file
filtered_df.to_csv('ja.tsv', sep='\t', index=False)



# Initialize a counter
count = 0

# Initialize a list to store rows that match the conditions
matching_rows = []

# Loop through filtered DataFrame
for idx, row in filtered_df.iterrows():
    # Split phrase into words
    words = 'takk for at du så på'.split()
    
    # Check if at least 4 words from the phrase occur in 'wav2vec'
    wav2vec_words = row['wav2vec'].lower().split()
    common_words = [word for word in words if word in wav2vec_words]

    if len(common_words) < 4:
        matching_rows.append(row)
        count += 1

print(f"Number of hits: {count}")

# Convert matching_rows to a DataFrame and save it to a tab-separated file
matching_df = pd.DataFrame(matching_rows)
matching_df.to_csv('takk_for_at_du_saa_paa.tsv', sep='\t', index=False)

