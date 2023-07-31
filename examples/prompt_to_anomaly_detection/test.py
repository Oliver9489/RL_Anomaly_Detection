full_filepath = "./data/3_shot/ksdd2.tsv"
with open(full_filepath) as f:
    source_text = [line.strip() for line in f]

print(source_text)