# import urllib.request
# url = ("https://raw.githubusercontent.com/rasbt/"
# "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
# "the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()
#     print("Total number of characters: ", len(raw_text))
#     print(raw_text[:99])

import re
# text = "Hello, World. This is a test."
# result = re.split(r'(\s)', text)
# result = re.split(r'([,.]|\s)', text)
# result = [item for item in result if item.strip()]
# text = "Hello, world. Is this-- a test?"


result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in result if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])
all_words = sorted(set(preprocessed))
print(len(all_words))

vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >=20:
        break

