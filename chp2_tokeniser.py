import re

class SimpleTokeniser:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for i,s in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[s] for s in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()

result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in result if item.strip()]
all_words = sorted(set(preprocessed))

vocab = {token:integer for integer, token in enumerate(all_words)}

tokenizer = SimpleTokeniser(vocab)
text = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride. "The last but one," she corrected herself--"but the other doesn't count, because he destroyed it."""

ids = tokenizer.encode(text)
print(ids)