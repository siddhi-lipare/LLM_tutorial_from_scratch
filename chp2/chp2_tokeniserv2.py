import re
with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()

result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in result if item.strip()]
all_words = sorted(set(preprocessed))

vocab = {token:integer for integer, token in enumerate(all_words)}

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab= {token:integer for integer,token in enumerate(all_tokens)}
# print(len(vocab.items()))

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

class SimpleTokeniserV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text



tokenizer = SimpleTokeniserV2(vocab)
# text = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
text = "Hello, do you like tea?" # Hello isn't present in vocabulary, thus will be printed as special token <|unk|>

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

