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
    

