from importlib.metadata import version
import tiktoken
# print(version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of some place."
) # BPE doesn't require <|unk|> token since it breaks down the words into sub-words and also includes individual characters as a part of its vocabulary. 

integers = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})

# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")
print("--------------------------------------------------")
for i in range(1, context_size+1):
    context = enc_sample[:i] 
    desired = enc_sample[i]
    print(f"{context} ------> {desired}") # Predicting one word at a time
    print(f"{tokenizer.decode(context)} ------> {tokenizer.decode(desired)}")

