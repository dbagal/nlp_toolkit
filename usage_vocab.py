
from vocab import Vocab
import re, os

def tokenizer(text):
    word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?])|(\.) |(\.$)|([a-z]'[a-z])| ''')
    tokens = [token for token in word_pattern.split(text) if token]
    return tokens

ROOT = os.path.dirname(os.path.realpath(__file__))
FOLDER_PATH = os.path.join(ROOT, "vocab_docs")
FILE_PATH = os.path.join(ROOT, "vocab_docs/42082.txt")
SAVE_PATH_1 = os.path.join(ROOT, "vocab1.json")
SAVE_PATH_2 = os.path.join(ROOT, "vocab2.json")
VOCAB_SIZE_1 = 200
VOCAB_SIZE_2 = 100


## Building vocabulary from a set of .txt files
vocab = Vocab(vocab_size=VOCAB_SIZE_1, tokenizer=tokenizer)
vocab.buildFromFolder(FOLDER_PATH)
vocab.save(SAVE_PATH_1)


## Building vocabulary from a single .txt file
vocab = Vocab(vocab_size=VOCAB_SIZE_2, tokenizer=tokenizer)
vocab.buildFromFile(FILE_PATH)
vocab.save(SAVE_PATH_2)

## Loading the vocabulary
vocab  = Vocab()
vocab.load(SAVE_PATH_2)
print(f"Number of words in the vocabulary: {vocab.size}")

""" 
Output:
Number of words in the vocabulary: 100 
"""


