import re, os
from vocab import Vocab
from glove import GloVeDataset, GloVeModel
import torch.optim as optim
from torch.utils.data import DataLoader

def tokenizer(text):
    word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?])|(\.) |(\.$)|([a-z]'[a-z])| ''')
    tokens = [token for token in word_pattern.split(text) if token]
    return tokens

ROOT = os.path.dirname(os.path.realpath(__file__))
FOLDER_PATH = os.path.join(ROOT, "vocab_docs")
VOCAB_FILE_PATH = os.path.join(ROOT, "vocab.json")

VOCAB_SIZE = 500
EMBEDDING_DIM = 80

BATCH_SIZE = 4096
NUM_WORKERS = 4

## Building vocabulary from a set of .txt files
vocab = Vocab(vocab_size=VOCAB_SIZE, tokenizer=tokenizer)
#vocab.buildFromFolder(FOLDER_PATH)
#vocab.save(VOCAB_FILE_PATH)
vocab.load(VOCAB_FILE_PATH)

dataset = GloVeDataset(vocab, tokenizer=tokenizer)
dataset.generate(FOLDER_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

model = GloVeModel(vocab.size+2, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr = 0.005
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

## Train the GloVe model
model.train(
    train_loader=train_loader,
    optimizer=optimizer,
    num_epochs=10,
    save_dir=ROOT,
    device="cpu"
)

## Retrain the GloVe model
model.train(
    train_loader=train_loader,
    optimizer=optimizer,
    num_epochs=10,
    save_dir=ROOT,
    device="cpu"
)