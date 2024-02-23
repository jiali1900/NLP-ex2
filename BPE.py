import re
from collections import defaultdict, Counter

# read file
def preprocess_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()
train_file_path = '/Users/lijiali/Desktop/en_ewt-ud-train.txt'


# Step 1: Preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

preprocessed_text = preprocess_text_from_file(train_file_path)

# Step 2: Build initial vocabulary (word frequency dictionary)
def build_vocab(text):
    vocab = Counter(text.split())
    vocab = {' '.join(word) + ' </w>': freq for word, freq in vocab.items()}
    return vocab

vocab = build_vocab(preprocessed_text)

# Step 3: Define BPE merge operation
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = vocab[word]
    return v_out

# Step 4: Apply BPE operations
num_merges = 20  # For demonstration, perform 20 merges
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

# Display part of the transformed vocabulary
for word in list(vocab.keys())[:20]:
    print(word, "->", vocab[word])
