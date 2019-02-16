import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, Vectors
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

class dataloader(object):
    def __init__(self, batch_size, max_vocab_size):
        self.lemma = WordNetLemmatizer()
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size

    def load_data(self):
        # remove HTML tags, punctuations -> lemmatize -> tokenize
        tokenize = lambda x: self.lemma.lemmatize(re.sub(r'<.*?>|[^\w\s]|\d+', '', x)).split()

        TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True,
                           include_lengths=True, batch_first=True, dtype=torch.long) #fix_length=200,
        LABEL = data.LabelField(batch_first=True, sequential=False)

        train, test = datasets.IMDB.splits(TEXT, LABEL)

        TEXT.build_vocab(train, max_size=self.max_vocab_size, vectors=GloVe(name='6B', dim=300)) # Glove Embedding
        LABEL.build_vocab(train)

        word_emb = TEXT.vocab.vectors
        vocab_size = len(TEXT.vocab)

        train, valid = train.split()
        train_data, valid_data, test_data = data.BucketIterator.splits((train, valid, test),
                                                                       batch_size=self.batch_size, repeat=False, shuffle=True)

        # print data info
        print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
        print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
        print ("Label Length: " + str(len(LABEL.vocab)))
        print ("\nSize of train set: {} \nSize of validation set: {} \nSize of test set: {}".format(len(train_data.dataset), len(valid_data.dataset), len(test_data.dataset)))
        print(LABEL.vocab.freqs.most_common(2))

        return TEXT, word_emb, train_data, valid_data, test_data, vocab_size