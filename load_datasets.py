import random
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
# PUT POLARITY DATASET PATH HERE
POLARITY_PATH = os.environ.get(
    'POLARITY_PATH',
    os.path.join(os.path.dirname(__file__), '/sorted_data')
)
def LoadDataset(dataset_name):
  if dataset_name.endswith('ng'):
    if dataset_name == '2ng':
      cats = ['alt.atheism', 'soc.religion.christian']
      class_names = ['Atheism', 'Christianity']
    if dataset_name == 'talkng':
      cats = ['talk.politics.guns', 'talk.politics.misc']
      class_names = ['Guns', 'PoliticalMisc']
    if dataset_name == '3ng':
      cats = ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.windows.x']
      class_names = ['windows.misc', 'ibm.hardware', 'windows.x']
    newsgroups_train = fetch_20newsgroups(subset='train',categories=cats)
    newsgroups_test = fetch_20newsgroups(subset='test',categories=cats)
    train_data = newsgroups_train.data
    train_labels = newsgroups_train.target
    test_data = newsgroups_test.data
    test_labels = newsgroups_test.target
    return train_data, train_labels, test_data, test_labels, class_names
  if dataset_name.startswith('multi_polarity_'):
    name = dataset_name.split('_')[2]
    return LoadMultiDomainDataset(os.path.join(POLARITY_PATH, name))
def LoadMultiDomainDataset(path_data, remove_bigrams=True):
  random.seed(1)
  pos = []
  neg = []
  def get_words(line, remove_bigrams=True):
    features = re.findall(r'(\w+):(\d+)', line)
    if remove_bigrams:
      features = [(word, count) for word, count in features if '_' not in word]
    features = [(word, count) for word, count in features if word.isalpha()]
    return ' '.join(word for word, count in features for _ in range(int(count)))
  with open(os.path.join(path_data, 'processed.review'), encoding='latin-1') as f:
    for line_number, line in enumerate(f, start=1):
      features, separator, label = line.partition('#label#:')
      if not separator:
        raise ValueError('Missing #label#: marker on line {} of processed.review'.format(line_number))
      words = get_words(features, remove_bigrams)
      label = label.strip()
      if label == 'positive':
        pos.append(words)
      elif label == 'negative':
        neg.append(words)
      else:
        raise ValueError('Unexpected label {!r} on line {} of processed.review'.format(label, line_number))
  random.shuffle(pos)
  random.shuffle(neg)
  split_pos = int(len(pos) * .8)
  split_neg = int(len(neg) * .8)
  train_data = pos[:split_pos] + neg[:split_neg]
  test_data = pos[split_pos:] + neg[split_neg:]
  train_labels = [1] * len(pos[:split_pos]) + [0] * len(neg[:split_neg])
  test_labels = [1] * len(pos[split_pos:]) + [0] * len(neg[split_neg:])
  return train_data, np.array(train_labels), test_data, np.array(test_labels), ['neg', 'pos']
