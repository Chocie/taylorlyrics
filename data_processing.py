import pandas as pd
import numpy as np
import string
import re
import collections

from collections import Counter
from numpy import array
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical

import nltkrt st
from nltk.corpus impoopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

from enum import Enum

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))


class SequenceType(Enum):
    PADDED_SEQUENCES = 'padded_sequences'


class LyricProcessor:

    def __init__(self, file_path, rows=2000):
        # rows: number of rows processed
        self.rows = rows
        self.file_path = file_path
        self.lyrics = self.__clean_lyrics(file_path, rows)

        # self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.label_dict, self.reverse_dict = self.__build_vocab_by_word()
        self.vocab_size = len(self.label_dict)

    def load_data(self):
        try:
            # Load the data from CSV file (assuming the CSV has a 'Lyrics' column)
            data = pd.read_csv(self.file_path)
            return data

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.file_path}' not found.")

        except Exception as e:
            raise Exception("Error occurred while loading data:", str(e))


    # Assemble all words back into a single word, and replace special chars
    def clean_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions (@usernames)
        text = re.sub(r'@\w+', '', text, flags=re.MULTILINE)
        # Remove hashtags (#hashtags)
        text = re.sub(r'#\w+', '', text, flags=re.MULTILINE)
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize text into words
        words = word_tokenize(text)
        # Remove stopwords and perform lemmatization
        # words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        # Reconstruct the cleaned text
        cleaned_text = ' '.join(words)
        return cleaned_text

    def __clean_lyrics(self, file_path, rows):
        try:
            # Read the CSV file using pandas
            df = pd.read_csv(file_path)

            col_name = 'Lyrics'
            if col_name not in df.columns:
                raise ValueError(f"'{col_name}' column not found in the CSV file.")
            df[col_name] = df[col_name].astype(str)

            # Drop duplicate rows
            df.drop_duplicates(subset=[col_name], inplace=True)
            # Drop rows with empty lyrics text
            df.dropna(subset=[col_name], inplace=True)

            # Clean lyrics text
            df['cleaned_lyrics'] = df[col_name].apply(self.clean_text)
            lines = df['cleaned_lyrics'].tolist()

            # Limit the number of rows
            lines = lines[:rows]

            print(f'...Loaded {len(lines)} lyrics from {self.file_path}.')
            return lines

        except Exception as e:
            print("Error occurred while cleaning data:", str(e))
            return None

    def __build_vocab_by_word(self):
        words = [word for line in self.lyrics for word in line.split()]
        n_words = len(set(words))

        count = [['UNK', 0]]  # Add 'UNK' as a placeholder for unknown words
        count.extend(collections.Counter(words).most_common(n_words))

        dictionary = {word: index for index, (word, _) in enumerate(count)}
        unk_count = sum(1 for word in words if word not in dictionary)
        count[0][1] = unk_count

        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        real_vocab_size = len(dictionary)

        print('...Loaded {0} total words and {1} unique words'.format(len(words), real_vocab_size))
        # print('...UNK represents {:.1%} of all words.'.format(unk_count / len(words)))
        print('...Most common words (+UNK)', count[:15])
        return dictionary, reversed_dictionary

        # Private method to shuffle two nparrays at the same time.

    def __shuffle(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

        # Public method to get a label by its index

    def get_label_by_index(self, index):
        return self.reverse_dict[index]

        # Public method to get an index by a label

    def get_index_by_label(self, label):
        try:
            return self.label_dict[label]
        except:
            return 0  # UNK

    def get_train_valid_set(self, sequence_length, sequence_type=SequenceType.PADDED_SEQUENCES, valid_percent=0.2):
        assert 0 <= valid_percent <= 1, "valid_percent must be between 0 and 1."

        X, y = self.__get_train_valid_sequences(sequence_length, sequence_type)

        valid_index = int(len(X) * (1 - valid_percent))
        X_train, y_train = X[:valid_index], y[:valid_index]
        X_valid, y_valid = X[valid_index:], y[valid_index:]

        # Reshape X_train, X_valid to 3D sequences
        X_train = self.__convert_to_3d_sequences(X_train)
        X_valid = self.__convert_to_3d_sequences(X_valid)

        return X_train, y_train, X_valid, y_valid

    def __convert_to_3d_sequences(self, sequences):
        num_samples = len(sequences)
        sequence_length = len(sequences[0])
        embedding_size = 1  # Change this if using word embeddings

        return sequences.reshape(num_samples, sequence_length, embedding_size)

    def __get_train_valid_sequences(self, sequence_length, sequence_type):
        X, y = self.__get_train_valid_set_padded_sequences(sequence_length)

        # Normalize and shuffle the sequences
        X_normalized = X.astype(float) / self.vocab_size
        y_normalized = to_categorical(y)

        X_shuffled, y_shuffled = self.__shuffle(X_normalized, y_normalized)
        return X_shuffled, y_shuffled

    def __get_train_valid_set_padded_sequences(self, sequence_length):
        X = []
        y = []

        for line in self.lyrics:
            x = np.zeros(sequence_length).tolist()
            words = line.split()

            for w in range(len(words)):
                for i in range(sequence_length):
                    row = []
                    row.append(words[w])

                    for j in range(sequence_length - 1):
                        if (j < i and (w + j + 1) < len(words)):
                            row.append(words[w + j + 1])
                        else:
                            row.insert(0, '')

                    if (w + i + 1 < len(words)):
                        X.append([self.get_index_by_label(word) for word in row])
                        y.append(self.get_index_by_label(words[w + i + 1]))

        return np.array(X), np.array(y)