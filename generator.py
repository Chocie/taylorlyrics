import numpy as np
import random
import re

from numpy import array
from keras.utils.np_utils import to_categorical
from textblob import TextBlob

class LyricGenerator:

  def __init__(self, lyric_processor, model, lyric_length=140):
    self.lyric_processor = lyric_processor
    self.model = model
    self.lyric_length = lyric_length

  def lyric(self, seed_text):
    seed_text = self.lyric_processor.clean_text(seed_text)  # Same cleanup on seed text as we did for training data
    full_lyric = seed_text

    # Convert the seed text into a list of integers (making sure to left pad if seed text is less than model input size)
    seed_words = seed_text.lower().split()

    if(len(seed_words) > self.model.input_shape[1]):
      seed_words = seed_words[:self.model.input_shape[1]]

    assert len(seed_words) <= self.model.input_shape[1], 'Seed text should have less than {0} words!'.format(self.model.input_shape[1])

    words = np.zeros(self.model.input_shape[1]).tolist()
    for i in range(len(seed_words)):
      words[self.model.input_shape[1] - len(seed_words) + i] = self.lyric_processor.get_index_by_label(seed_words[i])

    # Loop until we have a full lyric
    while len(full_lyric) < self.lyric_length:
      x = np.reshape(words, (1, len(words), 1))
      x = x / float(self.lyric_processor.vocab_size)
      predictions = model.predict(x, verbose=0)
      word_idx = self.__sample(predictions[-1], temperature=0.9)
      word = self.lyric_processor.get_label_by_index(word_idx)
      full_lyric = full_lyric + ' ' + word

      words.append(word_idx)
      words = words[1:len(words)]

    return full_lyric

  def generate_sample_lyrics(self, samples):
    seed_length = 5  # Length of the seed text for the sample
    seed_text = []   # Array of items to generate samples from
    seeds_found = 0

    print('')
    print('Showing {0} Sample Lyrics'.format(samples))
    print('')

    while seeds_found < samples:
      text = random.sample(self.lyric_processor.lyrics, 1)[0]
      words = text.split()
      if(len(words) >= seed_length):
        seed_text.append(' '.join([str(word) for word in words[:5]]))
        seeds_found = seeds_found + 1

    for i in range(len(seed_text)):
      sample = self.lyric(seed_text[i])
      print(sample)
      print('')

  def __sample(self, preds, temperature=1.0):
    if temperature <= 0:
      return np.argmax(preds)

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)