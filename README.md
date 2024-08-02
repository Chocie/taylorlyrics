# Taylor Lyrics Generator

The "Lyrics Generation using RNN" project aims to create an AI-powered system that generates song lyrics similar to Taylor Swift's style. By utilizing a pre-trained RNN model, the project generates lyrics that mimic the patterns and themes found in the input training lyrics.

TaylorLyrics is a Python package for processing Taylor Swift lyrics and training RNN models to analyze them.

## Installation

```bash
pip install git+https://github.com/yourusername/taylorlyrics.git

## Example Usage
from taylorlyrics import LyricProcessor, RNNProcessor

processor = LyricProcessor('path_to_lyrics_file.csv')
X_train, y_train, X_valid, y_valid = processor.get_train_valid_set(sequence_length=7, valid_percent=0.1, sequence_type='padded_sequences')

rnn = RNNProcessor()
model = rnn.create_and_train_rnn_model(X_train, y_train, X_valid, y_valid, batch_size=512, epochs=200, neurons=256, lstm_layers=1, learning_rate=0.001, dropout=0.2)

generator = LyricGenerator(processor, model, lyric_length=140)
generator.generate_sample_lyrics(10)
