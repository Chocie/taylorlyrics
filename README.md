# Taylor Lyrics Generator

The "Lyrics Generation using RNN" project aims to create an AI-powered system that generates song lyrics similar to Taylor Swift's style. By utilizing a pre-trained RNN model, the project generates lyrics that mimic the patterns and themes found in the input training lyrics.

## Install requires

pandas, numpy，nltk，keras，tensorflow，matplotlib，textblob

## Installation

```bash
pip install git+https://github.com/Chocie/taylorlyrics.git
```

## Example Usage

```bash
from taylorlyrics import LyricProcessor, RNNProcessor

# Process Data 
processor = LyricProcessor('path_to_lyrics_file.csv')
X_train, y_train, X_valid, y_valid = processor.get_train_valid_set(sequence_length=7, valid_percent=0.1, sequence_type='padded_sequences')

# Run Model
rnn = RNNProcessor()
model = rnn.create_and_train_rnn_model(X_train, y_train, X_valid, y_valid, batch_size=512, epochs=200, neurons=256, lstm_layers=1, learning_rate=0.001, dropout=0.2)

# Run Generator
generator = LyricGenerator(processor, model, lyric_length=140)
generator.generate_sample_lyrics(10)
```

## Example Result
![image](https://github.com/user-attachments/assets/f2e8e2ef-ebd4-4e26-822e-56a76531cdb2)

i took a chance i do of think of well thing all crowd in wishing for the red there you eyes i alone baby about when its i never and like and

this aint for the best york reputations of gon im meet im like im winestained plans i your little you might you blame out im and are i i love

hey hey hey hey hey and might there red you beautiful if see you could its do i you want are like you look wan i love were the i snow i here

i can see you standing just the all party it need have so do i i you wan my said your drive out friends who it i this i and hope i you i pull

we were in the backseat on the the i of just is line but a say and what all and you you i my might can i i a i youre how i players but a just

we were both young when but first saw you not be daylight them it can and you that like your know snow to was now if know like no know the acted

what did you think id na hear you you i never a now your marvelous years it a drives thinking but you mothers on her me run and reputation you

i see your face in i know as wan all i and you head wan me dont in keeps that im caught you life is no i the i like that the wan was is love

i took a chance i my sky you sure if a so i throw my when youre now also saw i but no do i na i crawling aside up a like your in they things

i never trust a narcissist as they i sent im the the rather tricks a story it man never call and say yeah forever you you know afterglow with


