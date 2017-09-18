# Artificial-Music-Generation
My senior project.

Uses LSTMs to generate classical music. Training data is created from MIDI inputs. MIDI files are not part of this repository. 

Generation uses 2 LSTM networks to generate music. 
The first network is trained on only melodies, and generates a melody based on a seed suplied.
The second network is trained on the songs that contain a melody AND accompaniment. The data is formated so that the training and generation songs contain a melody and accompaniment up to the next note to predict. Then the last input before generating is the melody ONLY for the moment that an accompaniment is to be created.

The project worked, and led to my graduation. 

It is currently being reworked to be more powerful and versital, as well as removing unneeded steps. 
