# IrishPoetry-Gen
This project implements a text generation model using LSTM (Long Short-Term Memory) networks in PyTorch. The model is trained on a corpus of Irish folk songs and poems, generating text based on a given seed phrase.



# Dataset prep
Data Collection: The dataset consists of lyrics from traditional Irish folk songs and poems. The text is processed to create a corpus suitable for training the model.
Tokenization: The text is converted to lowercase, and unique tokens (words) are identified. An unknown token (<unk>) is used for words not found in the vocabulary.
Padding: Sequences are padded to ensure uniform length, allowing for batch processing during training.

# Model
layers: An embedding layer to transform word indices into dense vectors.
        A bidirectional LSTM layer to capture context from both past and future words.
        A second LSTM layer for further processing the output from the first layer.
        A fully connected layer to project the LSTM outputs to the vocabulary size for word predictions.
Loss Function: The model uses Cross-Entropy Loss to evaluate the difference between the predicted word probabilities and the actual target words.
Optimizer: The Adam optimizer is used for updating the model parameters, which adapts the learning rate based on the first and second moments of the gradients.

# Eval
After training, the model can generate text based on a seed phrase.
Input Preparation: The seed text is tokenized and padded to match the input size expected by the model.
Prediction: The model outputs a probability distribution over the vocabulary for the next word. The word with the highest probability is selected.
Iterative Generation: This process repeats for a specified number of next words, appending each predicted word to the seed text to form a coherent sequence.
