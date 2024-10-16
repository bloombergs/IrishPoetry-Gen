import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

data = """'Twas down the glen one Easter morn
To a city fair rode I
Those armored lines of marching men
In squadrons passed me by
No pipe did hum nor battle drum
Did sound it's dread tattoo
But the Angelus bell o'er the liffey swell
Rang out of the foggy dew
Right proudly high over Dublin Town
Lay hung out the flag of war
'Twas better to die 'neath an Irish sky
Than at Suvla or Sud el Bar
And from the plains of Royal Meath
Strong men came hurrying through
And while Brittania's Huns with their long-range guns
Sailed out o'er the foggy dew
'Twas England bade our wild geese fly
That small nations might be free
Their lonely graves are by Sulva's waves
And the fringe of the great North Sea
Oh, had they died by Pearse's side
Or fought with Cathal Brugha
Their names we'd keep where fenians sleep
'Neath the shroud of the foggy dew
But the bravest fell as the requiem bell
Rang mournfully and clear
For those who died that Eastertide
In the spring time of the year
And the world did gaze with deep amaze
At those fearless men, but few
Who bore the fight so that freedom's light
Might shine through the foggy dew
Who bore the fight so that freedom's light
Might shine through the foggy dew
As I was a goin' over the far famed Kerry mountains
I met with captain Farrell and his money he was counting
I first produced me pistol and I then produced me rapier
Saying stand and deliver for I am the bold deceiver
Musha-ring dumma-do-damma-da
Whack for the daddy-o
Whack for the daddy-o
There's whiskey in the jar
I counted out his money, it made a pretty penny
I put it in my pocket and I took it home to Jenny
She sighed and she swore, she never would deceive me
But the devil take my women for they never can be easy
Musha-ring dumma-do-damma-da
Whack for the daddy-o
Whack for the daddy-o
There's whiskey in the jar
I went unto my chamber, all for to take a slumber
I dreamt of gold and jewels and for sure it was no wonder
But Jenny took my charges and she filled them up with water
And sent for captain Farrell to be ready for the slaughter
Musha-ring dumma-do-damma-da
Whack for the daddy-o
Whack for the daddy-o
There's whiskey in the jar
It was early in the morning before I rose to travel
Up comes a band of footmen and likewise captain Farrell
I first produced my pistol, for she stole away my rapier
But I couldn't shoot the water so a prisoner I was taken
There's whiskey in the jar
If anyone can aid me, it's my brother in the army
If I can find his station in Cork or in Killarney
And if he'll come save me, we'll go roving near Kilkenny
And I'm sure he'll treat me better than me own sporting Jenny
Whack for the daddy-o
There's whiskey in the jar
There's some takes delight in the carriages a-rolling
And others take delight in the hurley and the bowlin'
But I takes delight in the juice of the barley
And courting pretty fair maids in the morning bright and early
Musha-ring dumma-do-damma-da
Whack for the daddy-o
There's whiskey in the jar
Musha-ring dumma-do-damma-da
In Dublin's fair city
Where the girls are so pretty
I first set my eyes on sweet Molly Malone
As she wheeled her wheelbarrow
Through streets broad and narrow
Crying, "Cockles and mussels, alive, alive, oh!"
Alive, alive, oh
Alive, alive, oh
Crying, "Cockles and mussels, alive, alive, oh"
She was a fishmonger
And sure 'twas no wonder
For so were her father and mother before
And they both wheeled their barrows
Through streets broad and narrow
Crying, "Cockles and mussels, alive, alive, oh
Alive, alive, oh
Alive, alive, oh
Crying, "Cockles and mussels, alive, alive, oh
She died of a fever
And no one could save her
And that was the end of sweet Molly Malone
But her ghost wheels her barrow
Through streets broad and narrow
Crying, "Cockles and mussels, alive, alive, oh
"""

corpus = data.lower().split("\n")
token = {}
UNK_TOKEN = '<unk>'
token[UNK_TOKEN] = len(token) + 1 

for line in corpus:
    for word in line.split():
        if word not in token:
            token[word] = len(token) + 1 

input_seq = []
for line in corpus:
    token_line = []
    for word in line.split():
        token_line.append(token.get(word, token[UNK_TOKEN])) 
    for i in range(1, len(token_line)):
        output_seq = token_line[:i + 1]
        input_seq.append(output_seq)

max_len = max(len(seq) for seq in input_seq)
input_seq = np.array([np.pad(seq, (max_len - len(seq), 0), 'constant') for seq in input_seq])

xs = input_seq[:, :-1]
labels = input_seq[:, -1]
ys = np.zeros((labels.size, len(token) + 1))
ys[np.arange(labels.size), labels] = 1 

class TextDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

dataset = TextDataset(xs, ys)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :]) 
        return x

vocab_size = len(token) + 1
embedding_dim = 240
hidden_dim = 128

model = TextGenerator(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(targets, 1)[1]) 
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

seed_text = "i made a sick dew"
next_words = 30

model.eval()
for _ in range(next_words):
    tokenlist = [token.get(word, token[UNK_TOKEN]) for word in seed_text.split()] 
    
    if len(tokenlist) > max_len:
        tokenlist = tokenlist[-max_len:] 
    else:
        tokenlist = np.pad(tokenlist, (max_len - len(tokenlist), 0), 'constant')

    tokenlist = torch.tensor(tokenlist).unsqueeze(0) 
    with torch.no_grad():
        predicted_probs = model(tokenlist)
        predicted_index = torch.argmax(predicted_probs, dim=-1).item()

    output_word = [word for word, idx in token.items() if idx == predicted_index]
    if output_word:
        seed_text += " " + output_word[0]

print(seed_text)
