import torch
import torch.nn as nn

class Network(nn.Module):
  def __init__(self,embedding_matrix,hidden_dim,no_layers=1):
    super().__init__()
    vocab_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    self.hidden_dim = hidden_dim
    self.n_layers = no_layers
    self.embedding = nn.Embedding(vocab_size,embedding_dim)
    self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    self.embedding.weight.requires_grad = False

    self.lstm = nn.LSTM(embedding_dim,hidden_size=self.hidden_dim,batch_first=True,num_layers=self.n_layers)
    self.dropout = nn.Dropout(0.2)
    self.linear1 = nn.Linear(self.hidden_dim,self.hidden_dim//2)
    self.linear2 = nn.Linear(self.hidden_dim//2,2)

  def forward(self,x):
    batch_size= x.shape[0]     #batch_size,seq_len
    x = self.embedding(x)      #batch_size,seq_len,embed_dim
    lstm_out,h = self.lstm(x)  #batch_size,seq_len,hidden_dim*no_layers 
    lstm_out = self.dropout(lstm_out)
    lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)  #batch_size*seq_len*no_layers,hidden_dim
    out = self.linear1(lstm_out)   #batch_size*seq_len*no_layers, out_dim
    out = self.linear2(out)
    out = out.view(batch_size,-1) #batch_size,seq_len*out_dim*no_layers
    return out[:,-2:]