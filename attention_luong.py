import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch import nn


# Recurrent neural network (many-to-one)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, h_n = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # print('out_size and hidden_size', out.size(), h_n[-1,:,:].size(), h_n.size())
        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        return out, h_n

    

class LuongDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, attention, n_layers, num_classes, device):
    super(LuongDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = 1
    self.drop_prob = 0.1
    self.num_classes = 4
    self.device = device
    # The Attention Mechanism is defined in a separate class
    self.attention = attention
    self.dropout = nn.Dropout(self.drop_prob)
    self.lstm = nn.GRU(self.output_size, self.hidden_size, n_layers, batch_first=True)
    self.classifier = nn.Linear(self.hidden_size*21, self.num_classes)
    
  def forward(self, inputs, hidden, encoder_outputs):
    
    # Passing previous output word (embedded) and hidden state into LSTM cell
    lstm_out, hidden = self.lstm(inputs, hidden)
    # print('hidden', hidden.size(), hidden[0].size(), encoder_outputs.size())
    # Calculating Alignment Scores - see Attention class for the forward pass function
    alignment_scores = self.attention(hidden, encoder_outputs)
    # print('alignment_scores', alignment_scores.size())
    # Softmaxing alignment scores to obtain Attention weights
    attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
    attn_weights = attn_weights.view(inputs.size(0), -1)
    # print('attn_weights', attn_weights)
    
    # Multiplying Attention weights with encoder outputs to get context vector
    context_vector = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs)
    # print('context_vector', context_vector.size(), lstm_out.size())
    lstm_out = lstm_out.reshape(inputs.size(0), 1, -1)
    # Concatenating output from LSTM with context vector
    output = torch.cat((lstm_out, context_vector),-1)
    # print('output', output.size())
    # Pass concatenated vector through Linear layer acting as a Classifier
    output = F.log_softmax(self.classifier(output[:, -1, :]), dim=1)
    # print('final_output', output.size())
    return output, hidden, attn_weights
  
class Attention(nn.Module):
  def __init__(self, hidden_size, method="dot"):
    super(Attention, self).__init__()
    self.method = method
    self.hidden_size = hidden_size
    
    # Defining the layers/weights required depending on alignment scoring method
    if method == "general":
      self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
      
    elif method == "concat":
      self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
      self.weight = nn.Parameter(torch.FloatTensor(64, hidden_size))
  
  def forward(self, decoder_hidden, encoder_outputs):
    if self.method == "dot":
      # For the dot scoring method, no weights or linear layers are involved
      decoder_hidden = decoder_hidden.permute(1, 0, 2)  #(batch, num_layers, hidden_size)
      encoder_outputs = encoder_outputs.permute(0, 2, 1) #(batch, hidden_size, sequence_length)
      return decoder_hidden.bmm(encoder_outputs).squeeze(1)
      # return encoder_outputs.bmm(decoder_hidden).squeeze(-1)
    
    elif self.method == "general":
      # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
      decoder_hidden = decoder_hidden.permute(1, 0, 2)  #(batch, num_layers, hidden_size)
      encoder_outputs = encoder_outputs.permute(0, 2, 1) #(batch, hidden_size, sequence_length)
      out = self.fc(decoder_hidden)
      # print('out', out.size(), encoder_outputs.size(), decoder_hidden.size())
      return out.bmm(encoder_outputs).squeeze(1)
    
    elif self.method == "concat":
      # For concat scoring, decoder hidden state and encoder outputs are concatenated first
      decoder_hidden = decoder_hidden.permute(1, 0, 2)
      out = torch.tanh(self.fc(decoder_hidden+encoder_outputs))
      # print('out and weight', out.size(), self.weight.size())
      return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, attn_method, device):
        super(AutoEncoderRNN, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, num_classes, device)
        self.attention = Attention(hidden_size, attn_method)
        self.decoder = LuongDecoderRNN(hidden_size, input_size, self.attention, num_layers, num_classes, device)

    def forward(self, x):
        # encoded_x = self.encoder(x).expand(-1, sequence_length, -1)
        encoded_x, hidden_n = self.encoder(x)
        # print(encoded_x.size())
        decoded_x, hidden_n_dec, attn_weights = self.decoder(x, hidden_n, encoded_x)

        return decoded_x, hidden_n_dec