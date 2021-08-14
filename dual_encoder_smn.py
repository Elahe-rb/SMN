import datetime
import time
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import preprocess_data_smn
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# A simple GRU or LSTM
class Encoder(nn.Module):
    def __init__(
            self,
            vocab,
            input_size,
            hidden_size,
            vocab_size,
            bidirectional,
            rnn_type,
            num_layers,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.vocab = vocab
        self.vocab_size = len(vocab)+1  #+1 is for padding words
        self.emb_h_size = input_size  # embedding dim
        self.rnn_h_size = hidden_size  # rnn hidden dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.p_dropout = dropout

        self.embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(self.emb_h_size, self.rnn_h_size, num_layers=num_layers, dropout=0, bidirectional=bidirectional, batch_first=True).to(device)
        else:
            self.rnn = nn.LSTM(self.emb_h_size, self.rnn_h_size, num_layers=num_layers, dropout=0, bidirectional=bidirectional, batch_first=True).to(device)

        M = torch.FloatTensor(self.rnn_h_size, self.rnn_h_size).to(device)
        init.normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

        W = torch.FloatTensor(self.rnn_h_size, self.rnn_h_size).to(device)
        init.normal_(W)
        self.W = nn.Parameter(W, requires_grad=True)

        M1 = torch.FloatTensor(1, 1).to(device)
        init.normal_(M1)
        self.M1 = nn.Parameter(M1, requires_grad=True)

        M2 = torch.FloatTensor(1, 1).to(device)
        init.normal_(M2)
        self.M2 = nn.Parameter(M2, requires_grad=True)
        
        self.dropout_layer = nn.Dropout(self.p_dropout)
        # self.ReLU = nn.ReLU()
        # self.tanh = nn.Tanh()
        # dense_dim = 2 * self.encoder.hidden_size
        # self.dense = nn.Linear(2*dense_dim, dense_dim).to(device)
        # self.dense2 = nn.Linear(dense_dim, 1).to(device)

        self.layer1 = nn.Sequential(nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=2),  # number of conv kernels = 8
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)).to(device)
        self.fc1 = nn.Linear(125000, self.rnn_h_size).to(device)  # kernel_num * 80 * 80, hidden_size)
        self.fc2 = nn.Linear(self.rnn_h_size, 1).to(device)

        self.init_weights()

    def init_weights(self):
        init.uniform_(self.rnn.weight_ih_l0, a=-0.01, b=0.01)
        init.orthogonal_(self.rnn.weight_hh_l0)
        self.rnn.weight_ih_l0.requires_grad = True
        self.rnn.weight_hh_l0.requires_grad = True

        glove_embeddings = preprocess_data_smn.load_glove_embeddings(self.vocab)

        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_h_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k, v in glove_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0] * self.emb_h_size)

        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)

    def forward(self, contexts_features, responses_features):  #dim:b*seq*num_of_features (0:word_indx 1:df)
        #print("hello, in dual encoder in top_network dir!!")
        # contexts = contexts_features[:, :, 0].long()   #dim: b*seq
        # responses = responses_features[:, :, 0].long()
        # c_idf = contexts_features[:, :, 1]    #dim: b*seq
        # r_idf = responses_features[:, :, 1]
        # c_tf = contexts_features[:, :, 2]
        # r_tf = responses_features[:, :, 2]
        # c_tf_idf = torch.mul(c_idf,c_tf)   #element wise multiplication dim: b*seq <== a*b
        # r_tf_idf = torch.mul(r_idf, r_tf)  # element wise multiplication dim: b*seq <== a*b
        # results = self.att_on_embedding(contexts,responses, c_tf_idf, r_tf_idf)
        results = self.att_on_embedding(contexts_features, responses_features, '', '')
        #results = self.dmn_prf(contexts,responses, c_tf_idf, r_tf_idf)

        return results

    def attention(self, att_context, query, tf_idf):  # dim: (att_context): b*seq*hidden  (query):b*1*hidden
        # energy = torch.bmm(att_context, torch.mm(query.squeeze(1), self.W).unsqueeze(2)) #energy: b*seq*1 <== b*seq*hidden , b*hidden*1
        energy = torch.bmm(att_context, query.transpose(1, 2)).to(device)  # energy: b*seq*1 <== 512*500*300 , 512*300*1
        #energy = torch.bmm(energy, tf_idf)
        Attention_weights = F.softmax(energy, dim=1)  #b*seq*1
        Attention_weights = torch.mul(Attention_weights, tf_idf.unsqueeze(2))
        weighted_average = torch.bmm(att_context.transpose(1, 2), Attention_weights).to(device)  # 512*300*1 <== 512*300*500 , 512*500*1
        output = weighted_average.transpose(1, 2)  # 512*1*300
        return output  # dim: b * 1 * hidden

    def att_on_embedding(self, utterances, responses, c_tf_idf, r_tf_idf): #dim context:b*seq  response:b*seq


        contexts = utterances.view(utterances.size(0),-1)
        contexts_emb = self.embedding(contexts)     #dim: c
        responses_emb = self.embedding(responses)

        context_os, context_hs = self.rnn(
            contexts_emb)  # context_hs dimensions: ( (numlayers*num direction) * batch_size * hidden_size)
        response_os, response_hs = self.rnn(
            responses_emb)  # context_os dimensions: (batch_size * seq_length * hidden_size)
        context_hs = context_hs[
            -1]  # dim: b*h   ::get the hidden of last layer(for single layer this is same as context_hs.squeeze(0))
        response_hs = response_hs[
            -1]  # dim: batch_size * hidden_size <=== (numlayers*num direction) * batch_size * hidden_size

        ##multiply by idf
        #c_df = c_df.float()
        #r_df = r_df.float()
        #context_emb_idf = torch.mul(contexts_emb, c_tf_idf.unsqueeze(2).expand_as(contexts_emb))      #dim    <== b*seq*emb , b*seq
        #response_emb_idf = torch.mul(responses_emb, r_tf_idf.unsqueeze(2).expand_as(responses_emb))

        #### attention:
        #context_att = self.attention(response_emb_idf, context_hs.view(-1, 1, self.rnn_h_size))  # b*1*hidden
        #response_att = self.attention(context_emb_idf, response_hs.view(-1, 1, self.rnn_h_size))  # b*1*hidden <==

        # context_att = self.attention(responses_emb, context_hs.view(-1, 1, self.rnn_h_size), r_tf_idf)  # b*1*hidden
        # response_att = self.attention(contexts_emb, response_hs.view(-1, 1, self.rnn_h_size), c_tf_idf)  # b*1*hidden <==

        ### concat with gru encoding
        # context_concat = torch.cat((context_hs.unsqueeze(1), context_att), 2)  # b*1*(h+h)
        # response_concat = torch.cat((response_hs.unsqueeze(1), response_att), 2)  # b*1*(h+h)
        #
        # context_enc = context_concat
        # response_enc = response_concat
        # response_enc = response_att
        # context_enc = context_att

        ##here needs context_enc and response_enc with dim: b* 1 * h
        # context_enc = context_enc.squeeze(1)  # b*h <== b*1*h
        # context_rep = context_enc.mm(self.M).to(device)  # dimensions: (batch_size x hidden_size)  <== (b*h) , (h*h)
        # context_rep = context_rep.unsqueeze(1)  # dimensions: (batch_size x 1 x hidden_size)
        # response_rep = response_enc.transpose(1, 2)  # dimensions: (batch_size x hidden_size x 1)

        ### simple dual gru
        # context = context_hs.mm(self.M).to(device)  # dimensions: (batch_size x hidden_size)  <== (b*h) , (h*h)
        # context = context.view(-1, 1, self.rnn_h_size)  # dimensions: (batch_size x 1 x hidden_size)
        # response = response_hs.view(-1, self.rnn_h_size, 1)  # dimensions: (batch_size x hidden_size x 1)

        ### simple dual gru
        context = context_hs.mm(self.M).to(device)  # dimensions: (batch_size x hidden_size)  <== (b*h) , (h*h)
        context_rep = context.view(-1, 1, self.rnn_h_size)  # dimensions: (batch_size x 1 x hidden_size)
        response_rep = response_hs.view(-1, self.rnn_h_size, 1)  # dimensions: (batch_size x hidden_size x 1)

        ans = torch.bmm(context_rep, response_rep).view(-1, 1).to(
            device)  # dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)

        results = torch.sigmoid(ans)  # dim: batchsize * 1
        return results

