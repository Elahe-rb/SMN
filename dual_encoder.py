import torch
from torch import nn
from torch.nn import init
import preprocess_data_smn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# SMN from sequential matching network
class DualEncoder(nn.Module):
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
        super(DualEncoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.emb_h_size = input_size  # embedding dim
        self.rnn_h_size = hidden_size  # rnn hidden dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.p_dropout = dropout

        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)

        W = torch.FloatTensor(self.rnn_h_size, self.rnn_h_size).to(device)
        init.normal_(W)
        self.W = nn.Parameter(W, requires_grad=True)

        self.dropout_layer = nn.Dropout(self.p_dropout)


        ###########################################################################################################
        self.encoder = nn.GRU(self.emb_h_size, self.rnn_h_size, bidirectional=bidirectional, batch_first=True, bias=True, dropout=0).to(device)
        self.conv2d = nn.Conv2d(2, 8, kernel_size=(3, 3))
        conv2d_weight = (param.data for name, param in self.conv2d.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            init.kaiming_normal_(w)

        self.pool2d = nn.MaxPool2d((3, 3), stride=(3, 3))

        self.linear = nn.Linear(16 * 16 * 8, 50)
        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            init.xavier_uniform_(w)


        self.final_linear = nn.Linear(100, 1)
        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            init.xavier_uniform_(w)
        ############################################################################################################

        self.init_weights()

    def init_weights(self):
        init.uniform_(self.encoder.weight_ih_l0, a=-0.01, b=0.01)
        init.orthogonal_(self.encoder.weight_hh_l0)
        self.encoder.weight_ih_l0.requires_grad = True
        self.encoder.weight_hh_l0.requires_grad = True


        glove_embeddings = preprocess_data_smn.load_glove_embeddings(self.vocab)

        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_h_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k, v in glove_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0] * self.emb_h_size)

        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)

        #self.RNN = nn.GRU(self.emb_h_size, self.rnn_h_size, batch_first=True, bidirectional=False, bias=True)
        self.final = nn.Bilinear(self.rnn_h_size, self.rnn_h_size, 1, bias=False)

    def forward(self, utterances, response):
        '''
            utterances:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
            response:(self.batch_size, self.max_sentence_len)
        '''

        #convert a list of seprated utt into a sequence of utt as conv context
        context = utterances.view(utterances.size(0),-1)      #context dim:  [batchsize , (uttnum*seqlength)]

        context_emb = self.embedding(context)  # dim:  [batchsize , (uttnum*seqlength), emb_dim]
        response_emb = self.embedding(response) # dim: [batchsize , (seqlength), emb_dim]   ##??shouldn't be the same size with context???

        context_os, context_hs = self.encoder(
            context_emb)  # context_hs dimensions: ( (numlayers*num direction) * batch_size * hidden_size)
        response_os, response_hs = self.encoder(
            response_emb)  # context_os dimensions: (batch_size * seq_length * hidden_size)
        context_hs = context_hs[
            -1]  # dim: b*h   ::get the hidden of last layer(for single layer this is same as context_hs.squeeze(0))
        response_hs = response_hs[
            -1]  # dim: batch_size * hidden_size <=== (numlayers*num direction) * batch_size * hidden_size
        ### simple dual gru
        context = context_hs.mm(self.W).to(device)  # dimensions: (batch_size x hidden_size)  <== (b*h) , (h*h)
        context_rep = context.view(-1, 1, self.rnn_h_size)  # dimensions: (batch_size x 1 x hidden_size)
        response_rep = response_hs.view(-1, self.rnn_h_size, 1)  # dimensions: (batch_size x hidden_size x 1)

        ans = torch.bmm(context_rep, response_rep).view(-1, 1).to(
            device)  # dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)

        results = torch.sigmoid(ans)  # dim: batchsize * 1
        return results
        # c = self.emb(c)  # BLE
        # r = self.emb(r)
        #
        # c.masked_fill(c_mask.unsqueeze(-1), 0)
        # r.masked_fill(r_mask.unsqueeze(-1), 0)
        #
        # # c_len, dr_len = x_len
        # # c = pack_padded_sequence(c, c_len, batch_first=True, enforce_sorted=False)
        # # c = self.RNN(c)
        # # c = pad_packed_sequence(c[0], batch_first=True, total_length=MAX_LEN)[0]
        #
        # # r = pack_padded_sequence(r, r_len, batch_first=True, enforce_sorted=False)
        # # r = self.RNN(r)
        # # r = pad_packed_sequence(r[0], batch_first=True, total_length=MAX_LEN)[0]
        #
        # #
        # # r_len_sorted, r_len_sorted_idx = torch.sort(r_len, descending=True)
        # # r_len_unsorted_idx = torch.argsort(r_len_sorted_idx)
        # # r_sorted = r.index_select(0, Variable(r_len_sorted_idx))
        # # r_sorted = pack_padded_sequence(r_sorted, r_len_sorted, batch_first=True)
        # # r_sorted = self.RNN(r_sorted)
        # # r_sorted = pad_packed_sequence(r_sorted[0], batch_first=True)[0]
        # # r = r_sorted.index_select(0, Variable(r_len_unsorted_idx))
        # # r = r[:, -1, :]
        #
        # r = self.RNN(r)[0]
        # c = self.RNN(c)[0]
        #
        # r = r[:, -1, :]
        # c = c[:, -1, :]
        #
        # # r = r[[torch.arange(0, r.shape[0]), r_len - 1]]
        # # c = c[[torch.arange(0, c.shape[0]), c_len - 1]]
        #
        # o = self.final(c, r).squeeze()
        # return o
        #

