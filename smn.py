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


# SMN from sequential matching network
class SMN(nn.Module):
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
            emb_dir
    ):
        super(SMN, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.emb_h_size = input_size  # embedding dim
        self.rnn_h_size = hidden_size  # rnn hidden dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.p_dropout = dropout
        self.emd_dir = emb_dir


        self.embedding = nn.Embedding(vocab_size, self.emb_h_size, padding_idx=0)

        M_1 = torch.FloatTensor(self.rnn_h_size, self.rnn_h_size).to(device)
        init.normal_(M_1)
        self.M = nn.Parameter(M_1, requires_grad=True)

        M_2 = torch.FloatTensor(self.rnn_h_size, self.rnn_h_size).to(device)
        init.normal_(M_2)
        self.W = nn.Parameter(M_2, requires_grad=True)


        self.dropout_layer = nn.Dropout(self.p_dropout)


        ###########################################################################################################
        self.sentence_GRU = nn.GRU(self.emb_h_size, self.rnn_h_size, bidirectional=bidirectional, batch_first=True, dropout=0).to(device)
        ih_u = (param.data for name, param in self.sentence_GRU.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.sentence_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)
        #用于response的GRU
        #self.sentence_GRU = nn.GRU(self.emb_h_size, self.rnn_h_size, bidirectional=bidirectional, batch_first=True, dropout=0).to(device)
        # ih_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_ih' in name)
        # hh_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_hh' in name)
        # for k in ih_r:
        #     nn.init.orthogonal_(k)
        # for k in hh_r:
        #     nn.init.orthogonal_(k)

        #values are based on SMN paper
        cnn_out_channels = 8
        cnn_kernel_size = 3
        cnn_padding = 0
        #cnn_out_dim = cnn_out_channels * ((config.max_sent + 2 * cnn_padding - (cnn_kernel_size - 1)) // cnn_kernel_size) ** 2
        match_dim = 50

        self.conv2d = nn.Conv2d(2, cnn_out_channels, kernel_size=(cnn_kernel_size, cnn_kernel_size))
        conv2d_weight = (param.data for name, param in self.conv2d.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            init.kaiming_normal_(w)

        self.pool2d = nn.MaxPool2d((cnn_kernel_size, cnn_kernel_size), stride=(3, 3))

        self.linear = nn.Linear(2048, match_dim)#(16 * 16 * 8, match_dim)
        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            init.xavier_uniform_(w)

        self.Amatrix = torch.ones((self.rnn_h_size, self.rnn_h_size), requires_grad=True)
        init.xavier_uniform_(self.Amatrix)
        self.Amatrix = self.Amatrix.to(device)

        self.Bmatrix = torch.ones((self.emb_h_size, self.emb_h_size), requires_grad=True)
        init.xavier_uniform_(self.Bmatrix)
        self.Bmatrix = self.Bmatrix.to(device)

        self.final_GRU = nn.GRU(match_dim, self.rnn_h_size, bidirectional=False, batch_first=True)
        ih_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)

        self.final_linear = nn.Linear(self.rnn_h_size, 1)
        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            init.xavier_uniform_(w)
        ############################################################################################################

        self.init_weights()

    def init_weights(self):
        init.uniform_(self.sentence_GRU.weight_ih_l0, a=-0.01, b=0.01)
        init.orthogonal_(self.sentence_GRU.weight_hh_l0)
        self.sentence_GRU.weight_ih_l0.requires_grad = True
        self.sentence_GRU.weight_hh_l0.requires_grad = True

        glove_embeddings = preprocess_data_smn.load_glove_embeddings(self.vocab, self.emd_dir)

        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_h_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k, v in glove_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0] * self.emb_h_size)

        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)

    def forward(self, utterance, response):
        '''
            utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
            response:(self.batch_size, self.max_sentence_len)
        '''
        #uttereance: (batch_size,10(uttNum),50(uttlength))-->(batch_size,10,50,200)
        all_utterance_embeddings = self.embedding(utterance)
        # tensorflow:(batch_size,10,50,200)-->分解-->10个array(batch_size,50,200)
        # pytorch:(batch_size,10,50,200)-->(10,batch_size,50,200)
        all_utterance_embeddings = all_utterance_embeddings.permute(1, 0, 2, 3)

        # response:(batch_size,50)-->(batch_size,50,200)
        response_embeddings = self.embedding(response)

        # response_GRU_embeddings:(batch_size,50,embdsize)-->(batch_size,50,hiddensize)
        response_GRU_embeddings, _ = self.sentence_GRU(response_embeddings)
        response_embeddings = response_embeddings.permute(0, 2, 1)
        response_GRU_embeddings = response_GRU_embeddings.permute(0, 2, 1)
        matching_vectors = []

        for utterance_embeddings in all_utterance_embeddings:
            matrix1 = torch.einsum('aij,jk->aik', [utterance_embeddings, self.Amatrix]) #size:batchsize*50*200
            matrix1 = torch.matmul(matrix1, response_embeddings)     # batchsize*50*50          <--- size:batchsize*50*200 ,  size:batchsize*200*50
            #matrix1 = torch.matmul(utterance_embeddings, response_embeddings)  # batch*utlen*utlen<-- batch*uttlength*embdim   , batch*embdim*uttlength

            #b*50*200
            utterance_GRU_embeddings, _ = self.sentence_GRU(utterance_embeddings)
            matrix2 = torch.einsum('aij,jk->aik', [utterance_GRU_embeddings, self.Bmatrix])
            matrix2 = torch.matmul(matrix2, response_GRU_embeddings) #matrix2:: batchsize*50*50

            matrix = torch.stack([matrix1, matrix2], dim=1)  #torch.size(b*2*50*50)
            # matrix:(batch_size,channel,seq_len,embedding_size)
            conv_layer = self.conv2d(matrix)
            # add activate function
            conv_layer = F.relu(conv_layer)
            pooling_layer = self.pool2d(conv_layer)
            # flatten
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)
            matching_vector = self.linear(pooling_layer)
            # add activate function
            matching_vector = torch.tanh(matching_vector)
            matching_vectors.append(matching_vector)

        _, last_hidden = self.final_GRU(torch.stack(matching_vectors, dim=1))
        last_hidden = torch.squeeze(last_hidden)
        logits = self.final_linear(last_hidden)

        # use CrossEntropyLoss,this loss function would accumulate softmax
        #y_pred = F.softmax(logits)    #logit dim: [batchsize,1]
        #y_pred = logits)
        y_pred = torch.sigmoid(logits)
        #predddd = F.softmax(logits,dim=0)
        return y_pred     #dim: batchsize*1

    '''
    def forward(self, contexts_features, responses_features, scores, data_clusters):  # dim:b*seq*num_of_features (0:word_indx 1:df)   scores: dim: b   :is a vector of batch size
        # print("hello, in dual encoder in top_network dir!!")
        contexts = contexts_features[:, :, 0].long()  # dim: b*seq
        responses = responses_features[:, :, 0].long()
        c_idf = contexts_features[:, :, 1]  # dim: b*seq
        r_idf = responses_features[:, :, 1]
        c_tf = contexts_features[:, :, 2]
        r_tf = responses_features[:, :, 2]
        c_tf_idf = torch.mul(c_idf, c_tf)  # element wise multiplication dim: b*seq <== a*b
        r_tf_idf = torch.mul(r_idf, r_tf)  # element wise multiplication dim: b*seq <== a*b
        results = self.att_on_embedding(contexts, responses, c_tf_idf, r_tf_idf, scores.to(device),
                                        data_clusters.to(device))
        # results = self.dmn_prf(contexts,responses, c_tf_idf, r_tf_idf)

        return results
    '''

    def attention(self, att_context, query, tf_idf):  # dim: (att_context): b*seq*hidden  (query):b*1*hidden
        # energy = torch.bmm(att_context, torch.mm(query.squeeze(1), self.W).unsqueeze(2)) #energy: b*seq*1 <== b*seq*hidden , b*hidden*1
        energy = torch.bmm(att_context, query.transpose(1, 2)).to(device)  # energy: b*seq*1 <== 512*500*300 , 512*300*1
        # energy = torch.bmm(energy, tf_idf)
        Attention_weights = F.softmax(energy, dim=1)  # b*seq*1
        Attention_weights = torch.mul(Attention_weights, tf_idf.unsqueeze(2))
        weighted_average = torch.bmm(att_context.transpose(1, 2), Attention_weights).to(
            device)  # 512*300*1 <== 512*300*500 , 512*500*1
        output = weighted_average.transpose(1, 2)  # 512*1*300
        return output  # dim: b * 1 * hidden

    def att_on_embedding(self, contexts, responses, c_tf_idf, r_tf_idf, scores,
                         data_clusters):  # dim context:b*seq  response:b*seq

        contexts_emb = self.embedding(contexts)  # dim: c
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
        # c_df = c_df.float()
        # r_df = r_df.float()
        # context_emb_idf = torch.mul(contexts_emb, c_tf_idf.unsqueeze(2).expand_as(contexts_emb))      #dim    <== b*seq*emb , b*seq
        # response_emb_idf = torch.mul(responses_emb, r_tf_idf.unsqueeze(2).expand_as(responses_emb))

        #### attention:
        # context_att = self.attention(response_emb_idf, context_hs.view(-1, 1, self.rnn_h_size))  # b*1*hidden
        # response_att = self.attention(context_emb_idf, response_hs.view(-1, 1, self.rnn_h_size))  # b*1*hidden <==

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

        scores = scores.view(-1, 1)  # convert b ---> b*1
        n_scores = scores * data_clusters
        n_ans = ans * data_clusters
        res = (n_ans.mm(self.M1).to(device)) + (n_scores.mm(self.M2).to(device))

        results = torch.sigmoid(res)  # dim: batchsize * 1
        return results

