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
        self.emd_dir = emb_dir
        self.vocab = vocab

        print(f'this is SMN all share Model')
        # 参数设定
        self.word_embedding_size = input_size
        self.rnn_units = hidden_size
        self.total_words = vocab_size
        # batch_size指的是正例的个数，然后从负例数据集中随机抽config.negative_samples个负例，再和utterance组成一个完整的负例

        # 需要的模块
        self.word_embedding = nn.Embedding(self.total_words, self.word_embedding_size)

        glove_embeddings = preprocess_data_smn.load_glove_embeddings(self.vocab, self.emd_dir)

        embedding_weights = torch.FloatTensor(self.total_words, self.word_embedding_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k, v in glove_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0] * self.word_embedding_size)

        self.word_embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)

        # 这个版本的模型所有的模块都共享，即utterance都用相同的GRU
        self.utterance_GRU = nn.GRU(self.word_embedding_size, self.rnn_units, bidirectional=False, batch_first=True)
        ih_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)
        # 用于response的GRU
        self.response_GRU = nn.GRU(self.word_embedding_size, self.rnn_units, bidirectional=False, batch_first=True)
        ih_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_ih' in name)
        hh_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_r:
            nn.init.orthogonal_(k)
        for k in hh_r:
            nn.init.orthogonal_(k)
        # 1、初始化参数的方式要注意
        # 2、参数共享的问题要小心
        # 3、conv2d和linear共享参数
        self.conv2d = nn.Conv2d(2, 8, kernel_size=(3, 3))
        conv2d_weight = (param.data for name, param in self.conv2d.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            init.kaiming_normal_(w)

        self.pool2d = nn.MaxPool2d((3, 3), stride=(3, 3))

        self.linear = nn.Linear(16 * 16 * 8, 50)
        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            init.xavier_uniform_(w)

        self.Amatrix = torch.ones((self.rnn_units, self.rnn_units), requires_grad=True)
        init.xavier_uniform_(self.Amatrix)
        self.Amatrix = self.Amatrix.to(device)
        # 最后一层的gru
        self.final_GRU = nn.GRU(50, self.rnn_units, bidirectional=False, batch_first=True)
        ih_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)
        # final_GRU后的linear层
        self.final_linear = nn.Linear(200, 1)
        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            init.xavier_uniform_(w)


    def forward(self, utterance, response):
        '''
            utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
            response:(self.batch_size, self.max_sentence_len)
        '''
        # (batch_size,10,50)-->(batch_size,10,50,200)
        all_utterance_embeddings = self.word_embedding(utterance)
        response_embeddings = self.word_embedding(response)

        # tensorflow:(batch_size,10,50,200)-->分解-->10个array(batch_size,50,200)
        # pytorch:(batch_size,10,50,200)-->(10,batch_size,50,200)
        all_utterance_embeddings = all_utterance_embeddings.permute(1, 0, 2, 3)

        # 先处理response的gru
        response_GRU_embeddings, _ = self.response_GRU(response_embeddings)
        response_embeddings = response_embeddings.permute(0, 2, 1)
        response_GRU_embeddings = response_GRU_embeddings.permute(0, 2, 1)
        matching_vectors = []

        for utterance_embeddings in all_utterance_embeddings:
            matrix1 = torch.matmul(utterance_embeddings, response_embeddings)

            utterance_GRU_embeddings, _ = self.utterance_GRU(utterance_embeddings)
            matrix2 = torch.einsum('aij,jk->aik', [utterance_GRU_embeddings, self.Amatrix])
            matrix2 = torch.matmul(matrix2, response_GRU_embeddings)

            matrix = torch.stack([matrix1, matrix2], dim=1)
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
        # y_pred = F.softmax(logits)
        y_pred = torch.sigmoid(logits)
        return y_pred