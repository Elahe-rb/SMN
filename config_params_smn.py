#﻿stochastic gradient descent with Adam[11] algorithm

num_epochs = 3
learning_rate = 0.001
adam_beta1 = 0.9    #these are the defult betas value for adam
adam_beta2 = 0.999
num_classes = 1

DO_ClEAN = True
IS_SMN = False
min_freq = 10 #8  #10 for udc  #***** Tip ****:: 8 is ok for MSDialog

dataset="UDC"#"UDC"
model_name = 'Dual_GRU'#'Dual_GRU' #'SMN'   #or Dual_GRU


#udc params

batch_size = 100#128#40#512#128#512
evaluate_batch_size = 500#100#500
embed_dim = 200  # embedding dim: this is the input_size in rnn function
hidden_size = 200  # rnn dim hidden states
kernel_size = 3  #for both convolutional and maxpool
kernel_num = 8
dropout_rate = 0.3
# #vocabulary_size = 95154  # vocab size :based on existing tensorflow code for this paper
max_utterance_length = 50
max_conv_utt_num = 10   # the number of utterances in each conversation; if<10 padd it with zero else the most 10 recent utterance is kept


#MsDialog
'''
batch_size = 60#50
evaluate_batch_size = 250
embed_dim = 200  # embedding dim: this is the input_size in rnn function
hidden_size = 100  # rnn dim hidden states
kernel_size = 3  #for both convolutional and maxpool
kernel_num = 2
dropout_rate = 0.2
#vocab_size = 91620
max_utterance_length = 50#90   #90 as in paper said    but also check 50
max_conv_utt_num = 10

'''
