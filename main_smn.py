import os
import torch
import time
import argparse
import numpy as np
import random
import csv
from config_params_smn import *
from input_params_smn import *
import preprocess_data_smn
import train_smn
import evaluate_smn
import smn
import dual_encoder

#########################  Device configuration ###################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

set_seeds()
#
def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataPath', default=data_file_path)
    parser.add_argument('-trim', default=min_freq)
    parser.add_argument('-maxUttNum', default=max_conv_utt_num)
    parser.add_argument('-maxUttLen', default=max_utterance_length)
    return parser

args, _ = define_args().parse_known_args()

print('hello SMN!!')


##############################  load data and vocab #################################
train_rows, valid_rows, test_rows, vocab, train_uids_rows, valid_uids_rows, test_uids_rows = preprocess_data_smn.load_Data(args)
vocab_size = len(vocab)   #for padding

with open(os.path.join(args.dataPath,"clustered_completegraph.csv")) as f:
    next(f)  #ignore the header line
    reader = csv.reader(f)
    utts_cluster_ids_complete = {rows[0]:rows[5] for rows in reader}



############################ define model ############################################

model = dual_encoder.DualEncoder(
  vocab=vocab,
  input_size=embed_dim, # embedding dim
  hidden_size=hidden_size, # rnn dim
  vocab_size=vocab_size, # vocab size
  bidirectional=False, # really should change!
  rnn_type='gru',
  num_layers=1,
  dropout=dropout_rate
)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
loss_fn = torch.nn.BCELoss()  #binary cross entropy loss
loss_fn.to(device)
#pos_weight = torch.FloatTensor([NUM_PER_SAMPLE - 1]).to(device)
#criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

 #####################################################################################
 ###*****************************  run model **************************************###
 #####################################################################################

lr = learning_rate
best_val_metric = None
losses = [1]


######### train and validation #########################
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model, train_losses = train_smn.train(model, loss_fn, optimizer, train_rows, batch_size, epoch, num_epochs, vocab, max_conv_utt_num, max_utterance_length, device, train_uids_rows, utts_cluster_ids_complete)

    with torch.no_grad():
        val_R1, val_R2, val_R5, acc = evaluate_smn.evaluate(model, valid_rows, evaluate_batch_size, epoch, num_epochs, vocab, max_conv_utt_num, max_utterance_length, device, valid_uids_rows, utts_cluster_ids_complete)

    print('-' * 80)
    description = (
        'Valid: [end of epoch{:3d}|time: {:5.2f}s| R1:{:.3f}| R2:{:.3f}| R5:{:.3f}| Acc:{:.3f}'.format(
            epoch + 1,
            time.time() - epoch_start_time,
            val_R1,
            val_R2,
            val_R5,
            acc
        ))
    print(description)
    print('-' * 80)

    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_metric or val_R1 > best_val_metric:
        torch.save(model.state_dict(), os.path.join(args.dataPath,"saved_model.pth"))
        best_val_metric = val_R1
  #  else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
  #      lr /= 4.0

############################# run model on test data ###################################################
########################################################################################################

# Load the best saved model.
model.load_state_dict(torch.load(os.path.join(args.dataPath,"saved_model.pth")))
model.to(device)

# Run on test data.
test_R1, test_R2, test_R5, test_acc = evaluate_smn.evaluate(model, test_rows, evaluate_batch_size, 0, num_epochs, vocab, max_conv_utt_num, max_utterance_length, device, test_uids_rows, utts_cluster_ids_complete)

print('=' * 89)
description = (
    'Test: [R1: {:.3f} | R2: {:.3f} | R5: {:.3f} | Acc: {:.3f}'.format(
        test_R1,
        test_R2,
        test_R5,
        test_acc
    ))
print(description)
print('=' * 89)
