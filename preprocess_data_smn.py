import torch
import re
import csv
import nltk
import numpy as np
import os
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
nltk.download('punkt')

######################################################################

# class Voc:
#     def __init__(self):
#         self.trimmed = False
#         self.word2count = {}  #what about <UNK> ?  #without trimming
#         self.num_words = 0  # Count SOS, EOS, PAD  #without trimming
#         self.unique_vocabs = []
#         self.contexts_word_to_df = {}
#         self.responses_word_to_df = {}
#         self.num_cr_pairs_docs = 0
#
#     def addSentence(self, sentence, is_context, is_response):
#         words_in_sentece = []
#         word_list = sentence.split() #nltk.word_tokenize(sentence)
#         for word in word_list:
#             self.addWord(word)
#             #for tf_idf
#             if word not in words_in_sentece:
#                 words_in_sentece.append(word)
#         if is_context:
#             self.num_cr_pairs_docs += 1
#             for w in words_in_sentece:
#                 if w not in self.contexts_word_to_df:
#                     self.contexts_word_to_df[w] = 1
#                 else:
#                     self.contexts_word_to_df[w] += 1
#         if is_response:
#             for w in words_in_sentece:
#                 if w not in self.responses_word_to_df:
#                     self.responses_word_to_df[w] = 1
#                 else:
#                     self.responses_word_to_df[w] += 1
#
#     def addWord(self, word):
#         if word not in self.word2count:
#             self.word2count[word] = 1
#             self.num_words += 1
#         else:
#             self.word2count[word] += 1
#
#     # Remove words below a certain count threshold
#     def trim(self, min_count):
#         if self.trimmed:
#             return
#         self.trimmed = True
#
#         keep_words = []
#
#         keep_words.append("<UNK>")
#         for k, v in self.word2count.items():
#             if v >= min_count:
#                 keep_words.append(k)
#
#         print('keep_words {} / {} = {:.4f}'.format(
#             len(keep_words), (len(self.word2count)+1), len(keep_words) / (len(self.word2count)+1) #+1 is for unk which is not in word2count but in keep words
#         ))
#
#         # Reinitialize dictionaries
#         #self.word2count = {} #{"SOS": 1000, "EOS": 1000}
#         #self.num_words = 0  # Count default tokens
#
#         self.unique_vocabs = keep_words
#         #for word in keep_words:
#         # #self.addWord(word)
#
# ##########################################
#
#

# def load_vocab(filename):
#   lines = open(filename).readlines()
#   return {
#     word.strip() : i+1
#     for i,word in enumerate(lines)
#   }

# def build_vocab_1(train_data, valid_data, vocab_path, trim):
#
#     voc = Voc()
#
#     i = 0
#     for text in train_data:
#         if text[0] == '1':
#             context = ''
#             for j in range(1, len(text) - 1):
#                 context = context + text[j] + " eot "      #whole context of a conversation
#             voc.addSentence(context, True, False)
#             voc.addSentence(text[-1] + " eot ", False, True)  # for context data it contains only context response pairs with label 1
#             i += 1
#
#     # for text in valid_data:
#     #     if text[0] == '1':
#     #         context = ''
#     #         for j in range(1, len(text) - 1):
#     #             context = context + text[j] + " eot "
#     #         voc.addSentence(context, True, False)
#     #         voc.addSentence(text[-1] + " eot ", False, True)  #for valid data it contains only ground response
#     #         i += 1
#     if trim:
#         voc.trim(10)
#
#     with open(vocab_path, 'w') as f:
#         f.writelines("%s\n" % w for w in voc.unique_vocabs)
#     f.close()
#
#     return voc

def load_glove_embeddings(vocab, filename):
    print('load glove embedding ...')
    lines = open(filename).readlines()
    embeddings = {}
    not_oov = 0
    for line in lines:
        word = line.split()[0]
        embedding = list(map(float, line.split()[1:]))
        if word in vocab:
            embeddings[vocab[word]] = embedding
            not_oov = not_oov + 1
    print('#OOV:: {} / {} = {:.4f}'.format( (len(vocab)-not_oov),len(vocab), (len(vocab)-not_oov)/len(vocab)))
    return embeddings

####################################################################################################
def numberize_smn(data, vocab, max_utt_num , max_utt_length, device):

    #ToDo: check this!
    #max_len = max_utt_num * max_utt_length
    max_len = max_utt_length
    cs = []
    rs = []
    ys = []

    for dialog in data:

        #dialog[1:] = [(utt+' EOT') for utt in dialog[1:]] #append eot end of all utts
        label = dialog[0]
        context = dialog[1:-1]
        response = dialog[-1]

        selected_turns = context[-min(max_utt_num, len(context)):]
        selected_words_in_turns = [words.split()[:min(len(words), max_utt_length)] for words in selected_turns]

        #??or
        # selected_words_in_turns = [nltk.word_tokenize(words)[:min(len(words), max_utt_length)] for words in selected_turns]
        #selected_context = [w for ut in selected_words_in_turns for w in ut]

        selected_nested_context_idx = []
        PAD_SEQUENCE = [0] * max_utt_length
        #PAD_SEQUENCE[-1] = vocab.get('eot', 1)  # add eot end of each utterance
        for turn_sequence in selected_words_in_turns:
            context_idx = list(map(lambda k: vocab.get(k, 1), turn_sequence[:-1])) #-1 is for deleting the last word to substitute with EOT in next line of code
            context_idx.append(vocab.get('EOT',1))   ##!! if you want to comment this one, also change turn_sequence[:-1] to turn_sequence[:-] in previous line
            if len(context_idx) < max_utt_length:  #padding
                context_idx = [0] * (max_utt_length - len(context_idx)) + context_idx   #first padding
            selected_nested_context_idx.append(context_idx)
        if len(selected_nested_context_idx) < max_utt_num:
            selected_nested_context_idx = ([PAD_SEQUENCE] * (max_utt_num - len(selected_nested_context_idx))) + selected_nested_context_idx


        ## and also for response
        response_words = response.split()
        selected_response = response_words[:min(len(response_words), max_len)]
        selected_response[-1] = 'EOT'
        response_idx = list(map(lambda k: vocab.get(k, 1), selected_response[:]))
        if (len(response_idx) < max_len):  # padding
            response_idx = [0] * (max_len - len(response_idx)) + response_idx # first padding

        cs.append(torch.tensor(selected_nested_context_idx))
        rs.append(torch.tensor(response_idx))
        ys.append(torch.FloatTensor([int(label)]))

    cs = torch.stack(cs, 0).to(device)  # dim: batchsize * ut length * numoffeatures
    rs = torch.stack(rs, 0).to(device)
    ys = torch.stack(ys, 0).to(device)

    return cs, rs, ys


def numberize_rnn(data, vocab, max_utt_num, max_utt_length, device):
    max_len = max_utt_num * max_utt_length
    cs = []
    rs = []
    ys = []

    for dialog in data:
        #1***dialog[1:] = [(utt+' EOT') for utt in dialog[1:]] #append eot end of all utts
        label = dialog[0]
        context = dialog[1:-1]
        response = dialog[-1]
        #numberized_row = []

        selected_turns = context[-min(max_utt_num, len(context)):]
        selected_words_in_turns = [words.split()[:min(len(words), max_utt_length)] for words in selected_turns]
        # ??or
        # selected_words_in_turns = [nltk.word_tokenize(words)[:min(len(words), max_utt_length)] for words in selected_turns]
        # 1***
        selected_context = [w for ut in selected_words_in_turns for w in (ut[:-1]+['EOT'])]
        context_idx = list(map(lambda k: vocab.get(k, 1), selected_context))
        if len(context_idx) < max_len:
            context_idx = [0] * (max_len - len(context_idx)) + context_idx    #first_padding

        response_words = response.split()
        selected_response = response_words[:min(len(response_words), max_len)]
        selected_response[-1] = 'EOT'
        response_idx = list(map(lambda k: vocab.get(k, 1), selected_response))  # [-max_length:]   # 1 is index of unkown words
        if len(response_idx) < max_len:
            response_idx = [0] * (max_len - len(response_idx)) + response_idx  # first_padding

        cs.append(torch.tensor(context_idx))
        rs.append(torch.tensor(response_idx))
        ys.append(torch.FloatTensor([int(label)]))

    cs = torch.stack(cs, 0).to(device)  # dim: batchsize * ut length * numoffeatures
    rs = torch.stack(rs, 0).to(device)
    ys = torch.stack(ys, 0).to(device)

    return cs, rs, ys

def get_batch(rows,batch,batch_size):
    start = batch * batch_size
    return rows[start: start+batch_size]

def process_data(rows, batch, batch_size, vocab, args, device):

    batched_rows = get_batch(rows, batch, batch_size)

    if args.isSMN:
        cs, rs, ys = numberize_smn(batched_rows, vocab, args.maxUttNum, args.maxUttLen, device)
    else:
        cs, rs, ys = numberize_rnn(batched_rows, vocab, args.maxUttNum, args.maxUttLen, device)

    return cs, rs, ys

# def get_data_loaders(train_rows, valid_rows, test_rows, args, device):
#
#     train_data_loader = DataLoader(process_data(train_rows, device), batch_size=args.batchSize, shuffle=True)
#     valid_data_loader = DataLoader(process_data(valid_rows, device), batch_size=args.batchSize, shuffle=False)
#     test_data_loader = DataLoader(process_data(test_rows, device), batch_size=args.batchSize, shuffle=False)

#    return train_data_loader, valid_data_loader, test_data_loader
#########################################################################################################################

def build_vocab(data,args):
    text = []
    for dialog in data:
        if dialog[0] == '1':  #just for context and true response pairs + negative responses
            for utt in range(1,len(dialog)-1):
                text.extend(dialog[utt].split())
                #text.extend(nltk.word_tokenize(dialog[utt]))
                #text.extend(['eot'])
        text.extend(dialog[-1].split())
        #text.extend(nltk.word_tokenize(dialog[-1]))
        #text.extend(['eot'])
    vocab_counter = Counter(text)
    text_length = sum(vocab_counter.values())
    total_words = len(list(vocab_counter.keys()))
    unk_count = 0
    for w in list(vocab_counter.keys()):
        if vocab_counter[w] < args.trim:
            unk_count += vocab_counter[w]
            vocab_counter.pop(w)
    trimmed_text_length = sum(vocab_counter.values())
    words_set = set(vocab_counter.keys())
    kept_words = len(words_set)

    itos = ['PAD','UNK','EOT'] + list(words_set)
    stoi = {v: i for i, v in enumerate(itos)}

    print('keep_words:: {} / {} = {:.4f}'.format(kept_words, total_words, kept_words / total_words))

    print('keep_text_percentage:: {} / {} = {:.4f}'.format(trimmed_text_length, text_length, trimmed_text_length / text_length))

    return stoi


def convert_udc_valid_test_format_to_msdialog(data, args):
    new_data = []
    for row in data:
        context = row[0]
        response = row[1]
        distractors = row[2:]

        splitter = 'eot' if args.doClean else '__eot__'
        utterances = context.split(splitter)  # if use clean data change __eot__ to eot
        new_row = []
        new_row.append('1')
        new_row = new_row + utterances[:-1]  # -1 is for ignore the last one which is empty
        new_row.append(response)
        new_data.append(new_row)

        for distractor in distractors:
            new_row = []
            new_row.append('0')
            new_row = new_row + utterances[:-1]  # -1 is for ignore the last one which is empty
            new_row.append(distractor)
            new_data.append(new_row)
    return new_data

def convert_udc_format_to_msdialog(train,valid,test,args):

    new_train = []
    for row in train:
        new_row = []
        context, response, label = row

        splitter = 'eot' if args.doClean else '__eot__'
        utterances = context.split(splitter)   #if use clean data change __eot__ to eot
        new_row.append(label)
        new_row = new_row + utterances[:-1]   #-1 is for ignore the last one which is empty
        new_row.append(response)
        new_train.append(new_row)

    new_valid = convert_udc_valid_test_format_to_msdialog(valid, args)
    new_test = convert_udc_valid_test_format_to_msdialog(test, args)

    return new_train, new_valid, new_test


# Lowercase, trim, and remove non-letter characters
def normalizeString(str):
    #?? add tokenization lemma and stem
    #lemmatizer = WordNetLemmatizer()
    #s = ' '.join(list(map(lemmatizer.lemmatize, nltk.word_tokenize(str))))
    s = re.sub(r"([.!\?\\/+*:&$%#@~=,\-\)\(])", r" \1 ", str)
    s = re.sub(r"[^a-zA-Z0-9'!\?@:]", r" ", s)
    #s = re.sub(r"\s+", r" ", s).strip()
    s = s.lower().strip()
    stemmer = SnowballStemmer("english")
    s = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(s))))

    return s

def clean_data(rows):
    normalized_rows = []
    for row in rows:
        normalized_row = [normalizeString(r) for r in row]
        normalized_rows.append(normalized_row)
    return normalized_rows


def readFile(args, name):

    if args.dataset == "UDC":
        reader = csv.reader(open(os.path.join(args.dataPath,name+".csv")))
        rows = list(reader)[1:]    #the first line is just column names
    elif args.dataset == "MSDialog":
        reader = csv.reader(open(os.path.join(args.dataPath,name+".tsv")), delimiter="\t")
        rows = list(reader)[0:]
    print(f'# {name}_samples::{len(rows)}')

    if args.doClean:
        print(f'# cleaning {name}_data ...')
        rows = clean_data(rows)

    return rows


def load_Data(args):
    train = readFile(args,'train')
    train_uids = ''#readUidsFile(os.path.join(args.dataPath,"train_uids.tsv"))
    train_data = list(zip(train, train_uids))
    #random.shuffle(train)
    #train, train_uids = zip(*train_data)
    valid = readFile(args,'valid')
    valid_uids = ''#readUidsFile(os.path.join(args.dataPath,"valid_uids.tsv"))
    test = readFile(args,'test')
    test_uids = ''#readUidsFile(os.path.join(args.dataPath,"test_uids.tsv"))
    #to build vocabulary.txt
    # dic = build_vocab(train , valid, os.path.join(args.dataPath,"vocabulary.txt"), trim = True)
    # vocab = load_vocab(os.path.join(args.dataPath,"vocabulary.txt"))
    if args.dataset == "UDC":
        print('converting udc format to msdialog ...')
        train, valid, test = convert_udc_format_to_msdialog(train,valid,test,args)

    print('building vocabulary ...')
    vocab = build_vocab(train, args)
    print(f'vocabulary build with size: {len(vocab)}')
    return train, valid, test, vocab, train_uids, valid_uids, test_uids

    # print('data numberization ...')
    # if args.isSMN:
    #     numberized_train = numberize_smn(train, vocab, args.maxUttNum, args.maxUttLen)
    #     numberized_valid = numberize_smn(valid, vocab, args.maxUttNum, args.maxUttLen)
    #     numberized_test = numberize_smn(test, vocab, args.maxUttNum, args.maxUttLen)
    #
    # else:
    #     numberized_train = numberize_rnn(train, vocab, args.maxUttNum, args.maxUttLen)
    #     numberized_valid = numberize_rnn(valid, vocab, args.maxUttNum, args.maxUttLen)
    #     numberized_test = numberize_rnn(test, vocab, args.maxUttNum, args.maxUttLen)
    #
    # #calss list:[str,list,list]
    # return numberized_train, numberized_valid, numberized_test, vocab, train_uids, valid_uids, test_uids
