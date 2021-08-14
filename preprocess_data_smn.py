import torch
import re
import csv
import nltk
import random
import os
from torch.autograd import Variable
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
nltk.download('punkt')

######################################################################
# Load & Preprocess Data
# ---
# Default word tokens
PAD_token = 0  # Used for padding short sentences
EOU_token = 1  # end-of-utt token
EOT_token = 2  # End-of-turn token

class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2count = {}  #what about <UNK> ?  #without trimming
        self.num_words = 0  # Count SOS, EOS, PAD  #without trimming
        self.unique_vocabs = []
        self.contexts_word_to_df = {}
        self.responses_word_to_df = {}
        self.num_cr_pairs_docs = 0

    def addSentence(self, sentence, is_context, is_response):
        words_in_sentece = []
        word_list = sentence.split() #nltk.word_tokenize(sentence)
        for word in word_list:
            self.addWord(word)
            #for tf_idf
            if word not in words_in_sentece:
                words_in_sentece.append(word)
        if is_context:
            self.num_cr_pairs_docs += 1
            for w in words_in_sentece:
                if w not in self.contexts_word_to_df:
                    self.contexts_word_to_df[w] = 1
                else:
                    self.contexts_word_to_df[w] += 1
        if is_response:
            for w in words_in_sentece:
                if w not in self.responses_word_to_df:
                    self.responses_word_to_df[w] = 1
                else:
                    self.responses_word_to_df[w] += 1

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        keep_words.append("<UNK>")
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), (len(self.word2count)+1), len(keep_words) / (len(self.word2count)+1) #+1 is for unk which is not in word2count but in keep words
        ))

        # Reinitialize dictionaries
        #self.word2count = {} #{"SOS": 1000, "EOS": 1000}
        #self.num_words = 0  # Count default tokens

        self.unique_vocabs = keep_words
        #for word in keep_words:
        # #self.addWord(word)

##########################################


# Lowercase, trim, and remove non-letter characters
def normalizeString(s,lemmatizer):
    #s = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(s))))
    #word_list = nltk.word_tokenize(s)
    #s = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    s = s.lower().strip()
    s = re.sub(r"([.!\?\\/+*:&$%#@~=,\-\)\(])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z0-9'!\?]", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def clean_data(rows):
    #tokenization lemma and stem     but stemer is not good at all!!  tried--> tri !!! but in lemma: tried-->try  what about pos?!
    normalized_rows = []
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    for row in rows:
        normalized_row = [normalizeString(r, lemmatizer) for r in row]
        normalized_rows.append(normalized_row)
    return normalized_rows

def readFile(filepath, name):

    reader = csv.reader(open(filepath), delimiter="\t")
    rows = list(reader)[0:]
    print('#',name,'_samples::',len(rows))
    #rows = clean_data(rows)  #if uncomment change _eot_ to eot in numberize function
    return rows

def readUidsFile(filepath):
    reader = csv.reader(open(filepath), delimiter="\t")
    rows = list(reader)[0:]
    return rows

#iter starts from 0
def get_batch(rows,iter,batch_size):
    start = iter * batch_size
    return rows[start: start+batch_size]

def load_vocab(filename):
  lines = open(filename).readlines()
  return {
    word.strip() : i+1
    for i,word in enumerate(lines)
  }

def build_vocab(data,args):
    text = []
    for dialog in data:
        if dialog[0] == '1':  #just for context and true response pairs + negative responses
            for utt in range(1,len(dialog)-1):
                text.extend(dialog[utt].split())
                #text.extend(nltk.word_tokenize(dialog[utt]))
                text.extend(['eot'])
        text.extend(dialog[-1].split())
        #text.extend(nltk.word_tokenize(dialog[-1]))
        text.extend(['eot'])
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

    itos = ['__PAD__','__UNK__'] + list(words_set)
    stoi = {v: i for i, v in enumerate(itos)}

    print('keep_words:: {} / {} = {:.4f}'.format(kept_words, total_words, kept_words / total_words))

    print('keep_text_percentage:: {} / {} = {:.4f}'.format(trimmed_text_length, text_length, trimmed_text_length / text_length))

    return stoi



def build_vocab_1(train_data, valid_data, vocab_path, trim):

    voc = Voc()

    i = 0
    for text in train_data:
        if text[0] == '1':
            context = ''
            for j in range(1, len(text) - 1):
                context = context + text[j] + " eot "      #whole context of a conversation
            voc.addSentence(context, True, False)
            voc.addSentence(text[-1] + " eot ", False, True)  # for context data it contains only context response pairs with label 1
            i += 1

    # for text in valid_data:
    #     if text[0] == '1':
    #         context = ''
    #         for j in range(1, len(text) - 1):
    #             context = context + text[j] + " eot "
    #         voc.addSentence(context, True, False)
    #         voc.addSentence(text[-1] + " eot ", False, True)  #for valid data it contains only ground response
    #         i += 1
    if trim:
        voc.trim(8)

    with open(vocab_path, 'w') as f:
        f.writelines("%s\n" % w for w in voc.unique_vocabs)
    f.close()

    return voc

def load_glove_embeddings(vocab, filename='../glove.6B.200d.txt'):
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

def numberize(inp, vocab, max_utt_num , max_utt_length, is_context):
    #max_len = max_utt_num * max_utt_length
    if is_context:
        nested_inp = inp.split('eot')[:-1]   #-1 is for ignoring the last one which is empty #change to eot for run in cuda and uncomment clean_data
        #nested_inp = [item for item in nested_inp if len(item)>2]
        selected_turns = nested_inp[-min(max_utt_num, len(nested_inp)):]
        selected_words_in_turns = [words.split()[:min(len(words),max_utt_length)] for words in selected_turns]

        padded_nested_results = []
        PAD_SEQUENCE = [0] * max_utt_length
        for turn_sequence in selected_words_in_turns:
            selected_context = list(map(lambda k: vocab.get(k, 1), turn_sequence[:]))
            if(len(selected_context)<max_utt_length):
                selected_context += [0] * (max_utt_length - len(selected_context))
            padded_nested_results.append(selected_context)
        padd_len = len(padded_nested_results)
        if(padd_len<max_utt_num):
            padded_nested_results += [PAD_SEQUENCE] * (max_utt_num - padd_len)
        final_seq = padded_nested_results
        #selected_context = [w for ut in selected_words_in_turns for w in ut]
        #df_features =  list(map(lambda k: dic.contexts_word_to_df.get(k, 1), selected_context))  #it should be .get(k,0) for full data


    else:
        words = inp.split()
        selected_context = words[:min(len(words), max_utt_length)]
        result = list(map(lambda k: vocab.get(k, 1), selected_context))  # [-max_length:]   # 1 is index of unkown words
        pad_first_size = max_utt_length - len(result)
        if len(result) < max_utt_length:
            #result = [0] * pad_first_size + result
            result = result + [0] * pad_first_size   #post padding as is used in smn implementation of paper
        if len(result) != max_utt_length:
            print('errrrrorrr')

        final_seq = result
        #df_features = list(map(lambda k: dic.contexts_word_to_df.get(k, 1), selected_context))

    #tf_features = list(map(lambda k: (selected_context.count(k)/len(selected_context)) , selected_context))
    #idf_features = [math.log(dic.num_cr_pairs_docs / x) for x in df_features]
    #result = list(map(lambda k: vocab.get(k, 1), selected_context))#[-max_length:]   # 1 is index of unkown words



    #result dim: seqlength ===> unsquueze(1) :seq*1
    #result = torch.cat((torch.FloatTensor(result).unsqueeze(1),torch.FloatTensor(idf_features).unsqueeze(1), torch.FloatTensor(tf_features).unsqueeze(1)),1)
    return final_seq   #dim seq*(numOffeatures+1)  here is 2(0:word indx, 1:idf, 2:tf)

#################################

def numberize(data, vocab, max_utt_num , max_utt_length):
    #max_len = max_utt_num * max_utt_length
    numberized_data = []
    for dialog in data:
        numberized_row = []

        #dialog[1:] = [(utt+' eot') for utt in dialog[1:]] #append eot end of all utts
        selected_turns = dialog[-min(max_utt_num, len(dialog)-1):-1]   #=1 is for response and -1 is for first word which is label 0 or 1
        selected_words_in_turns = [words.split()[:min(len(words), max_utt_length)] for words in selected_turns]
        #selected_words_in_turns = [nltk.word_tokenize(words)[:min(len(words), max_utt_length)] for words in selected_turns]
        padded_nested_results = []
        PAD_SEQUENCE = [0] * max_utt_length
        PAD_SEQUENCE[-1] = vocab.get('eot', 1)  # add eot end of each utterance
        for turn_sequence in selected_words_in_turns:
            selected_context = list(map(lambda k: vocab.get(k, 1), turn_sequence[:]))
            if len(selected_context)<max_utt_length:  #padding
                selected_context += [0] * (max_utt_length - len(selected_context))   #post padding
            selected_context[-1] = vocab.get('eot', 1)    #add eot end of each utterance
            padded_nested_results.append(selected_context)
        if len(padded_nested_results)<max_utt_num:
            padded_nested_results += [PAD_SEQUENCE] * (max_utt_num - len(padded_nested_results))


        ## and also for response
        response_words = dialog[-1].split()
        selected_words_in_response = response_words[:min(len(response_words), (max_utt_length*max_utt_num))]
        selected_response = list(map(lambda k: vocab.get(k, 1), selected_words_in_response[:]))
        if (len(selected_response) < (max_utt_length*max_utt_num)):  # padding
            selected_response += [0] * ((max_utt_length*max_utt_num) - len(selected_response))  # post padding
        if len(selected_response) != (max_utt_length*max_utt_num):
            print('errrrrorrr')
        selected_response[-1] = vocab.get('eot', 1)    #add eot end of each utterance

        numberized_row.append(dialog[0])
        numberized_row = numberized_row + padded_nested_results
        numberized_row.append(selected_response)
        numberized_data.append(numberized_row)

    return numberized_data   #dim seq*(numOffeatures+1)  here is 2(0:word indx, 1:idf, 2:tf)

#????????????????????????????????
def process_predict_embed(response):
    stemmer = SnowballStemmer("english")
    response = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(response))))
    response = numberize(response)
    return response

def process_train_data(rows, batch, batch_size, vocab, device, is_topNet, uids_rows, cluster_ids):
    count = 0
    cs = []
    rs = []
    ys = []
    contexts = []
    clusters = []

    batched_rows = get_batch(rows, batch, batch_size)
    #batched_uids_rows = get_batch(uids_rows, batch, batch_size)

    #max_length_context = max(len(row[0]) for row in batched_rows )
    #max_length_response = max(len(row[1]) for row in batched_rows )

    #for (row, rowuid) in zip(batched_rows, batched_uids_rows):
    for row in batched_rows:

        label = row[0]
        #context = ''
        # for i in range(1, len(row) - 1):
        #     context = context + row[i] + str(vocab.get('eot'))
        # context.append(str(vocab.get('eot'))
        # response = row[-1] + str(vocab.get('eot'))

        #[utt.append(vocab.get('eot')) for utt in row[1:-1]]
        #row[-1].append(vocab.get('eot'))
        context = row[1:-1]
        response = row[-1]

        if is_topNet:
            #response_user_id = rowuid[0].split(",")[-1].split("-")[1]
            cluster = 0#cluster_ids.get(response_user_id)

        #context = ''#numberize(context, vocab, max_utt_num, max_utt_length, True)  #dim: context: seq*num_of_features
        #response = ''#numberize(response, vocab, max_utt_num, max_utt_length, False) #dim: response: seq*num_of_features
        label = int(label)
        count += 1
        cs.append(torch.LongTensor(context))
        rs.append(torch.LongTensor(response))
        ys.append(torch.FloatTensor([label]))
        contexts.append(context)
        if is_topNet:
            clusters.append(cluster)

    cs = torch.stack(cs, 0).to(device)  # dim: batchsize * max-utt-num * max_utt-length
    rs = torch.stack(rs, 0).to(device)  #dim: batchsize * max_utt_num
    ys = torch.stack(ys, 0).to(device)  #dim: batchsize * 1

    if is_topNet:
        return cs, rs, ys, clusters

    return cs, rs, ys               #element type: torch.int64

def process_valid_data(rows, batch, batch_size, vocab, max_utt_num, max_utt_length, device):

    batched_cs = []
    batched_rs = [] #contains ground response and its corresponding distractors

    batched_rows = get_batch(rows, batch, batch_size)
    for row in batched_rows:

        context = row[0]
        correct_response = row[1]
        distractors = row[2:]
        temp_dis = []

        context = numberize(context, vocab, max_utt_num, max_utt_length, True)
        correct_response = numberize(correct_response, vocab, max_utt_num, max_utt_length, False)
        distractors = [numberize(distractor, vocab, max_utt_num, max_utt_length, False) for distractor in distractors] #dim: 9*seq*numFeatures

        with torch.no_grad():
            cs = torch.stack([context for i in range(10)], 0).to(device)   #10*seq*numF
            rs = [correct_response]
            rs += [distractor for distractor in distractors]
            rs = torch.stack(rs, 0).to(device)      #dim: 10*seq*numF

            batched_cs.append(cs)
            batched_rs.append(rs)

    return batched_cs, batched_rs   #b*10*seq*numF


def load_Data(args):
    train = readFile(os.path.join(args.dataPath,"train.tsv"),'train')
    train_uids = ''#readUidsFile(os.path.join(args.dataPath,"train_uids.tsv"))
    #train_data = list(zip(train, train_uids))
    #random.shuffle(train_data)
    #random.shuffle(train)
    #train, train_uids = zip(*train_data)
    valid = readFile(os.path.join(args.dataPath,"valid.tsv"),'valid')
    valid_uids = ''#readUidsFile(os.path.join(args.dataPath,"valid_uids.tsv"))
    test = readFile(os.path.join(args.dataPath,"test.tsv"),'test')
    test_uids = ''#readUidsFile(os.path.join(args.dataPath,"test_uids.tsv"))
    #to build vocabulary.txt
    # dic = build_vocab(train , valid, os.path.join(args.dataPath,"vocabulary.txt"), trim = True)
    # vocab = load_vocab(os.path.join(args.dataPath,"vocabulary.txt"))
    vocab = build_vocab(train, args)

    numberized_train = numberize(train, vocab, args.maxUttNum, args.maxUttLen)
    numberized_valid = numberize(valid, vocab, args.maxUttNum, args.maxUttLen)
    numberized_test = numberize(test, vocab, args.maxUttNum, args.maxUttLen)
    return numberized_train, numberized_valid, numberized_test, vocab, train_uids, valid_uids, test_uids
