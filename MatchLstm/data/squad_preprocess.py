from __future__ import print_function
from tqdm import tqdm
import json
import linecache
import nltk
import numpy as np
import os

# Size train: 30288272
# size dev: 4854279


def data_from_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def list_topics(data):
    list_topics = [data['data'][idx]['title'] for idx in range(0,len(data['data']))]
    return list_topics


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return list(map(lambda x:str(x), tokens))


def token_idx_map(context, context_tokens):
    acc = ''
    current_token_idx = 0
    token_map = dict()
    context_tokens = list(context_tokens)
    
    for char_idx, char in enumerate(context):
        if char != u' ':
            acc += char
            context_token = str(context_tokens[current_token_idx])
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    return token_map


def invert_map(answer_map):
    return {v[1]: [v[0], k] for k, v in answer_map.iteritems()}


def read_write_dataset(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    skipped = 0

    with open(tier +'.context', 'w') as context_file,\
        open(tier +'.question', 'w') as question_file,\
        open(tier +'.answer', 'w') as text_file,\
        open(tier +'.span', 'w') as span_file:

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens = tokenize(context)
                answer_map = token_idx_map(context, context_tokens)

                qas = article_paragraphs[pid]['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    question_tokens = tokenize(question)
                    qn += 1

                    num_answers = range(1)
                    for ans_id in num_answers:
                        # it contains answer_start, text
                        text = qas[qid]['answers'][ans_id]['text']
                        text_tokens = tokenize(text)
                        answer_start = qas[qid]['answers'][ans_id]['answer_start']
                        answer_end = answer_start + len(text)
                        last_word_answer = len(text_tokens[-1]) # add one to get the first char

                        try:
                            a_start_idx = answer_map[answer_start][1]
                            a_end_idx = answer_map[answer_end - last_word_answer][1]

                            # remove length restraint since we deal with it later
                            context_file.write(' '.join(context_tokens) + '\n')
                            question_file.write(' '.join(question_tokens) + '\n')
                            text_file.write(' '.join(text_tokens) + '\n')
                            span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')

                        except Exception as e:
                            skipped += 1

                        an += 1
    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return qn,an


def save_files(prefix, tier, indices):
    with open(tier + '.context', 'w') as context_file,\
        open(tier + '.question', 'w') as question_file,\
        open(tier + '.answer', 'w') as text_file,\
        open(tier + '.span', 'w') as span_file:

        for i in indices:
            context_file.write(linecache.getline('train.context', i))
            question_file.write(linecache.getline('train.question', i))
            text_file.write(linecache.getline('train.answer', i))
            span_file.write(linecache.getline('train.span', i))


def split_tier(prefix, train_percentage = 0.9, shuffle=False):
    # Get number of lines in file
    context_filename = 'train.context'
    with open(context_filename) as current_file:
        num_lines = sum(1 for line in current_file)
    # Get indices and split into two files
    indices_dev = list(range(num_lines)[int(num_lines * train_percentage)::])
    if shuffle:
        np.random.shuffle(indices_dev)
        print("Shuffling...")
    save_files(prefix, 'val', indices_dev)
    indices_train = list(range(num_lines)[:int(num_lines * train_percentage)])
    if shuffle:
        np.random.shuffle(indices_train)
    save_files(prefix, 'train', indices_train)


if __name__ == '__main__':
    data_prefix = os.path.join("data", "squad")
    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"
    train_data = data_from_json(train_filename)
    train_num_questions, train_num_answers = read_write_dataset(train_data, 'train', data_prefix)

    # In train we have 87k+ questions, and one answer per question.
    # The answer start range is also indicated

    # 1. Split train into train and validation into 95-5
    # 2. Shuffle train, validation
    print("Splitting the dataset into train and validation")
    split_tier(data_prefix, 0.95, shuffle=True)
    print("Processed {} questions and {} answers in train".format(train_num_questions, train_num_answers))

    # In dev, we have 10k+ questions, and around 3 answers per question (totaling
    # around 34k+ answers).
    dev_data = data_from_json(dev_filename)
    list_topics(dev_data)
    dev_num_questions, dev_num_answers = read_write_dataset(dev_data, 'dev', data_prefix)
    print("Processed {} questions and {} answers in dev".format(dev_num_questions, dev_num_answers))
