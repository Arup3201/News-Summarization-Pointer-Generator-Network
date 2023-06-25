## Project Folder Structure 👇

# Project-Folder
# -- data
# ---- external
# ---- interim
# ---- preprocessed
# ---- raw
# -- models
# ---- weights.best.hdf5
# -- notebooks
# ---- Text Summarization Project.ipynb
# ---- run model script and predict script.ipynb
# -- reports
# -- src
# ---- data
# ------ make_data.py
# ---- features
# ---- models
# ------ train_model.py
# ------ predict.py
# ---- visualizations


import os
import sys
import collections
import csv
import pandas as pd
import nltk

nltk.download('punkt')

project_dir = sys.argv[1]
data_dir = os.path.join(project_dir, "data")
raw_dir = os.path.join(data_dir, 'raw')
interim_dir = os.path.join(data_dir, "interim")
preprocessed_dir = os.path.join(data_dir, "preprocessed")

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote]

train_file = os.path.join(raw_dir, "train.csv")
val_file = os.path.join(raw_dir, "validation.csv")
test_file = os.path.join(raw_dir, "test.csv")

# Tokenized csv files
tokenized_train_path = os.path.join(interim_dir, 'tokenized_train.csv')
tokenized_val_path = os.path.join(interim_dir, 'tokenized_val.csv')
tokenized_test_path = os.path.join(interim_dir, 'tokenized_test.csv')


expected_train_rows = 287113
expected_val_rows = 13368
expected_test_rows = 11490

finished_dir = os.path.join(preprocessed_dir, "finished_files")

# No of vocabolaries to store
VOCAB_SIZE = 200000


def tokenize_row(x):
    word_tokenizer = nltk.tokenize.word_tokenize

    tok_article = word_tokenizer(x[0])
    tok_highlight = word_tokenizer(x[1])

    tok_article = ' '.join(tok_article)
    tok_highlight = ' '.join(tok_highlight)

    return tok_article, tok_highlight


def check_num_stories(csv_path, expected_rows):
    df = pd.read_csv(csv_path)
    num_stories = df.shape[0]
    if num_stories != expected_rows:
        raise Exception("The csv file %s contains %i rows/stories but should contain %i" %
                        (csv_path, num_stories, expected_rows))


def tokenize_stories(csv_file_path, tokenized_csv_file_path):
    data = pd.read_csv(csv_file_path)
    stories = data.iloc[:, 1:]

    del data

    word_tokenizer = nltk.tokenize.word_tokenize

    tokenized_stories = stories.apply(
        tokenize_row, axis=1, result_type='broadcast')
    tokenized_stories.to_csv(tokenized_csv_file_path)

    # Check whether we tokenized all stories or not, if not raise Exception
    num_orig = pd.read_csv(csv_file_path).shape[0]
    num_tokenized = pd.read_csv(tokenized_csv_file_path).shape[0]

    if num_orig != num_tokenized:
        raise Exception("The tokenized stories csv %s contains %i \
    files, but it should contain the same number as %s (which has %i files). Was \
    there an error during tokenization?" % (tokenized_csv_file_path, num_tokenized, csv_file_path, num_orig))

    # If no problem occurs
    print("Successfully finished tokenizing %s to %s.\n" %
          (csv_file_path, tokenized_csv_file_path))


def to_article_abstract(tokenized_story):
    article, highlight = tokenized_story[0], tokenized_story[1]

    article = article.lower()
    abstract = highlight.lower()

    return article, abstract


def write_to_csv(tokenized_csv_file_path, out_csv, makevocab=False):
    print("Making csv file for stories listed in %s..." %
          tokenized_csv_file_path)

    stories = pd.read_csv(tokenized_csv_file_path)

    num_stories = len(stories)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_csv, 'w') as csvfile:
        # Creating the csv writer
        csvwriter = csv.writer(csvfile)
        fields = ['article', 'highlights']

        # Write the headers
        csvwriter.writerow(fields)

        for idx in stories.index:
            # For every 1000 successful writing
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" %
                      (idx, num_stories, float(idx)*100.0/float(num_stories)))

            story = stories['article'][idx], stories['highlights'][idx]

            article, abstract = to_article_abstract(story)

            csvwriter.writerow([article, abstract])

            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')

                tokens = art_tokens + abs_tokens

                tokens = [t.strip() for t in tokens]
                tokens = [t for t in tokens if t != ""]

                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_csv)

    if makevocab:
        with open(os.path.join(finished_dir, "vocab"), 'w') as writer:
            print("Writing vocab file...")

            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')

        print("Finished writing vocab file")


# Check the stories directories contain the correct number of .story files
check_num_stories(train_file, expected_train_rows)
check_num_stories(val_file, expected_val_rows)
check_num_stories(test_file, expected_test_rows)

# Create some new directories
if not os.path.exists(finished_dir):
    os.makedirs(finished_dir)

# Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
tokenize_stories(train_file, tokenized_train_path)
tokenize_stories(val_file, tokenized_val_path)
tokenize_stories(test_file, tokenized_test_path)

# Read the tokenized stories, do a little postprocessing then write to bin files
write_to_csv(tokenized_train_path, os.path.join(
    finished_dir, "final_train.csv"), makevocab=True)
write_to_csv(tokenized_val_path, os.path.join(finished_dir, "final_val.csv"))
write_to_csv(tokenized_test_path, os.path.join(finished_dir, "final_test.csv"))
