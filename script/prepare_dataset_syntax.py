""" Generate data (in .csv file) for gloss selection task.

Headers in the csv file:
    - id:           id of each csv record
    - sentence:     sentence containing a word to be disambiguated (surrounded by 2 '[TGT]' tokens)
    - sense_keys:   list of candidate sense keys
    - glosses:      list of gloss definition of each candidate sense key
    - targets:      list of indices of the correct sense keys (ground truths)
"""

import argparse
import csv
import random
import re
from pathlib import Path
from xml.etree.ElementTree import ElementTree

from tqdm import tqdm

from utils.wordnet import get_example_sentences, get_glosses, get_all_wordnet_lemma_names
from utils.dataset_syntax import get_pos_tokens

HEADERS = ['id', 'sentence', 'pos', 'sense_keys', 'glosses', 'gloss_word_and_poss', 'targets']
TGT_TOKEN = '[TGT]'
special_token_pos = ['[NONE]']
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Path to directory consisting of a .xml file and a .txt file "
             "corresponding to the sense-annotated data and its gold keys respectively."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the .csv file will be written."
    )
    parser.add_argument(
        "--max_num_gloss",
        type=int,
        default=None,
        help="Maximum number of candidate glosses a record can have (include glosses from ground truths)"
    )
    parser.add_argument(
        "--use_augmentation",
        action='store_true',
        help="Whether to augment training dataset with example sentences from WordNet"
    )

    parser.add_argument(
        "--spacy_model",
        default="en_core_web_sm",
        help="Which spacy model to use. The base model is en_core_web_sm; the transformer model is en_core_web_trf."
    )

    parser.add_argument(
        "--use_pos",
        action='store_true',
        help="Whether to add POS to the data."
    )

    #parser.add_argument(
    #    "--use_dependencies",
    #    action='store_true',
    #    help="Whether to add dependencies to the data."
    #)

    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    corpus_name = corpus_dir.name.lower()
    xml_path = str(corpus_dir.joinpath(f"{corpus_name}.data.xml"))
    txt_path = str(corpus_dir.joinpath(f"{corpus_name}.gold.key.txt"))
    output_filename = f"{corpus_name}"
    if args.max_num_gloss:
        output_filename += f"-max_num_gloss={args.max_num_gloss}"
    if args.use_augmentation:
        output_filename += "-augmented"
    if args.use_pod:
        output_filename += "-POS"
    csv_path = str(Path(args.output_dir).joinpath(f"{output_filename}.csv"))

    print("Creating data for gloss selection task...")
    record_count = 0
    gloss_count = 0
    max_gloss_count = 0

    xml_root = ElementTree(file=xml_path).getroot()
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(HEADERS)

        def _write_to_csv(_id, _sentence, _lemma, _pos, _pos_list, _gold_keys):
            nonlocal record_count, gloss_count, max_gloss_count

            # Make _pos to None to get all senses.
            # for WordNet_pos in ["ADJ", "ADV", "NOUN", "VERB"]:
            sense_info = get_glosses(_lemma, None)
            if args.max_num_gloss is not None:
                sense_gloss_pairs = []
                for k in _gold_keys:
                    sense_gloss_pairs.append((k, sense_info[k]))
                    del sense_info[k]

                remainder = args.max_num_gloss - len(sense_gloss_pairs)
                if len(sense_info) > remainder:
                    for p in random.sample(sense_info.items(), remainder):
                        sense_gloss_pairs.append(p)
                elif len(sense_info) > 0:
                    sense_gloss_pairs += list(sense_info.items())

                random.shuffle(sense_gloss_pairs)
                sense_keys, glosses = zip(*sense_gloss_pairs)
            else:
                sense_keys, glosses = zip(*sense_info.items())

            targets = [sense_keys.index(k) for k in _gold_keys]
            csv_writer.writerow([_id, _sentence, _pos_list, list(sense_keys), list(glosses), list((lemma, pos)), targets])

            record_count += 1
            gloss_count += len(glosses)
            max_gloss_count = max(max_gloss_count, len(glosses))

        with open(txt_path, 'r', encoding='utf-8') as g:
            for doc in tqdm(xml_root):
                for sent in doc:
                    tokens = []
                    instances = []
                    # Create list of pos tokens
                    pos_list = get_pos_tokens(sent)
                    # extract tokens and instances (tokens tagged with lemma and POS)
                    for token in sent:
                        tokens.append(token.text)
                        if token.tag == 'instance':
                            start_idx = len(tokens) - 1 # the number of tokens before target/ the position of token before the target
                            end_idx = start_idx + 1 # the position of the target
                            instances.append((
                                token.attrib['id'],
                                start_idx,
                                end_idx,
                                token.attrib['lemma'],
                                token.attrib['pos'],
                                pos_list)
                            )

                    # construct records for gloss selection task
                    for id_, start, end, lemma, pos, pos_list in instances:
                        gold = g.readline().strip().split() # Note: g is the gold keys .txt file. Gold is formatted as id; key
                        gold_keys = gold[1:]
                        assert id_ == gold[0]

                        sentence = " ".join(
                            tokens[:start] + [TGT_TOKEN] + tokens[start:end] + [TGT_TOKEN] + tokens[end:]
                        )

                        sentence_no_tgt =

                        pos_list_with_tgt = pos_list[:start] + [special_token_pos] + pos_list[start:end] + [special_token_pos] + pos_list[end:]
                        _write_to_csv(id_, sentence, lemma, pos, gold_keys)

        if args.use_augmentation:
            print("Creating additional training data using example sentences from WordNet...")
            counter = 0
            for pos, lemma_name_generator in get_all_wordnet_lemma_names():
                print(f"Processing {pos}...")
                for lemma in tqdm(list(lemma_name_generator)):
                    for gold_key, examples in get_example_sentences(lemma, pos).items():
                        for example_sentence in examples:
                            re_result = re.search(rf"\b{lemma.lower()}\b", example_sentence.lower())
                            if re_result is not None:
                                start, end = re_result.span()
                                sentence = f"{example_sentence[:start]}" \
                                    f"{TGT_TOKEN} {example_sentence[start:end]} {TGT_TOKEN}" \
                                    f"{example_sentence[end:]}".strip()
                                _write_to_csv(f"wn-aug-{counter}", sentence, lemma, pos, [gold_key])
                                counter += 1

    print(
        f"Done.\n"
        f"Number of records: {record_count}\n"
        f"Average number of glosses per record: {gloss_count / record_count:.2f}\n"
        f"Maximum number of glosses in one record: {max_gloss_count}"
    )


if __name__ == '__main__':
    main()
