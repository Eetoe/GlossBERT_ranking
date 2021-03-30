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

from utils.wordnet_syntax import get_example_sentences, get_glosses, get_all_wordnet_lemma_names, _get_gloss_extension
from utils.dataset_syntax import get_pos_tokens


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

    #parser.add_argument(
    #    "--spacy_model",
    #    default="en_core_web_sm",
    #    help="Which spacy model to use. The base model is en_core_web_sm; the transformer model is en_core_web_trf."
    #)

    parser.add_argument(
        "--use_pos",
        action='store_true',
        help="Whether to add POS to the data."
    )

    parser.add_argument(
        "--use_dependencies",
        action='store_true',
        help="Whether to add dependencies to the data."
    )

    parser.add_argument(
        "--cross_pos_train",
        action='store_true',
        help="Whether to have candidate senses of all pos"
    )

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
    if args.use_pos:
        output_filename += "-POS"
    if args.use_dependencies:
        output_filename += "-DEP"
    csv_path = str(Path(args.output_dir).joinpath(f"{output_filename}.csv"))

    print("Creating data for gloss selection task...")
    record_count = 0
    gloss_count = 0
    max_gloss_count = 0

    # Make appropriate header based on args
    if args.use_pos or args.use_dependencies:
        HEADERS = ['id', 'sentence', 'sense_keys', 'glosses', 'gloss_extensions', 'targets']
    else:
        HEADERS = ['id', 'sentence', 'sense_keys', 'glosses', 'targets']


    # Make a dict to convert wn pos of synsets to the pos used elsewhere
    conv_dict = {
        "n": "[NOUN]",
        "v": "[VERB]",
        "a": "[ADJ]",
        # Satellite adjective, adjective for my purposes
        "s": "[ADJ]",
        "r": "[ADV]"
    }

    xml_root = ElementTree(file=xml_path).getroot()
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(HEADERS)

        def _write_to_csv(_id, _sentence, _lemma, _pos, _gold_keys):
            nonlocal record_count, gloss_count, max_gloss_count

            # Check if lemma can be of other pos. This distinguishes between permit (n, v) and permitted (v)
            if args.cross_pos_train:
                noun_dict = get_glosses(_lemma, "NOUN")
                verb_dict = get_glosses(_lemma, "VERB")
                adj_dict = get_glosses(_lemma, "ADJ")
                adv_dict = get_glosses(_lemma, "ADV")
                sense_info = {**noun_dict, **verb_dict, **adj_dict, **adv_dict}
            else:
                sense_info = get_glosses(_lemma, _pos)
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

            if args.use_dependencies or args.use_pos:
                gloss_extensions = [_get_gloss_extension(key, _lemma, conv_dict, args) for key in sense_keys]
                csv_writer.writerow([_id, _sentence, list(sense_keys), list(glosses), gloss_extensions, targets])
            else:
                csv_writer.writerow([_id, _sentence, list(sense_keys), list(glosses), targets])


            record_count += 1
            gloss_count += len(glosses)
            max_gloss_count = max(max_gloss_count, len(glosses))

        with open(txt_path, 'r', encoding='utf-8') as g:
            for doc in tqdm(xml_root):
                for sent in doc:
                    tokens = []
                    instances = []
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
                                token.attrib['pos']
                            )
                            )

                    # construct records for gloss selection task
                    for id_, start, end, lemma, pos in instances:
                        gold = g.readline().strip().split()  # Note: g is the gold keys .txt file. Gold is formatted as id; key
                        gold_keys = gold[1:]
                        assert id_ == gold[0]

                        sentence = " ".join(
                            tokens[:start] + [TGT_TOKEN] + tokens[start:end] + [TGT_TOKEN] + tokens[end:]
                        )

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
