import csv
import os
from collections import namedtuple

import nltk
import regex as re
import torch
from tqdm import tqdm

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import wordnet as wn

BertInputPosDep = namedtuple("BertInputPosDep",
                             ["input_ids", "input_mask", "segment_ids", "pos_ids", "dep_ids", "label_id"])
BertInputPos = namedtuple("BertInputPos",
                          ["input_ids", "input_mask", "segment_ids", "pos_ids", "label_id"])
BertInputDep = namedtuple("BertInputDep",
                          ["input_ids", "input_mask", "segment_ids", "dep_ids", "label_id"])
BertInput = namedtuple("BertInput",
                       ["input_ids", "input_mask", "segment_ids", "label_id"])

"""
This script is made from dataset_syntax.py.

The goal is to make multiple cached datasets in the same run. The reasoning is that SpaCy is one of the heavier parts.
Thus, if the SpaCy analysis is only done once instead of multiple times, it might save some time in the long run.
Also, the logic has been simplified, so that [CLS] is always put in front and padding is always done with 0's.
"""


def load_dataset(args, csv_path, tokenizer, max_sequence_length, spacy_model=None):
    # Define record structure and how to deserialize it
    GlossSelectionRecord = namedtuple("GlossSelectionRecord",
                                      ["guid", "sentence", "sense_keys", "glosses", "targets"])

    def deserialize_csv_record(row):
        return GlossSelectionRecord(row[0], row[1], eval(row[2]), eval(row[3]), [int(t) for t in eval(row[4])])

    _load_and_cache_dataset(
        args,
        csv_path,
        tokenizer,
        max_sequence_length,
        deserialize_csv_record,
        spacy_model
    )


def _load_and_cache_dataset(args, csv_path, tokenizer, max_sequence_length, deserialze_fn, spacy_model):
    # ====== Create file names for cached data ======
    data_dir = os.path.dirname(csv_path)
    dataset_name = os.path.basename(csv_path).split('.')[0]
    cached_features_file = os.path.join(data_dir, f"cached_{dataset_name}-{max_sequence_length}")
    if re.search("bert-base", args.model_name_or_path):
        cached_features_file = cached_features_file + "-bert-base"
    if re.search("bert-large", args.model_name_or_path):
        cached_features_file = cached_features_file + "-bert-large"
    if re.search("-cased", args.model_name_or_path):
        cached_features_file = cached_features_file + "-cased"
    if re.search("-uncased", args.model_name_or_path):
        cached_features_file = cached_features_file + "-uncased"
    if re.search("-whole-word-masking", args.model_name_or_path):
        cached_features_file = cached_features_file + "-wh_w_m"


    cached_features_file_pd_ge = cached_features_file + "-pos"
    cached_features_file_pd_ge = cached_features_file_pd_ge + "-dep"
    cached_features_file_pd_ge = cached_features_file_pd_ge + "-glosses_extended_w_tgt-no_syntax_for_special"
    if os.path.exists(cached_features_file_pd_ge) and re.search("pd", args.to_cache_with_gloss_extensions):
        raise ValueError(f"{cached_features_file_pd_ge} already exists, stopping script to avoid overwriting")

    cached_features_file_p_ge = cached_features_file + "-pos"
    cached_features_file_p_ge = cached_features_file_p_ge + "-glosses_extended_w_tgt-no_syntax_for_special"
    if os.path.exists(cached_features_file_p_ge) and re.search("pos", args.to_cache_with_gloss_extensions):
        raise ValueError(f"{cached_features_file_p_ge} already exists, stopping script to avoid overwriting")

    cached_features_file_d_ge = cached_features_file + "-dep"
    cached_features_file_d_ge = cached_features_file_d_ge + "-glosses_extended_w_tgt-no_syntax_for_special"
    if os.path.exists(cached_features_file_d_ge) and re.search("dep", args.to_cache_with_gloss_extensions):
        raise ValueError(f"{cached_features_file_d_ge} already exists, stopping script to avoid overwriting")

    cached_features_file_pd = cached_features_file + "-pos"
    cached_features_file_pd = cached_features_file_pd + "-dep"
    cached_features_file_pd = cached_features_file_pd + "-no_syntax_for_special"
    if os.path.exists(cached_features_file_pd) and re.search("pd", args.to_cache_wo_gloss_extensions):
        raise ValueError(f"{cached_features_file_pd} already exists, stopping script to avoid overwriting")

    cached_features_file_p = cached_features_file + "-pos"
    cached_features_file_p = cached_features_file_p + "-no_syntax_for_special"
    if os.path.exists(cached_features_file_p) and re.search("pos", args.to_cache_wo_gloss_extensions):
        raise ValueError(f"{cached_features_file_p} already exists, stopping script to avoid overwriting")

    cached_features_file_d = cached_features_file + "-dep"
    cached_features_file_d = cached_features_file_d + "-no_syntax_for_special"
    if os.path.exists(cached_features_file) and re.search("dep", args.to_cache_wo_gloss_extensions):
        raise ValueError(f"{cached_features_file_d} already exists, stopping script to avoid overwriting")

    # ====== Create cached data ======
    print(f"Creating features to cache from dataset {csv_path}")
    records = _create_records_from_csv(csv_path, deserialze_fn)

    features_pd, features_p, features_d, features_pd_ge, features_p_ge, features_d_ge = _create_features_from_records(
        args, spacy_model, records, max_sequence_length, tokenizer,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=1,
        pad_token_segment_id=0)

    # ====== Save cached data ======
    if features_pd[-1]:
        print("Saving features into cached file %s", cached_features_file_pd)
        torch.save(features_pd, cached_features_file_pd)

    if features_p[-1]:
        print("Saving features into cached file %s", cached_features_file_p)
        torch.save(features_p, cached_features_file_p)

    if features_d[-1]:
        print("Saving features into cached file %s", cached_features_file_d)
        torch.save(features_d, cached_features_file_d)

    if features_pd_ge[-1]:
        print("Saving features into cached file %s", cached_features_file_pd_ge)
        torch.save(features_pd_ge, cached_features_file_pd_ge)

    if features_p_ge[-1]:
        print("Saving features into cached file %s", cached_features_file_p_ge)
        torch.save(features_p_ge, cached_features_file_p_ge)

    if features_d_ge[-1]:
        print("Saving features into cached file %s", cached_features_file_d_ge)
        torch.save(features_d_ge, cached_features_file_d_ge)


# Read in dataset and deserialize each row into a named tuple.
def _create_records_from_csv(csv_path, deserialize_fn):
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # read off header

        return [deserialize_fn(row) for row in reader]


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _get_gloss_extension(key, conversion_dict):
    synset = wn.lemma_from_key(key).synset()

    # Find the lemmas, of the synset. The correct lemma can be found in the key name.
    lemma = [lemma.name() for lemma in synset.lemmas() if re.search(lemma.name(), key)]
    if len(lemma) == 0:
        # find the word
        lemma = re.findall(r"([^%]+)", key)[0]
    # If there still multiple lemmas, go with the first. Could be improved upon later.
    lemma = re.sub("_", " ", lemma[0])
    pos = str(synset.pos())
    if conversion_dict != None:
        pos = conversion_dict[pos]
    out_tuple = (lemma, pos, "[PAD]")
    return out_tuple


# Creates features, i.e., the input given to BERT, to train on.
def _create_features_from_records(args, spacy_model, records, max_seq_length, tokenizer,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  disable_progress_bar=False, ):
    """ Convert records to list of features. Each feature is a list of sub-features where the first element is
        always the feature created from context-gloss pair while the rest of the elements are features created from
        context-example pairs (if available)
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    spacy_model = spacy_model

    spacy_tgt_tpl = ('[TGT]', '[PAD]', "[PAD]", ['[TGT]'])

    # Used for creating pos tag for gloss extenstion
    conv_dict = {
        "n": "[NOUN]",
        "v": "[VERB]",
        "a": "[ADJ]",
        # Satellite adjective, adjective for my purposes
        "s": "[ADJ]",
        "r": "[ADV]"
    }

    # ====== Create list of features for each dataset ======
    """
    p for "pos tags"
    d for "grammatical dependencies"
    pd for both of the above
    ge for "(with) gloss extensions" 
    """
    features_p = []
    features_d = []
    features_pd = []
    features_p_ge = []
    features_d_ge = []
    features_pd_ge = []

    for record in tqdm(records, disable=disable_progress_bar):
        # ====== Create tokens for BERT from the sentence ======
        """
            - First take care of the case where syntax info is used
            - Second take care of the case where it isn't

        """
        """
            - Analyze the sentence with and without [TGT] tokens using SpaCy
            - Check if the two analyses are equal. If not, use the one without the [TGT] tokens
                - Ensures the correct SpaCy analysis is used & eliminates the need to reinsert the [TGT] tokens
            - Create the context tokens of the context-gloss pair.
        """
        # ====== Analyze the sentence ======
        sentence = record.sentence
        # For sentences like "Kids were [TGT] break [TGT]-dancing at the street corner":
        #   - Make sure to insert a space before/after the target tokens
        sentence = re.sub(r"(\[TGT\])([^ ])", r"\1 \2", sentence)
        sentence = re.sub(r"([^ ])(\[TGT\])", r"\1 \2", sentence)

        # Create sentence without [TGT] tokens
        no_tgt_sent = re.sub(r"\[TGT\]", "", sentence).strip()
        no_tgt_sent = re.sub(r"\s{2,}", " ", no_tgt_sent)

        # Analyze sentence with and without [TGT] tokens
        spacy_doc = spacy_model(no_tgt_sent)
        spacy_tokens = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text)) for token in spacy_doc]
        tgt_doc = spacy_model(sentence)
        tgt_tokens = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text)) for token in tgt_doc]

        # Check if the Spacy analysis is the same with and without [TGT] tokens
        tgt_tokens2 = [ele for ele in tgt_tokens if ele[0] != "[TGT]"]
        if tgt_tokens2 == spacy_tokens:
            # If they're the same, simply add the predefined, correct tokens for [TGT]
            complete_tokens = [ele if ele[0] != "[TGT]" else spacy_tgt_tpl for ele in tgt_tokens]
        else:
            # Find the indexes of the [TGT] words and add the predefined [TGT] tokens add those indexes
            tgt_index = [i for i, tokens in enumerate(tgt_tokens) if tokens[0] == "[TGT]"]
            complete_tokens = spacy_tokens
            complete_tokens.insert(tgt_index[0], spacy_tgt_tpl)
            complete_tokens.insert(tgt_index[1], spacy_tgt_tpl)

        # ====== Create the context tokens for the context-gloss pairs for BERT ======
        tokens_a = [token for sublist in complete_tokens for token in sublist[3]]
        bert_pos_tokens_a = [[ele[1]] * len(ele[-1]) for ele in complete_tokens]
        bert_pos_tokens_a = [pos if re.search(r"^\[.+\]$", pos) else "[" + pos + "]"
                             for sublist in bert_pos_tokens_a for pos in sublist]
        bert_dep_tokens_a = [[ele[2]] * len(ele[-1]) for ele in complete_tokens]
        bert_dep_tokens_a = [dep if re.search(r"^\[.+\]$", dep) else "[" + dep + "]"
                             for sublist in bert_dep_tokens_a for dep in sublist]
        assert (len(tokens_a) == len(bert_pos_tokens_a) == len(bert_dep_tokens_a))

        # ====== Create tokens for BERT from the gloss ======
        # Make empty lists for context-gloss pairs
        pairs_p = []
        pairs_d = []
        pairs_pd = []
        pairs_p_ge = []
        pairs_d_ge = []
        pairs_pd_ge = []

        # Sequences to loop through
        sequences = [(gloss, 1 if i in record.targets else 0, record.sense_keys[i])
                     for i, gloss in enumerate(record.glosses)]

        # Get the gloss from the record; 1 if i is in the record targets or 0 if i is not in the record targets.
        # Loop through sequences
        for seq, label, key in sequences:  # seq is the gloss, label is either 0 or 1
            # ====== Analyze sentence using spacy ======
            spacy_doc = spacy_model(seq)
            spacy_tokens_complete = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text))
                                     for token in spacy_doc]

            # ====== Create the gloss tokens to feed to BERT ======
            tokens_b = [word for sublist in spacy_tokens_complete for word in sublist[3]]
            bert_pos_tokens_b = [[ele[1]] * len(ele[3]) for ele in spacy_tokens_complete]
            bert_pos_tokens_b = [pos if re.search(r"^\[.+\]$", pos) else "[" + pos + "]"
                                 for sublist in bert_pos_tokens_b for pos in sublist]
            assert (len(tokens_b) == len(bert_pos_tokens_b))
            # if args.use_dependencies:
            bert_dep_tokens_b = [[ele[2]] * len(ele[3]) for ele in spacy_tokens_complete]
            bert_dep_tokens_b = [dep if re.search(r"^\[.+\]$", dep) else "[" + dep + "]"
                                 for sublist in bert_dep_tokens_b for dep in sublist]
            assert (len(tokens_b) == len(bert_dep_tokens_b))

            # ====== From here on, the code is split into with/without gloss extensions ======
            # ====== Without gloss extensions ======
            if args.to_cache_wo_gloss_extensions != "":
                # ====== Truncate tokens ======
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                _truncate_seq_pair(bert_pos_tokens_a, bert_pos_tokens_b, max_seq_length - 3)
                _truncate_seq_pair(bert_dep_tokens_a, bert_dep_tokens_b, max_seq_length - 3)

                # ====== Merge the context and gloss tokens to form the context gloss pair ======
                tokens = tokens_a + [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)
                pos_tokens = bert_pos_tokens_a + ["[PAD]"]
                dep_tokens = bert_dep_tokens_a + ["[PAD]"]

                tokens += tokens_b + [sep_token]
                # +1 to account for [SEP] token
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
                pos_tokens += bert_pos_tokens_b + ["[PAD]"]
                dep_tokens += bert_dep_tokens_b + ["[PAD]"]

                # ====== Add [CLS] and [PAD] tokens ======
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids
                pos_tokens = ["[PAD]"] + pos_tokens
                dep_tokens = ["[PAD]"] + dep_tokens

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                pos_ids = tokenizer.convert_syntax_tokens_to_ids(pos_tokens, "pos")
                dep_ids = tokenizer.convert_syntax_tokens_to_ids(dep_tokens, "dep")

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)

                input_ids = input_ids + ([pad_token] * padding_length)  # [pad_token] is 0, the [PAD] token's id
                input_mask = input_mask + ([0] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                pos_ids = pos_ids + ([pad_token] * padding_length)
                dep_ids = dep_ids + ([pad_token] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(pos_ids) == max_seq_length
                assert len(dep_ids) == max_seq_length

                # ====== Append pairs tokens to pairs ======
                if re.search("pd", args.to_cache_wo_gloss_extensions):
                    pairs_pd.append(
                        BertInputPosDep(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                        pos_ids=pos_ids, dep_ids=dep_ids,
                                        label_id=label)
                    )
                if re.search("pos", args.to_cache_wo_gloss_extensions):
                    pairs_p.append(
                        BertInputPos(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                     pos_ids=pos_ids, label_id=label)
                    )
                if re.search("dep", args.to_cache_wo_gloss_extensions):
                    pairs_d.append(
                        BertInputDep(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                     dep_ids=dep_ids, label_id=label)
                    )

            # ====== With gloss extensions ======
            if args.to_cache_with_gloss_extensions != "":
                # Make token lists for gloss extensions
                tokens_a_ge = tokens_a
                bert_pos_tokens_a_ge = bert_pos_tokens_a
                bert_dep_tokens_a_ge = bert_dep_tokens_a
                tokens_b_ge = tokens_b
                bert_pos_tokens_b_ge = bert_pos_tokens_b
                bert_dep_tokens_b_ge = bert_dep_tokens_b

                # ====== Create gloss extension and then truncate =======
                gloss_extension_tuple = _get_gloss_extension(key, conv_dict)
                gloss_ex_tokens = tokenizer.tokenize(gloss_extension_tuple[0])
                num_gloss_ex = len(gloss_ex_tokens)
                gloss_ex_pos_tokens = [gloss_extension_tuple[1]] * num_gloss_ex
                gloss_ex_dep_tokens = [gloss_extension_tuple[2]] * num_gloss_ex
                # add target tokens to gloss extension tokens, then add appropriate syntax tokens.
                gloss_ex_tokens.insert(0, "[TGT]")
                gloss_ex_tokens.append("[TGT]")
                gloss_ex_pos_tokens.insert(0, spacy_tgt_tpl[1])
                gloss_ex_pos_tokens.append(spacy_tgt_tpl[1])
                gloss_ex_dep_tokens.insert(0, spacy_tgt_tpl[2])
                gloss_ex_dep_tokens.append(spacy_tgt_tpl[2])
                num_gloss_ex = len(gloss_ex_tokens)

                max_seq_len_with_gloss_ex = max_seq_length - num_gloss_ex - 3

                _truncate_seq_pair(tokens_a_ge, tokens_b_ge, max_seq_len_with_gloss_ex)
                _truncate_seq_pair(bert_pos_tokens_a_ge, bert_pos_tokens_b_ge, max_seq_len_with_gloss_ex)
                _truncate_seq_pair(bert_dep_tokens_a_ge, bert_dep_tokens_b_ge, max_seq_len_with_gloss_ex)

                # ====== Merge the context and gloss tokens to form the context gloss pair ======
                tokens_ge = tokens_a_ge + [sep_token]
                segment_ids_ge = [sequence_a_segment_id] * len(tokens_ge)
                pos_tokens_ge = bert_pos_tokens_a + ["[PAD]"]
                dep_tokens_ge = bert_dep_tokens_a + ["[PAD]"]

                tokens_ge += gloss_ex_tokens
                pos_tokens_ge += gloss_ex_pos_tokens
                dep_tokens_ge += gloss_ex_dep_tokens

                tokens_ge += tokens_b_ge + [sep_token]
                # +1 to account for [SEP] token
                segment_ids_ge += [sequence_b_segment_id] * (len(tokens_b_ge) + 1 + num_gloss_ex)
                pos_tokens_ge += bert_pos_tokens_b_ge + ["[PAD]"]
                dep_tokens_ge += bert_dep_tokens_b_ge + ["[PAD]"]

                tokens_ge = [cls_token] + tokens_ge
                segment_ids_ge = [cls_token_segment_id] + segment_ids_ge
                pos_tokens_ge = ["[PAD]"] + pos_tokens_ge
                dep_tokens_ge = ["[PAD]"] + dep_tokens_ge

                input_ids_ge = tokenizer.convert_tokens_to_ids(tokens_ge)
                pos_ids_ge = tokenizer.convert_syntax_tokens_to_ids(pos_tokens_ge, "pos")
                dep_ids_ge = tokenizer.convert_syntax_tokens_to_ids(dep_tokens_ge, "dep")

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask_ge = [1] * len(input_ids_ge)

                # Zero-pad up to the sequence length.
                padding_length_ge = max_seq_length - len(input_ids_ge)

                input_ids_ge = input_ids_ge + (
                        [pad_token] * padding_length_ge)  # [pad_token] is 0, the [PAD] token's id
                input_mask_ge = input_mask_ge + ([0] * padding_length_ge)
                segment_ids_ge = segment_ids_ge + ([pad_token_segment_id] * padding_length_ge)
                pos_ids_ge = pos_ids_ge + ([pad_token] * padding_length_ge)
                dep_ids_ge = dep_ids_ge + ([pad_token] * padding_length_ge)

                assert len(input_ids_ge) == max_seq_length
                assert len(input_mask_ge) == max_seq_length
                assert len(segment_ids_ge) == max_seq_length
                assert len(pos_ids_ge) == max_seq_length
                assert len(dep_ids_ge) == max_seq_length

                # ====== Append pairs tokens to pairs ======
                if re.search("pd", args.to_cache_with_gloss_extensions):
                    pairs_pd_ge.append(
                        BertInputPosDep(input_ids=input_ids_ge, input_mask=input_mask_ge, segment_ids=segment_ids_ge,
                                        pos_ids=pos_ids_ge, dep_ids=dep_ids_ge,
                                        label_id=label)
                    )
                if re.search("pos", args.to_cache_with_gloss_extensions):
                    pairs_p_ge.append(
                        BertInputPos(input_ids=input_ids_ge, input_mask=input_mask_ge, segment_ids=segment_ids_ge,
                                     pos_ids=pos_ids_ge, label_id=label)
                    )
                if re.search("dep", args.to_cache_with_gloss_extensions):
                    pairs_d_ge.append(
                        BertInputDep(input_ids=input_ids_ge, input_mask=input_mask_ge, segment_ids=segment_ids_ge,
                                     dep_ids=dep_ids_ge, label_id=label)
                    )

    features_pd.append(pairs_pd)
    features_p.append(pairs_p)
    features_d.append(pairs_d)
    features_pd_ge.append(pairs_pd_ge)
    features_p_ge.append(pairs_p_ge)
    features_d_ge.append(pairs_d_ge)

    w_ge = [ft for ft in [features_pd_ge, features_p_ge, features_d_ge] if len(ft[0]) > 0]
    if len(w_ge) > 0:
        print(w_ge[0], "\n")
        print(tokens_ge, "\n")
        print(pos_tokens_ge, "\n")
        print(dep_tokens_ge, "\n")
    else:
        wo_ge = [ft for ft in [features_pd, features_p, features_d] if len(ft[0]) > 0]
        print("\n", wo_ge[0], "\n")
        print(tokens, "\n")
        print(pos_tokens, "\n")
        print(dep_tokens, "\n")


    return features_pd, features_p, features_d, features_pd_ge, features_p_ge, features_d_ge
