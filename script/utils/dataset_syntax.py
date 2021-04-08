import csv
import os
from collections import namedtuple

import torch
from tqdm import tqdm
import spacy
import regex as re


"""
load_dataset(): loads the dataset and prepares it for BERT
 - _load_and_cache_dataset(): this function is the one that creates the dataset for load_dataset()
    - _create_records_from_csv(): is used within _load_and_cache_dataset() to deserialize the dataset into records, i.e.
                                  single rows, which can be used to create the set of candidate senses for the target.
    - _create_features_from_record(): is used on the records to create the features that BERT actually trains on
        - _truncate_seq_pair: used within _create_features_from_records() to truncate the context-gloss pairs.

collate_batch functions: used to split the datset into batches once it has been loaded. There are four, which are used 
                         depending on the use of pos and dep.
 
write_predictions(): used to write the model's predictions during evaluation to csv files, which can then be used to
                     calculate the model's score.
"""


def load_dataset(args, csv_path, tokenizer, max_sequence_length, spacy_model):
    # Change deserialization function depending
    if args.use_gloss_extension:
        GlossSelectionRecord = namedtuple("GlossSelectionRecord",
                                          ["guid", "sentence", "sense_keys",
                                           "glosses", "gloss_extensions",
                                           "targets"])
        def deserialize_csv_record(row):
            return GlossSelectionRecord(row[0], row[1], eval(row[2]), eval(row[3]), eval(row[4]), [int(t) for t in eval(row[5])])
    else:
        GlossSelectionRecord = namedtuple("GlossSelectionRecord",
                                          ["guid", "sentence", "sense_keys", "glosses", "targets"])
        def deserialize_csv_record(row):
            return GlossSelectionRecord(row[0], row[1], eval(row[2]), eval(row[3]), [int(t) for t in eval(row[4])])

    # The following part(s) is unmodified
    return _load_and_cache_dataset(
        args,
        csv_path,
        tokenizer,
        max_sequence_length,
        deserialize_csv_record,
        spacy_model
    )


def _load_and_cache_dataset(args, csv_path, tokenizer, max_sequence_length, deserialze_fn, spacy_model):
    # Load data features from cache or dataset file
    data_dir = os.path.dirname(csv_path)
    dataset_name = os.path.basename(csv_path).split('.')[0]
    cached_features_file = os.path.join(data_dir, f"cached_{dataset_name}_{max_sequence_length}")
    if os.path.exists(cached_features_file):
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        print(f"Creating features from dataset {csv_path}")
        records = _create_records_from_csv(csv_path, deserialze_fn)

        features = _create_features_from_records(args, spacy_model, records, max_sequence_length, tokenizer,
                                                 cls_token=tokenizer.cls_token,
                                                 sep_token=tokenizer.sep_token,
                                                 cls_token_segment_id=1,
                                                 pad_token_segment_id=0)

        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    class FeatureDataset(torch.utils.data.Dataset):  # Map style dataset
        def __init__(self, features_):
            self.features = features_

        def __getitem__(self, index):
            return self.features[index]

        def __len__(self):
            return len(self.features)

    return FeatureDataset(features)


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

# Creates features, i.e., the input given to BERT, to train on.
def _create_features_from_records(args, spacy_model, records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False,):
    """ Convert records to list of features. Each feature is a list of sub-features where the first element is
        always the feature created from context-gloss pair while the rest of the elements are features created from
        context-example pairs (if available)

        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    if args.use_pos or args.use_dependencies:
        spacy_model = spacy_model

    # Set up BERT input, might be smarter to do in the different if statements
    # yes: simpler code to read
    # No: only needs to be done once, not at every step in loop
    if args.use_pos and args.use_dependencies:
        BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "pos_ids", "dep_ids", "label_id"])
    elif args.use_pos and not args.use_dependencies:
        BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "pos_ids", "label_id"])
    elif not args.use_pos and args.use_dependencies:
        BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "dep_ids", "label_id"])
    else:
        BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"])

    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        # ====== Create tokens for BERT from the sentence ======
        """
            - First take care of the case where syntax info is used
            - Second take care of the case where it isn't
        
        """
        if args.use_pos or args.use_dependencies:
            """
            Spacy is analyzing the sentence without the [TGT] tokens. To do this, the following is needed:
             - the target word is found
             - then the sentence is checked for multiple instances of the target word
                - Example of multiple instances: "Did he [TGT] plant [TGT] the plant yet ?"
             - If there are multiple instances of the target word, keep track of which one is the target
             - Remove [TGT] tokens for spacy
             - Analyse sentence without [TGT] tokens using spacy
             - Split the sentence into three parts:
                - the tokens before the target
                - the target
                - the tokens after the target
             - insert the [TGT] tokens before and after the target
             - create tokens for the sentence
            """
            # ====== Find target ======
            sentence = record.sentence
            dissect_sent = re.findall("(.*)(\[TGT\]\s)(\w+)(\s\[TGT\])(.*)", sentence)
            # The reason why the indexing below works is that the (.*) still creates an empty element "".
            target_word = dissect_sent[0][2]
            # Find the instance of the target word that is the target
            instances_of_target_word = re.findall(f"({target_word})(\s\[TGT\])*", sentence)
            # print(instances_of_target_word)
            for i, instance in enumerate(instances_of_target_word):
                if instance[1] == " [TGT]":
                    instance_that_is_target = i

            # ====== Remove [TGT] tokens ======
            no_tgt_sent = re.sub(r"\[TGT\]", "", sentence).strip()
            no_tgt_sent = re.sub(r"\s{2,}", " ", no_tgt_sent)

            # ====== Analyze sentence using spacy ======
            spacy_doc = spacy_model(no_tgt_sent)
            spacy_tokens = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text)) for token in spacy_doc]

            # ====== Add the [TGT] tokens back in ======
            # Make [TGT]'s token entries (as in text, POS, dependencies, tokenized form)
            if args.zero_syntax_for_special_tokens:
                spacy_tgt_list = [('[TGT]', '[PAD]', "[PAD]", ['[TGT]'])]
            else:
                spacy_tgt_list = [('[TGT]', '[TGT_POS]', "[TGT_DEP]", ['[TGT]'])]

            # Add the [TGT] tokens to the sentence again
            # get index of all instances of the target word. Non-target words have index None
            tgt_indexes = [i if tokens[0] == target_word else None for i, tokens in enumerate(spacy_tokens)]
            # Remove all None from the tgt indexes to be left only with the indexes of the possible targets
            tgt_indexes = [ele for ele in tgt_indexes if ele != None]
            # Keep only the index of the instance of the target word that is actually the target
            tgt_index = [index for num, index in enumerate(tgt_indexes) if num == instance_that_is_target]

            # Split the sentence into three parts:
            #  - The part before the target
            #  - The target
            #  - The part after the target
            # Before the target token
            if tgt_index[0] == 0:  # If the target is the first token of the sentence
                tokens_before_tgt = []
            elif tgt_index == len(spacy_tokens) - 1:  # if the target is the last word of the sentence
                tokens_before_tgt = spacy_tokens[0:-1]
            else:
                tokens_before_tgt = spacy_tokens[0:tgt_index[0]]

            # The target token
            tgt_word_token = [spacy_tokens[tgt_index[0]]]

            # After the target token
            if tgt_index[0] == 0:  # if the target is the first word
                tokens_after_tgt = spacy_tokens[1:]
            elif tgt_index[0] == len(spacy_tokens) - 1:  # if the target is the last word of the sentence
                tokens_after_tgt = []
            else:
                tokens_after_tgt = spacy_tokens[tgt_index[0]:]

            spacy_tokens_complete = tokens_before_tgt +\
                                    spacy_tgt_list + tgt_word_token + spacy_tgt_list +\
                                    tokens_after_tgt

            # ====== Create the tokens to feed to BERT ======
            tokens_a = [word for sublist in spacy_tokens_complete for word in sublist[3]]

            if args.use_pos:
                bert_pos_tokens_a = [[ele[1]] * len(ele[3]) for ele in spacy_tokens_complete]
                bert_pos_tokens_a = [pos if re.search(r"^\[.+\]$", pos) else "[" + pos + "]"
                                     for sublist in bert_pos_tokens_a for pos in sublist]
                assert (len(tokens_a) == len(bert_pos_tokens_a))
            if args.use_dependencies:
                bert_dep_tokens_a = [[ele[2]] * len(ele[3]) for ele in spacy_tokens_complete]
                bert_dep_tokens_a = [dep if re.search(r"^\[.+\]$", dep) else "[" + dep + "]"
                                     for sublist in bert_dep_tokens_a for dep in sublist]
                assert (len(tokens_a) == len(bert_dep_tokens_a))

        else:
            tokens_a = tokenizer.tokenize(record.sentence)



        # ====== Create tokens for BERT from the gloss ======
        """
            - First create the case where gloss extension is used. This requires POS by default.
                - both pos and dep are found, only relevant information is passed on.
            - Create the case where gloss extension isn't used.
                - Using syntax info. Both pos and dep found, only relevant passed on.
                - Without syntax, i.e. the original script.
        
        """
        # in tqdm: get the gloss from the record and a 1 if i is in the record targets or 0 if i is not in the record targets.
        # It only makes sense to use the gloss extension when POS is used, as dep is not guaranteed for any sense
        if args.use_gloss_extension:
            sequences = [(gloss, 1 if i in record.targets else 0, record.gloss_extensions[i])
                         for i, gloss in enumerate(record.glosses)]
        else:
            sequences = [(gloss, 1 if i in record.targets else 0) for i, gloss in enumerate(record.glosses)]

        # Make empty list for context-gloss pairs
        pairs = []

        # ====== Using gloss extension ======
        if args.use_gloss_extension:
            # Since gloss extension is only used alongside pos, the spacy analysis is always done here.
            for seq, label, gloss_extension in sequences:  # seq is the gloss, label is either 0 or 1
                # ====== Analyze sentence using spacy ======
                spacy_doc = spacy_model(seq)
                spacy_tokens_complete = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text))
                                for token in spacy_doc]

                # ====== Create the tokens to feed to BERT ======
                tokens_b = [word for sublist in spacy_tokens_complete for word in sublist[3]]

                if args.use_pos:
                    bert_pos_tokens_b = [[ele[1]] * len(ele[3]) for ele in spacy_tokens_complete]
                    bert_pos_tokens_b = [pos if re.search(r"^\[.+\]$", pos) else "[" + pos + "]"
                                         for sublist in bert_pos_tokens_b for pos in sublist]
                    assert (len(tokens_b) == len(bert_pos_tokens_b))
                if args.use_dependencies:
                    bert_dep_tokens_b = [[ele[2]] * len(ele[3]) for ele in spacy_tokens_complete]
                    bert_dep_tokens_b = [dep if re.search(r"^\[.+\]$", dep) else "[" + dep + "]"
                                         for sublist in bert_dep_tokens_b for dep in sublist]
                    assert (len(tokens_b) == len(bert_dep_tokens_b))



                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], gloss extension, [SEP]

                target_word_tokens = tokenizer.tokenize(target_word)
                number_of_target_tokens = len(target_word_tokens)
                target_pos = [gloss_extension[1]]*number_of_target_tokens
                target_dep = ["[PAD]"]*number_of_target_tokens

                # Max sequence length with spots reserved for [CLS], [SEP] and target word tokens
                msl_with_gloss_extension = max_seq_length - 3 - number_of_target_tokens

                # Truncate based on msl_with_gloss_extension
                _truncate_seq_pair(tokens_a, tokens_b, msl_with_gloss_extension)
                _truncate_seq_pair(bert_pos_tokens_a, bert_pos_tokens_b, msl_with_gloss_extension)
                _truncate_seq_pair(bert_dep_tokens_a, bert_dep_tokens_b, msl_with_gloss_extension)

                # Now the tokens of the context and gloss halves are merged together.
                tokens = tokens_a + [sep_token] + target_word_tokens
                segment_ids = [sequence_a_segment_id] * len(tokens)
                if args.zero_syntax_for_special_tokens:
                    pos_tokens = bert_pos_tokens_a + ["[PAD]"] + target_pos
                    dep_tokens = bert_dep_tokens_a + ["[PAD]"] + target_dep
                else:
                    pos_tokens = bert_pos_tokens_a + ["[SEP_POS]"] + target_pos
                    dep_tokens = bert_dep_tokens_a + ["[SEP_DEP]"] + target_dep

                tokens += tokens_b + [sep_token]
                # +1 to account for [SEP] token
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
                if args.zero_syntax_for_special_tokens:
                    pos_tokens += bert_pos_tokens_b + ["[PAD]"]
                    dep_tokens += bert_dep_tokens_b + ["[PAD]"]
                else:
                    pos_tokens += bert_pos_tokens_b + ["[SEP_POS]"]
                    dep_tokens += bert_dep_tokens_b + ["[SEP_DEP]"]

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                    if args.zero_syntax_for_special_tokens:
                        pos_tokens = pos_tokens + ["[PAD]"]
                        dep_tokens = dep_tokens + ["[PAD]"]
                    else:
                        pos_tokens = pos_tokens + ["[CLS_POS]"]
                        dep_tokens = dep_tokens + ["[CLS_DEP]"]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids
                    if args.zero_syntax_for_special_tokens:
                        pos_tokens = ["[PAD]"] + pos_tokens
                        dep_tokens = ["[PAD]"] + dep_tokens
                    else:
                        pos_tokens = ["[CLS_POS]"] + pos_tokens
                        dep_tokens = ["[CLS_DEP]"] + dep_tokens

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                pos_ids = tokenizer.convert_tokens_to_ids(pos_tokens)
                dep_ids = tokenizer.convert_tokens_to_ids(dep_tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                    pos_ids = ([pad_token] * padding_length) + pos_ids
                    dep_ids = ([pad_token] * padding_length) + dep_ids
                else:
                    input_ids = input_ids + \
                                ([pad_token] * padding_length)  # [pad_token] defined as 0, the [PAD] token's id
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                    pos_ids = pos_ids + ([pad_token] * padding_length)
                    dep_ids = dep_ids + ([pad_token] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(pos_ids) == max_seq_length
                assert len(dep_ids) == max_seq_length

                # Append context-pair to the pairs list.
                # This simple if statement works, as gloss extensions require the use of pos
                if args.use_dependencies:
                    pairs.append(
                        BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                  pos_ids=pos_ids, dep_ids=dep_ids,
                                  label_id=label)
                    )
                else:
                    pairs.append(
                        BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                  pos_ids=pos_ids, label_id=label)
                    )

            features.append(pairs)

        # ====== Without gloss extension ======
        else:
            # ====== When using syntax info =======
            if args.use_pos or args.use_dependencies:
                for seq, label in sequences:  # seq is the gloss, label is either 0 or 1
                    # ====== Analyze sentence using spacy ======
                    spacy_doc = spacy_model(seq)
                    spacy_tokens_complete = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text))
                                             for token in spacy_doc]

                    # ====== Create the tokens to feed to BERT ======
                    tokens_b = [word for sublist in spacy_tokens_complete for word in sublist[3]]

                    if args.use_pos:
                        bert_pos_tokens_b = [[ele[1]] * len(ele[3]) for ele in spacy_tokens_complete]
                        bert_pos_tokens_b = [pos if re.search(r"^\[.+\]$", pos) else "[" + pos + "]"
                                             for sublist in bert_pos_tokens_b for pos in sublist]
                        assert (len(tokens_b) == len(bert_pos_tokens_b))
                    if args.use_dependencies:
                        bert_dep_tokens_b = [[ele[2]] * len(ele[3]) for ele in spacy_tokens_complete]
                        bert_dep_tokens_b = [dep if re.search(r"^\[.+\]$", dep) else "[" + dep + "]"
                                             for sublist in bert_dep_tokens_b for dep in sublist]
                        assert (len(tokens_b) == len(bert_dep_tokens_b))

                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], gloss extension, [SEP]
                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
                    _truncate_seq_pair(bert_pos_tokens_a, bert_pos_tokens_b, max_seq_length-3)
                    _truncate_seq_pair(bert_dep_tokens_a, bert_dep_tokens_b, max_seq_length-3)

                    # Now the tokens of the context and gloss halves are merged together.
                    tokens = tokens_a + [sep_token]
                    segment_ids = [sequence_a_segment_id] * len(tokens)
                    if args.zero_syntax_for_special_tokens:
                        pos_tokens = bert_pos_tokens_a + ["[NONE]"]
                        dep_tokens = bert_dep_tokens_a + ["[NONE]"]
                    else:
                        pos_tokens = bert_pos_tokens_a + ["[SEP_POS]"]
                        dep_tokens = bert_dep_tokens_a + ["[SEP_DEP]"]

                    tokens += tokens_b + [sep_token]
                    # +1 to account for [SEP] token
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
                    if args.zero_syntax_for_special_tokens:
                        pos_tokens += bert_pos_tokens_b + ["[NONE]"]
                        dep_tokens += bert_dep_tokens_b + ["[NONE]"]
                    else:
                        pos_tokens += bert_pos_tokens_b + ["[SEP_POS]"]
                        dep_tokens += bert_dep_tokens_b + ["[SEP_DEP]"]

                    if cls_token_at_end:
                        tokens = tokens + [cls_token]
                        segment_ids = segment_ids + [cls_token_segment_id]
                        if args.zero_syntax_for_special_tokens:
                            pos_tokens = pos_tokens + ["[NONE]"]
                            dep_tokens = dep_tokens + ["[NONE]"]
                        else:
                            pos_tokens = pos_tokens + ["[CLS_POS]"]
                            dep_tokens = dep_tokens + ["[CLS_DEP]"]
                    else:
                        tokens = [cls_token] + tokens
                        segment_ids = [cls_token_segment_id] + segment_ids
                        if args.zero_syntax_for_special_tokens:
                            pos_tokens = ["[NONE]"] + pos_tokens
                            dep_tokens = ["[NONE]"] + dep_tokens
                        else:
                            pos_tokens = ["[CLS_POS]"] + pos_tokens
                            dep_tokens = ["[CLS_DEP]"] + dep_tokens

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    pos_ids = tokenizer.convert_tokens_to_ids(pos_tokens)
                    dep_ids = tokenizer.convert_tokens_to_ids(dep_tokens)

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    padding_length = max_seq_length - len(input_ids)
                    if pad_on_left:
                        input_ids = ([pad_token] * padding_length) + input_ids
                        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                        pos_ids = ([pad_token] * padding_length) + pos_ids
                        dep_ids = ([pad_token] * padding_length) + dep_ids
                    else:
                        input_ids = input_ids + \
                                    ([pad_token] * padding_length)  # [pad_token] defined as 0, the [PAD] token's id
                        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                        pos_ids = pos_ids + ([pad_token] * padding_length)
                        dep_ids = dep_ids + ([pad_token] * padding_length)

                    assert len(input_ids) == max_seq_length
                    assert len(input_mask) == max_seq_length
                    assert len(segment_ids) == max_seq_length
                    assert len(pos_ids) == max_seq_length
                    assert len(dep_ids) == max_seq_length

                    # Append context-pair to the pairs list.
                    # This simple if statement works, as gloss extensions require the use of pos
                    if args.use_dependencies and args.use_dependencies:
                        pairs.append(
                            BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                      pos_ids=pos_ids, dep_ids=dep_ids,
                                      label_id=label)
                        )
                    elif args.use_dependencies and not args.use_dependencies:
                        pairs.append(
                            BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                      pos_ids=pos_ids, label_id=label)
                        )
                    elif not args.use_dependencies and args.use_dependencies:
                        pairs.append(
                            BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                      dep_ids=dep_ids, label_id=label)
                        )

                features.append(pairs)

            else:
                for seq, label in sequences:  # seq is the gloss, label is either 0 or 1
                    tokens_b = tokenizer.tokenize(seq)

                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

                    # The convention in BERT is:
                    # (a) For sequence pairs:
                    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                    #
                    # Where "type_ids" are used to indicate whether this is the first
                    # sequence or the second sequence. The embedding vectors for `type=0` and
                    # `type=1` were learned during pre-training and are added to the wordpiece
                    # embedding vector (and position vector). This is not *strictly* necessary
                    # since the [SEP] token unambiguously separates the sequences, but it makes
                    # it easier for the model to learn the concept of sequences.
                    #
                    # For classification tasks, the first vector (corresponding to [CLS]) is
                    # used as as the "sentence vector". Note that this only makes sense because
                    # the entire model is fine-tuned.
                    tokens = tokens_a + [sep_token]
                    segment_ids = [sequence_a_segment_id] * len(tokens)

                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                    if cls_token_at_end:
                        tokens = tokens + [cls_token]
                        segment_ids = segment_ids + [cls_token_segment_id]
                    else:
                        tokens = [cls_token] + tokens
                        segment_ids = [cls_token_segment_id] + segment_ids

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    padding_length = max_seq_length - len(input_ids)
                    if pad_on_left:
                        input_ids = ([pad_token] * padding_length) + input_ids
                        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                    else:
                        input_ids = input_ids + (
                                    [pad_token] * padding_length)  # [pad_token] defined as 0, the [PAD] token's id
                        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                    assert len(input_ids) == max_seq_length
                    assert len(input_mask) == max_seq_length
                    assert len(segment_ids) == max_seq_length

                    # These are the input embedding IDs, the input mask, segment IDs and labels.
                    pairs.append(
                        BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label)
                    )

                features.append(pairs)

    return features


# ====== Four functions for collating batch ======
# Pos and dep have four comninations: TT, TF, FT and FF
def collate_batch_pos_dep(batch):
    # The following part(s) is unmodified
    max_seq_length = len(
        batch[0][0].input_ids)  # length of the input ids, which is the dimensionality of the BERT model.
    """
    Batch: contains instances
    Instance: contains context-gloss pairs, that all share the same input sentence but have different candidate glosses.
    context-gloss pair: sentence and sense definition in the format: [CLS] sentence [SEP] gloss [SEP] [PAD]...
    """
    collated = []
    # The following part(s) is modified
    for sub_batch in batch:
        batch_size = len(sub_batch)
        # 6 set of tensors (input_ids, input_mask, segment_ids, label_id)
        sub_collated = [torch.zeros([batch_size, max_seq_length], dtype=torch.long) for _ in range(5)] + \
                       [torch.zeros([batch_size],
                                    dtype=torch.long)]  # dtype is the data format, long is 64-bit integer and signed.

        for i, bert_input in enumerate(sub_batch):
            sub_collated[0][i] = torch.tensor(bert_input.input_ids, dtype=torch.long)
            sub_collated[1][i] = torch.tensor(bert_input.input_mask, dtype=torch.long)
            sub_collated[2][i] = torch.tensor(bert_input.segment_ids, dtype=torch.long)
            sub_collated[3][i] = torch.tensor(bert_input.pos_ids, dtype=torch.long)
            sub_collated[4][i] = torch.tensor(bert_input.dep_ids, dtype=torch.long)
            sub_collated[5][i] = torch.tensor(bert_input.label_id, dtype=torch.long)

        collated.append(sub_collated)

    return collated

def collate_batch_pos(batch):
    # The following part(s) is unmodified
    max_seq_length = len(
        batch[0][0].input_ids)  # length of the input ids, which is the dimensionality of the BERT model.
    """
    Batch: contains instances
    Instance: contains context-gloss pairs, that all share the same input sentence but have different candidate glosses.
    context-gloss pair: sentence and sense definition in the format: [CLS] sentence [SEP] gloss [SEP] [PAD]...
    """
    collated = []
    # The following part(s) is modified
    for sub_batch in batch:
        batch_size = len(sub_batch)
        # 4 set of tensors (input_ids, input_mask, segment_ids, label_id)
        sub_collated = [torch.zeros([batch_size, max_seq_length], dtype=torch.long) for _ in range(4)] + \
                       [torch.zeros([batch_size],
                                    dtype=torch.long)]  # dtype is the data format, long is 64-bit integer and signed.

        for i, bert_input in enumerate(sub_batch):
            sub_collated[0][i] = torch.tensor(bert_input.input_ids, dtype=torch.long)
            sub_collated[1][i] = torch.tensor(bert_input.input_mask, dtype=torch.long)
            sub_collated[2][i] = torch.tensor(bert_input.segment_ids, dtype=torch.long)
            sub_collated[3][i] = torch.tensor(bert_input.pos_ids, dtype=torch.long)
            sub_collated[4][i] = torch.tensor(bert_input.label_id, dtype=torch.long)

        collated.append(sub_collated)

    return collated

def collate_batch_dep(batch):
    # The following part(s) is unmodified
    max_seq_length = len(
        batch[0][0].input_ids)  # length of the input ids, which is the dimensionality of the BERT model.
    """
    Batch: contains instances
    Instance: contains context-gloss pairs, that all share the same input sentence but have different candidate glosses.
    context-gloss pair: sentence and sense definition in the format: [CLS] sentence [SEP] gloss [SEP] [PAD]...
    """
    collated = []
    # The following part(s) is modified
    for sub_batch in batch:
        batch_size = len(sub_batch)
        # 4 set of tensors (input_ids, input_mask, segment_ids, label_id)
        sub_collated = [torch.zeros([batch_size, max_seq_length], dtype=torch.long) for _ in range(4)] + \
                       [torch.zeros([batch_size],
                                    dtype=torch.long)]  # dtype is the data format, long is 64-bit integer and signed.

        for i, bert_input in enumerate(sub_batch):
            sub_collated[0][i] = torch.tensor(bert_input.input_ids, dtype=torch.long)
            sub_collated[1][i] = torch.tensor(bert_input.input_mask, dtype=torch.long)
            sub_collated[2][i] = torch.tensor(bert_input.segment_ids, dtype=torch.long)
            sub_collated[3][i] = torch.tensor(bert_input.dep_ids, dtype=torch.long)
            sub_collated[4][i] = torch.tensor(bert_input.label_id, dtype=torch.long)

        collated.append(sub_collated)

    return collated

def collate_batch(batch):
    # The following part(s) is unmodified
    max_seq_length = len(
        batch[0][0].input_ids)  # length of the input ids, which is the dimensionality of the BERT model.
    """
    Batch: contains instances
    Instance: contains context-gloss pairs, that all share the same input sentence but have different candidate glosses.
    context-gloss pair: sentence and sense definition in the format: [CLS] sentence [SEP] gloss [SEP] [PAD]...
    """
    collated = []
    # The following part(s) is modified
    for sub_batch in batch:
        batch_size = len(sub_batch)
        # 4 set of tensors (input_ids, input_mask, segment_ids, label_id)
        sub_collated = [torch.zeros([batch_size, max_seq_length], dtype=torch.long) for _ in range(3)] + \
                       [torch.zeros([batch_size],
                                    dtype=torch.long)]  # dtype is the data format, long is 64-bit integer and signed.

        for i, bert_input in enumerate(sub_batch):
            sub_collated[0][i] = torch.tensor(bert_input.input_ids, dtype=torch.long)
            sub_collated[1][i] = torch.tensor(bert_input.input_mask, dtype=torch.long)
            sub_collated[2][i] = torch.tensor(bert_input.segment_ids, dtype=torch.long)
            sub_collated[3][i] = torch.tensor(bert_input.label_id, dtype=torch.long)

        collated.append(sub_collated)

    return collated

# Indexing changed to fit row with POS information
def write_predictions(output_dir, csv_path, predictions, suffix=None):
    # The following part(s) is modified
    def deserialize_csv_record(row):
        return row[0], eval(row[3])

    # The following part(s) is unmodified
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_name = os.path.basename(csv_path).split('.')[0]
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}_predictions.txt" if suffix is None else f"{dataset_name}_predictions_{suffix}.txt"
    )
    records = _create_records_from_csv(csv_path, deserialize_csv_record)
    with open(output_file, "w") as f:
        for predicted, (guid, candidates) in zip(predictions, records):
            print(f"{guid} {candidates[predicted]}", file=f)

