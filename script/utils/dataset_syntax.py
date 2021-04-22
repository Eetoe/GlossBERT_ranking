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


def load_dataset(args, csv_path, tokenizer, max_sequence_length, spacy_model=None):
    # Define record structure and how to deserialize it
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
    cached_features_file = os.path.join(data_dir, f"cached_{dataset_name}-{max_sequence_length}")
    if args.use_pos_tags:
        cached_features_file = cached_features_file + "-pos"
    if args.use_dependencies:
        cached_features_file = cached_features_file + "-dep"
    if args.use_gloss_extensions:
        cached_features_file = cached_features_file + "-glosses_extended"
    if args.gloss_extensions_w_tgt:
        cached_features_file = cached_features_file + "_w_tgt"
    if args.zero_syntax_for_special_tokens:
        cached_features_file = cached_features_file + "-no_syntax_for_special"


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


def _get_gloss_extension(key, conversion_dict):
    # synset = wn.synset_from_sense_key(key) # <-- This is flawed, use below code instead.
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
def _create_features_from_records(args, spacy_model, records, max_seq_length, tokenizer, cls_token_at_end=False,
                                  pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False, ):
    """ Convert records to list of features. Each feature is a list of sub-features where the first element is
        always the feature created from context-gloss pair while the rest of the elements are features created from
        context-example pairs (if available)

        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    if args.use_pos_tags or args.use_dependencies:
        spacy_model = spacy_model

        if args.zero_syntax_for_special_tokens:
            # Make [TGT]'s token entries (as in text, POS, dependencies, tokenized form)
            spacy_tgt_tpl = ('[TGT]', '[PAD]', "[PAD]", ['[TGT]'])
        else:
            spacy_tgt_tpl = ('[TGT]', '[TGT_POS]', "[TGT_DEP]", ['[TGT]'])

    # Used for creating pos tag for gloss extenstion
    if args.use_gloss_extensions:
        conv_dict = {
            "n": "[NOUN]",
            "v": "[VERB]",
            "a": "[ADJ]",
            # Satellite adjective, adjective for my purposes
            "s": "[ADJ]",
            "r": "[ADV]"
        }

    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        # ====== Create tokens for BERT from the sentence ======
        """
            - First take care of the case where syntax info is used
            - Second take care of the case where it isn't
        
        """
        if args.use_pos_tags or args.use_dependencies:
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

        else:
            tokens_a = tokenizer.tokenize(record.sentence)

        # ====== Create tokens for BERT from the gloss ======
        # Make empty list for context-gloss pairs
        pairs = []

        # Sequences to loop through
        sequences = [(gloss, 1 if i in record.targets else 0, record.sense_keys[i]) for i,
                                                                                        gloss in
                     enumerate(record.glosses)]

        # ====== If using syntax tokens ======
        # Get the gloss from the record; 1 if i is in the record targets or 0 if i is not in the record targets.
        if args.use_pos_tags or args.use_dependencies:
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

                # ====== Truncate sentences and implement gloss extensions if applicable ======
                if args.use_gloss_extensions:
                    gloss_extension_tuple = _get_gloss_extension(key, conv_dict)
                    gloss_ex_tokens = tokenizer.tokenize(gloss_extension_tuple[0])
                    num_gloss_ex = len(gloss_ex_tokens)
                    gloss_ex_pos_tokens = [gloss_extension_tuple[1]]*num_gloss_ex
                    gloss_ex_dep_tokens = [gloss_extension_tuple[2]]*num_gloss_ex
                    max_seq_len_with_gloss_ex = max_seq_length - num_gloss_ex - 3

                    if args.gloss_extensions_w_tgt:
                        # add target tokens to gloss extension tokens, then add appropriate syntax tokens.
                        gloss_ex_tokens.insert(0, "[TGT]")
                        gloss_ex_tokens.append("[TGT]")

                        gloss_ex_pos_tokens.insert(0, spacy_tgt_tpl[1])
                        gloss_ex_pos_tokens.append(spacy_tgt_tpl[1])

                        gloss_ex_dep_tokens.insert(0, spacy_tgt_tpl[2])
                        gloss_ex_dep_tokens.append(spacy_tgt_tpl[2])

                        max_seq_len_with_gloss_ex -= 2
                        num_gloss_ex += 2

                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_len_with_gloss_ex)
                    _truncate_seq_pair(bert_pos_tokens_a, bert_pos_tokens_b, max_seq_len_with_gloss_ex)
                    _truncate_seq_pair(bert_dep_tokens_a, bert_dep_tokens_b, max_seq_len_with_gloss_ex)

                else:
                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                    _truncate_seq_pair(bert_pos_tokens_a, bert_pos_tokens_b, max_seq_length - 3)
                    _truncate_seq_pair(bert_dep_tokens_a, bert_dep_tokens_b, max_seq_length - 3)


                # ====== Merge the context and gloss tokens to form the context gloss pair ======
                tokens = tokens_a + [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)
                if args.zero_syntax_for_special_tokens:
                    pos_tokens = bert_pos_tokens_a + ["[PAD]"]
                    dep_tokens = bert_dep_tokens_a + ["[PAD]"]
                else:
                    pos_tokens = bert_pos_tokens_a + ["[SEP_POS]"]
                    dep_tokens = bert_dep_tokens_a + ["[SEP_DEP]"]

                # Add gloss extension
                if args.use_gloss_extensions:
                    tokens += gloss_ex_tokens
                    pos_tokens += gloss_ex_pos_tokens
                    dep_tokens += gloss_ex_dep_tokens

                tokens += tokens_b + [sep_token]
                # +1 to account for [SEP] token
                if args.use_gloss_extensions:
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1 + num_gloss_ex)
                else:
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

                # When converting tokens to ids, it depends on whether the tokenizer has the syntax embedding.
                # This depends on args. Therefore, if statements based on args are implemented from here on out.
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                if args.use_pos_tags:
                    pos_ids = tokenizer.convert_syntax_tokens_to_ids(pos_tokens, "pos")
                if args.use_dependencies:
                    dep_ids = tokenizer.convert_syntax_tokens_to_ids(dep_tokens, "dep")

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                    if args.use_pos_tags:
                        pos_ids = ([pad_token] * padding_length) + pos_ids
                    if args.use_dependencies:
                        dep_ids = ([pad_token] * padding_length) + dep_ids
                else:
                    input_ids = input_ids + \
                                ([pad_token] * padding_length)  # [pad_token] defined as 0, the [PAD] token's id
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                    if args.use_pos_tags:
                        pos_ids = pos_ids + ([pad_token] * padding_length)
                    if args.use_dependencies:
                        dep_ids = dep_ids + ([pad_token] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                if args.use_pos_tags:
                    assert len(pos_ids) == max_seq_length
                if args.use_dependencies:
                    assert len(dep_ids) == max_seq_length

                # Append context-pair to the pairs list.
                if args.use_pos_tags and args.use_dependencies:
                    pairs.append(
                        BertInputPosDep(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                        pos_ids=pos_ids, dep_ids=dep_ids,
                                        label_id=label)
                    )
                elif args.use_pos_tags and not args.use_dependencies:
                    pairs.append(
                        BertInputPos(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                     pos_ids=pos_ids, label_id=label)
                    )
                elif not args.use_pos_tags and args.use_dependencies:
                    pairs.append(
                        BertInputDep(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                     dep_ids=dep_ids, label_id=label)
                    )

            features.append(pairs)

        # ====== Without syntax tokens ======
        else:
            for seq, label, key in sequences:  # seq is the gloss, label is either 0 or 1
                tokens_b = tokenizer.tokenize(seq)

                if args.use_gloss_extensions:
                    gloss_extension_tuple = _get_gloss_extension(key, conv_dict)
                    gloss_ex_tokens = tokenizer.tokenize(gloss_extension_tuple[0])
                    num_gloss_ex = len(gloss_ex_tokens)
                    max_seq_len_with_gloss_ex = max_seq_length - num_gloss_ex - 3

                    if args.gloss_extensions_w_tgt:
                        # add target tokens to gloss extension tokens, then add appropriate syntax tokens.
                        gloss_ex_tokens.insert(0, "[TGT]")
                        gloss_ex_tokens.append("[TGT]")

                        max_seq_len_with_gloss_ex -= 2
                        num_gloss_ex += 2

                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_len_with_gloss_ex)
                else:
                    # Truncate sentence gloss pair so thy're less than the maximum input length
                    # -3 to make room for the [CLS] token and the 2 [SEP] tokens.
                    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

                # Start merging the context gloss pairs
                # The segment ids are built gradually to keep track of what's context and what's glosses

                tokens = tokens_a + [sep_token]
                segment_ids = [sequence_a_segment_id] * len(tokens)

                if args.use_gloss_extensions:
                    tokens += gloss_ex_tokens

                tokens += tokens_b + [sep_token]
                if args.use_gloss_extensions:
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1 + num_gloss_ex)
                else:
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
    print(features[0][0])
    print(tokens)
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
        return row[0], eval(row[2])

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
