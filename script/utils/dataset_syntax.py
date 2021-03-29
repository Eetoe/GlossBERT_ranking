import csv
import os
from collections import namedtuple

import torch
from tqdm import tqdm
import spacy

# POS information has been added.
GlossSelectionRecord = namedtuple("GlossSelectionRecord",
                                  ["guid", "sentence", "pos", "sense_keys", "glosses", "gloss_word_and_pos", "targets"])
BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "pos_ids", "label_id"])

def get_pos_tokens(args, sentence):
    model_to_use = args.spacy_model
    nlp_model = spacy.load(model_to_use)
    doc = nlp_model(sentence)
    pos_list = ["["+str(token.pos_)+"]" for token in doc]
    return pos_list

# POS information has been added
def load_dataset(csv_path, tokenizer, max_sequence_length):
    # The following part(s) is modified
    def deserialize_csv_record(row):
        return GlossSelectionRecord(row[0], row[1], eval(row[2]), eval(row[3]), eval(row[4]), eval(row[5]), [int(t) for t in eval(row[6])])

    # The following part(s) is unmodified
    return _load_and_cache_dataset(
        csv_path,
        tokenizer,
        max_sequence_length,
        deserialize_csv_record
    )


# POS information has been added
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


def _load_and_cache_dataset(csv_path, tokenizer, max_sequence_length, deserialze_fn):
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

        features = _create_features_from_records(records, max_sequence_length, tokenizer,
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


# Creates features, i.e., the input given to BERT, to train on.
def _create_features_from_records(records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]',
                                  pad_token=0, special_token_pos='[NONE]', gloss_pos_token='[GLOSS_POS]',
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False):
    """ Convert records to list of features. Each feature is a list of sub-features where the first element is
        always the feature created from context-gloss pair while the rest of the elements are features created from
        context-example pairs (if available)

        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        tokens_a = tokenizer.tokenize(record.sentence)
        # in tqdm: get the gloss from the record and a 1 if i is in the record targets or 0 if i is not in the record targets.
        sequences = [(gloss, 1 if i in record.targets else 0, record.gloss_word_and_pos[i]) for i, gloss in enumerate(record.glosses)]

        # The following part(s) is modified
        # make a list of pos tokens, so that each token has its word's pos.
        # Word refers to full word, token refers to subword.
        pos_of_sentence = record.pos  # Should be a list
        original_word_position = 0
        pos_tokens_a = []
        for i, token in enumerate(tokens_a):
            # Every time the token doesn't start with ##, indicating a non-initial subword, go to the next word.
            # Exception for the very first token, which is the start of the first word and thus belongs to word 0.
            if i != 0 and token.startswith("##") == False:
                original_word_position += 1

            # Add the pos of the token's parent word to the token's pos embedding.
            pos_tokens_a[i] = pos_of_sentence[original_word_position]

        # Make sure there's a pos for each token.
        assert (len(pos_tokens_a) == len(tokens_a))

        pairs = []
        for seq, label, word_and_pos in sequences:  # seq is the gloss, label is either 0 or 1
            tokens_b = tokenizer.tokenize(seq)
            pos_tokens_b = [gloss_pos_token]*len(tokens_b)

            # From the tuple word and pos, extract target word and tokenize.
            tokens_b_target_word = tokenizer.tokenize(word_and_pos[0])
            # From the tuple word and pos, extract pos and add 1 for each token in target word.
            tokens_b_gloss_pos = word_and_pos[1]*len(tokens_b_target_word)

            # Add the target word and its pos to tokens_b
            tokens_b = tokens_b_target_word + tokens_b
            pos_tokens_b = tokens_b_gloss_pos + pos_tokens_b

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] "- 3"
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
            pos_tokens = pos_tokens_a + [special_token_pos]

            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            pos_tokens += pos_tokens_b + [special_token_pos]

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
                pos_tokens = pos_tokens + [special_token_pos]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids
                pos_tokens = [special_token_pos] + pos_tokens

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            pos_ids = tokenizer.convert_tokens_to_ids(pos_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                pos_ids = ([special_token_pos] * padding_length) + pos_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)  # [pad_token] defined as 0, the [PAD] token's id
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                pos_ids = pos_ids + ([special_token_pos] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(pos_ids) == max_seq_length

            # These are the input embedding IDs, the input mask, segment IDs and labels. What exactly is label?
            pairs.append(
                BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, pos_ids=pos_ids, label_id=label)
            )

        features.append(pairs)

    return features


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
