import math
from collections import OrderedDict

import torch
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
import spacy
from spacy.symbols import ORTH
import regex as re
from utils.dataset_syntax import _get_gloss_extension

pos_vocab_list = ["[PAD]",
                  "[ADJ]", "[ADP]", "[ADV]", "[AUX]", "[CCONJ]", "[CONJ]",
                  "[DET]", "[INTJ]", "[NOUN]", "[NUM]", "[PART]", "[PRON]",
                  "[PROPN]", "[PUNCT]", "[SCONJ]", "[SYM]", "[VERB]", "[X]",
                  "[TGT_POS]", "[CLS_POS]", "[SEP_POS]",
                  "[NONE_POS]", "[GLOSS_POS]"]

dep_vocab_list = ["[PAD]", "[TGT_DEP]", "[CLS_DEP]", "[SEP_DEP]", "[NONE_DEP]", "[GLOSS_DEP]"]

if __name__ == '__main__':
    # Load spacy model and BERT's tokenizer
    spacy_model = spacy.load("en_core_web_trf")
    add_TGT = [{ORTH: "[TGT]"}]
    spacy_model.tokenizer.add_special_case("[TGT]", add_TGT)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    target_token = "[TGT]"
    #if target_token not in tokenizer.additional_special_tokens:
    #    tokenizer.add_special_tokens({'additional_special_tokens': [target_token]})
    #    assert target_token in tokenizer.additional_special_tokens

    print(tokenizer.convert_ids_to_tokens(1))


    print(tokenizer.additional_special_tokens_ids)
    print(len([(i, w) for i, w in tokenizer.vocab.items() if re.search("\[unused\d+\]", i)]))
    print({w: i for w, i in tokenizer.vocab.items() if re.search("\[unused\d+\]", w)})
    print(tokenizer.vocab["[unused0]"])
    tokenizer.vocab = OrderedDict([("[TGT]", i) if w == "[unused0]" else (w, i) for w, i in tokenizer.vocab.items()])
    print(tokenizer.convert_ids_to_tokens(1))
    print(tokenizer.convert_tokens_to_ids("[TGT]"))


    print("SpaCy model and tokenizer loaded!")
    # Make [TGT]'s token entries (as in text, POS, dependencies, tokenized form)
    spacy_tgt_tpl = ('[TGT]', '[PAD]', "[PAD]", ['[TGT]'])  # If statement: whether to use syntax for special tokens

    labels = spacy_model.pipe_labels
    dep_vocab = ["[" + dep + "]" for dep in labels['parser']]
    #[print(ele) for ele in dep_vocab]
    #print(dep_vocab)

    conv_dict = {
        "n": "[NOUN]",
        "v": "[VERB]",
        "a": "[ADJ]",
        # Satellite adjective, adjective for my purposes
        "s": "[ADJ]",
        "r": "[ADV]"
    }

    key = "break_dance%1:04:00::"
    gloss_extension = _get_gloss_extension(key, conv_dict)

    print("\n\n", gloss_extension,"\n\n")

    sentence = "Kids were-[TGT]-break-[TGT]-dancing at the loveliest street corner"
    # For sentences like "Kids were [TGT] break [TGT]-dancing at the street corner":
    # Make sure to insert a space after the target tokens
    sentence = re.sub(r"(\[TGT\])([^ ])", r"\1 \2", sentence)
    sentence = re.sub(r"([^ ])(\[TGT\])", r"\1 \2", sentence)
    print(sentence)

    # Remove [TGT] tokens from sentence
    no_tgt_sent = re.sub(r"\[TGT\]", "", sentence).strip()
    no_tgt_sent = re.sub(r"\s{2,}", " ", no_tgt_sent)
    print(no_tgt_sent)

    # Analyze sentence with and without [TGT] tokens
    spacy_doc = spacy_model(no_tgt_sent)
    spacy_tokens = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text)) for token in spacy_doc]
    tgt_doc = spacy_model(sentence)
    tgt_tokens = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text)) for token in tgt_doc]

    print(tgt_tokens)

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
        print(complete_tokens)
        complete_tokens.insert(tgt_index[1], spacy_tgt_tpl)

   # [print(ele[3]) for ele in complete_tokens]
    print(complete_tokens)

    print("\n\nBERT input tokens:")
    bert_tokens = [token for sublist in complete_tokens for token in sublist[3]]
    print(bert_tokens)
    bert_pos_tokens = [[ele[1]] * len(ele[-1]) for ele in complete_tokens]
    bert_pos_tokens = [pos if re.search(r"^\[.+\]$", pos) else "[" + pos + "]" for sublist in bert_pos_tokens for pos in
                       sublist]
    print(bert_pos_tokens)
    bert_dep_tokens = [[ele[2]] * len(ele[-1]) for ele in complete_tokens]
    bert_dep_tokens = [dep if re.search(r"^\[.+\]$", dep) else "[" + dep + "]" for sublist in bert_dep_tokens for dep in
                       sublist]
    print(bert_dep_tokens)
    assert (len(bert_tokens) == len(bert_pos_tokens) == len(bert_dep_tokens))




    #print(tgt_tokens)






























































    run_below = False

    if run_below:
        # Load spacy model and BERT's tokenizer
        spacy_model = spacy.load("en_core_web_trf")
        add_TGT = [{ORTH: "[TGT]"}]
        spacy_model.tokenizer.add_special_case("[TGT]", add_TGT)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # tm = spacy_model.tokenizer.vocab.morphology.tag_map

        # print(tm)

        labels = spacy_model.pipe_labels
        dep_vocab = ["[" + dep + "]" for dep in labels['parser']]
        # [print(ele) for ele in dep_vocab]
        # [print(lab) for lab in labels["tagger"]]
        # [print(key) for key in labels.keys()]

        sentence = "Is your [TGT] purchasing agent [TGT] offering too much free buying service for employees and the other purchasing agent ?"
        """
        1) find target word(s)
        2) remove [TGT] tokens to analyze the sentence with spacy
        3) there can be multiple instances of the target word(s), so make sure to keep track of the correct target!
        4) Further tokenize spacy's tokens with BERT's tokenizer.
        5) add the [TGT] tokens again to get the complete sentence
        """

        # ====== Find target word(s) in the sentence ======
        #sentence = "Polls will [TGT] be [TGT] in the water office , Edward will be here and I will be at the arcadian office etc. ."
        #sentence = "[TGT] plant [TGT] the tree there ."
        #sentence = "do not [TGT] speed [TGT]"
        sentence = "Is your [TGT] purchasing agent [TGT] offering too much free buying service for employees and the other purchasing agent ?"
        # The target is found using .+ rather than \w+ as there are multi-word targets like "purchasing agent"
        dissect_sent = re.findall("(.*)(\[TGT\]\s)(.+)(\s\[TGT\])(.*)", sentence)
        print(dissect_sent)
        # The reason why the indexing below works is that the (.*) still creates an empty element "".
        target_word = dissect_sent[0][2]
        if re.findall(" ", target_word):
            target_tokens = spacy_model(target_word)
            target_tokens = [token.text for token in target_tokens]
            number_of_target_tokens = len(target_tokens)
            print(target_tokens)
        else:
            number_of_target_tokens = 1
        print(target_word)
        print("Target contains: ", number_of_target_tokens, "token(s)")

        # Find all instances of the target word
        instances_of_target_word = re.findall(f"({target_word})(\s\[TGT\])*", sentence)
        #print(instances_of_target_word)
        # Find the actual target word among the candidate targets
        for i, instance in enumerate(instances_of_target_word):
            if instance[1] == " [TGT]":
                instance_that_is_target = i
        print("instance of target word that is the target:", instance_that_is_target)

        # ====== Remove [TGT] tokens ======
        no_tgt_sent = re.sub(r"\[TGT\]", "", sentence).strip()
        no_tgt_sent = re.sub(r"\s{2,}", " ", no_tgt_sent)
        print(no_tgt_sent)

        # ====== Analyze sentence with spacy ======
        spacy_doc = spacy_model(no_tgt_sent)
        spacy_tokens = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text)) for token in spacy_doc]
        print("===== Print analyzed sentence =========")
        [print(ele) for ele in spacy_tokens]
        print("============================================")

        # Make [TGT]'s token entries (as in text, POS, dependencies, tokenized form)
        spacy_tgt_tpl = [('[TGT]', '[PAD]', "[PAD]", ['[TGT]'])]

        # ====== Add the [TGT] tokens again now that spacy has analyzed the sentence ======
        # First deal with case where the target is multiple words
        if number_of_target_tokens > 1:
            indexes = [(tokens[0], i) for i, tokens in enumerate(spacy_tokens)]
            # Find indexes of the occurrences of the first word in the target, e.g., "fire" in "fire truck".
            first_target_word = [tpl for tpl in indexes if tpl[0] == target_tokens[0]]
            # Find the possible word sequences starting at one of the initial target words.
            possible_target_sequences = [[i + index for i in range(number_of_target_tokens)]
                                         for token, index in first_target_word]
            #print(possible_target_sequences)
            # If there are multiple possible sequences pick the instance that is target, else just go with that sequence.
            if len(possible_target_sequences) > 1:
                # Check if each sequences is a complete match or a partial match.
                seqs_matching_whole_target = []
                for seq in possible_target_sequences:
                    #print(indexes[seq[0]:seq[-1]+1])
                    #print(spacy_tokens[seq[0]:seq[-1]+1][0])
                    seq_tokens = [token for token, i in indexes[seq[0]:seq[-1]+1]]
                    print(seq_tokens)
                    if seq_tokens == target_tokens:
                        seqs_matching_whole_target.append(seq)
                #print("seqs matching whole target:", seqs_matching_whole_target)
                target_indexes = seqs_matching_whole_target[instance_that_is_target]
                #print("target indexes:", target_indexes)
            else:
                target_indexes = possible_target_sequences[0]

            # Split the sentence into three parts:
            #  - The part before the target
            #  - The target
            #  - The part after the target
            # Before the target token
            # If the target is the first token of the sentence
            if target_indexes[0] == 0:
                tokens_before_tgt = []
            # if the target is the last word of the sentence
            #elif target_indexes[-1] == len(spacy_tokens) - 1:  # -1 because of 0 indexing
            #    tokens_before_tgt = spacy_tokens[0:target_indexes[0]]
            else:
                tokens_before_tgt = spacy_tokens[0:target_indexes[0]]

            # The target tokens
            tgt_word_token = spacy_tokens[target_indexes[0]:target_indexes[-1]+1]

            # After the target token
            if target_indexes[0] == 0:  # if the target is the first word
                tokens_after_tgt = spacy_tokens[target_indexes[-1]+1:]
            elif target_indexes[-1] == len(spacy_tokens) - 1:  # if the target is the last word of the sentence
                tokens_after_tgt = []
            else:
                tokens_after_tgt = spacy_tokens[target_indexes[-1] + 1:]

        # Now deal with cases where the target is a single word
        else:
            tgt_indexes = [i if tokens[0] == target_word else None for i, tokens in enumerate(spacy_tokens)]
            #print(spacy_tgt_indexes)
            # Remove all None from the tgt indexes to be left only with the indexes of the possible targets
            tgt_indexes = [ele for ele in tgt_indexes if ele != None]
            # Keep only the index of the instance of the target word that is actually the target
            tgt_index = [index for num, index in enumerate(tgt_indexes) if num == instance_that_is_target]

            # Split the sentence into three parts:
            #  - The part before the target
            #  - The target
            #  - The part after the target
            # Before the target token
            if tgt_index[0] == 0: # If the target is the first token of the sentence
                tokens_before_tgt = []
            elif tgt_index == len(spacy_tokens)-1: # if the target is the last word of the sentence
                tokens_before_tgt = spacy_tokens[0:-1]
            else:
                tokens_before_tgt = spacy_tokens[0:tgt_index[0]]

            # The target token
            tgt_word_token = [spacy_tokens[tgt_index[0]]]

            # After the target token
            if tgt_index[0] == 0: # if the target is the first word
                tokens_after_tgt = spacy_tokens[1:]
            elif tgt_index[0] == len(spacy_tokens)-1: # if the target is the last word of the sentence
                tokens_after_tgt = []
            else:
                tokens_after_tgt = spacy_tokens[tgt_index[0]+1:]

        print("tokens before target:", tokens_before_tgt)
        print("Target word token:", tgt_word_token)
        print("tokens after target: ", tokens_after_tgt)

        spacy_tokens_complete = tokens_before_tgt + spacy_tgt_tpl + tgt_word_token + spacy_tgt_tpl + tokens_after_tgt
        #print("\n\n========================================\n\n")
        #[print(ele) for ele in spacy_tokens_complete]
        bert_tokens = [word for sublist in spacy_tokens_complete for word in sublist[3]]
        print(bert_tokens)
        bert_pos_tokens = [[ele[1]]*len(ele[3]) for ele in spacy_tokens_complete]
        bert_pos_tokens = [pos if re.search(r"^\[.+\]$", pos) else "["+pos+"]" for sublist in bert_pos_tokens for pos in sublist]
        print(bert_pos_tokens)
        bert_dep_tokens = [[ele[2]]*len(ele[3]) for ele in spacy_tokens_complete]
        bert_dep_tokens = [dep if re.search(r"^\[.+\]$", dep) else "["+dep+"]" for sublist in bert_dep_tokens for dep in sublist]
        print(bert_dep_tokens)
        assert (len(bert_tokens) == len(bert_pos_tokens) == len(bert_dep_tokens))
