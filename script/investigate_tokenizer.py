import math

import torch
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
import spacy
import regex as re

pos_vocab_list = ["[ADJ]", "[ADP]", "[ADV]", "[AUX]", "[CCONJ]", "[CONJ]",
                  "[DET]", "[INTJ]", "[NOUN]", "[NUM]", "[PART]", "[PRON]",
                  "[PROPN]", "[PUNCT]", "[SCONJ]", "[SYM]", "[VERB]", "[X]",
                  "[TGT_POS]", "[CLS_POS]", "[SEP_POS]",
                  "[TGT_DEP]", "[CLS_DEP]", "[SEP_DEP]",
                  "[NONE_POS]", "[NONE_DEP]",
                  "[GLOSS_POS]", "[GLOSS_DEP]"]

if __name__ == '__main__':
    # Load spacy model and BERT's tokenizer
    spacy_model = spacy.load("en_core_web_trf")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tm = spacy_model.tokenizer.vocab.morphology.tag_map

    print(tm)

    labels = spacy_model.pipe_labels
    dep_vocab = ["[" + dep + "]" for dep in labels['parser']]
    #[print(ele) for ele in dep_vocab]
    #[print(lab) for lab in labels["tagger"]]
    #[print(key) for key in labels.keys()]

    """
    1) find target word
    2) remove [TGT] tokens to analyze the sentence with spacy
    3) there can be multiple instances of the target word, so make sure to keep track of the correct target!
    4) Further tokenize spacy's tokens with BERT's tokenizer.
    5) add the [TGT] tokens again to get the complete sentence
    """

    # Find target word in the sentence
    #sentence = "Polls will [TGT] be [TGT] in the water office , Edward will be here and I will be at the arcadian office etc. ."
    #sentence = "[TGT] plant [TGT] the tree there ."
    #sentence = "do not [TGT] speed [TGT]"
    sentence = "dislike intensely; feel antipathy or aversion [TGT] towards [TGT]"
    dissect_sent = re.findall("(.*)(\[TGT\]\s)(\w+)(\s\[TGT\])(.*)", sentence)
    #print(dissect_sent)
    # The reason why the indexing below works is that the (.*) still creates an empty element "".
    target_word = dissect_sent[0][2]
    print(target_word)

    # Find the instance of the target word that is the target
    instances_of_target_word = re.findall(f"({target_word})(\s\[TGT\])*", sentence)
    #print(instances_of_target_word)
    for i, instance in enumerate(instances_of_target_word):
        if instance[1] == " [TGT]":
            instance_that_is_target = i
    print("instance of target word that is the target: ", instance_that_is_target)

    # Remove [TGT] tokens
    no_tgt_sent = re.sub(r"\[TGT\]", "", sentence).strip()
    no_tgt_sent = re.sub(r"\s{2,}", " ", no_tgt_sent)
    print(no_tgt_sent)

    # Analyze sentence with spacy
    spacy_doc = spacy_model(no_tgt_sent)
    spacy_tokens = [(token.text, token.pos_, token.dep_, tokenizer.tokenize(token.text)) for token in spacy_doc]
    print("===== Print analyzed sentence =========")
    [print(ele) for ele in spacy_tokens]
    print("============================================")

    # Make [TGT]'s token entries (as in text, POS, dependencies, tokenized form)
    # Make it togglable
    spacy_tgt_list = [('[TGT]', '[NONE]', "[NONE]", ['[TGT]'])]

    # Add the [TGT] tokens again now that spacy has analyzed the sentence
    # get index of all instances of the target word. Non-target words have index None
    tgt_indexes = [i if tokens[0] == target_word else None for i, tokens in enumerate(spacy_tokens)]
    #print(spacy_tgt_indexes)
    # Remove all None from the tgt indexes to be left only with the indexes of the possible targets
    tgt_indexes = [ele for ele in tgt_indexes if ele != None]
    # Keep only the index of the instance of the target word that is actually the target
    tgt_index = [index for num, index in enumerate(tgt_indexes) if num == instance_that_is_target]
    print(tgt_index)

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
        tokens_after_tgt = spacy_tokens[tgt_index[0]:]

    print("tokens before target:", tokens_before_tgt)
    print("tokens after target: ", tokens_after_tgt)

    spacy_tokens_complete = tokens_before_tgt + spacy_tgt_list + tgt_word_token + spacy_tgt_list + tokens_after_tgt
    #print("\n\n========================================\n\n")
    #[print(ele) for ele in spacy_tokens_complete]
    bert_tokens = [word for sublist in spacy_tokens_complete for word in sublist[3]]
    print(bert_tokens)
    bert_pos_tokens = [[ele[1]]*len(ele[3]) for ele in spacy_tokens_complete]
    bert_pos_tokens = ["["+pos+"]" if pos != "[NONE]" else pos for sublist in bert_pos_tokens for pos in sublist]
    print(bert_pos_tokens)
    bert_dep_tokens = [[ele[2]]*len(ele[3]) for ele in spacy_tokens_complete]
    bert_dep_tokens = ["[" + dep + "]" if dep != "[NONE]" else dep for sublist in bert_dep_tokens for dep in sublist]
    print(bert_dep_tokens)
    assert (len(bert_tokens) == len(bert_pos_tokens) == len(bert_dep_tokens))
