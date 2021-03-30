import argparse
import re

from torchvision import models
from torchsummary import summary

import torch
from tabulate import tabulate
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer

from utils.dataset import GlossSelectionRecord, _create_features_from_records
from utils.model import BertWSD, forward_gloss_selection
from utils.wordnet import get_glosses

MAX_SEQ_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predictions(model, tokenizer, sentence):
    re_result = re.search(r"\[TGT\](.*)\[TGT\]", sentence)
    if re_result is None:
        print("\nIncorrect input format. Please try again.")
        return

    ambiguous_word = re_result.group(1).strip()
    sense_keys = []
    definitions = []
    for sense_key, definition in get_glosses(ambiguous_word, None).items():
        sense_keys.append(sense_key)
        definitions.append(definition)

    record = GlossSelectionRecord("test", sentence, sense_keys, definitions, [-1]) # target is -1 so it isn't in targets for _create_features_from_records()
    features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                             cls_token=tokenizer.cls_token,
                                             sep_token=tokenizer.sep_token,
                                             cls_token_segment_id=1,
                                             pad_token_segment_id=0,
                                             disable_progress_bar=True)[0]
    # a feature per gloss-sentence pair
    #for feature in features:
    #    print("token ids")
    #    print(feature[0])
    #    print("attention mask")
    #    print(feature[1])
    #    print("segment ids")
    #    print(feature[2])
    #    print("===============================================\n")

    with torch.no_grad(): # no_grad() disables gradient calculation. Used during inference.
        logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE) # a numnber of zeroes equal to the number of senses.
        for i, bert_input in tqdm(list(enumerate(features)), desc="Progress"):
            logits[i] = model.ranking_linear( # inputs the sentence into BERT and calculates logits used for calculating scores
                model.bert(
                    input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                )[1]
            )
        scores = softmax(logits, dim=0)

    return sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True) # zip adds each the lists together into rows [[a1,b1], [a2, b2]]


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "model_dir",
        default=None,
        type=str,
        help="Directory of pre-trained model."
    )
    args = parser.parse_args()

    # Load fine-tuned model and vocabulary
    print("Loading model...")
    model = BertWSD.from_pretrained(args.model_dir)
    #print(model)
    #summary(model, input_size=(2, 768))
    #print(model.tok)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model.to(DEVICE)
    model.eval()

    while True:
        sentence = input("\nEnter a sentence with an ambiguous word surrounded by [TGT] tokens.\n> ")
        predictions = get_predictions(model, tokenizer, sentence)
        if predictions:
            print("\nPredictions:")
            print(tabulate( # tabulate is for printing in a pretty format.
                [[f"{i+1}.", key, gloss, f"{score:.5f}"] for i, (key, gloss, score) in enumerate(predictions)],
                headers=["No.", "Sense key", "Definition", "Score"])
            )
            # for i, (sense_key, definition, score) in enumerate(predictions):
            #     # print(f"  {i + 1:>3}. sense key: {sense_key:<15} score: {score:<8.5f} definition: {definition}")


if __name__ == '__main__':

    main()
