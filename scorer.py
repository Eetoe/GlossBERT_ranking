import argparse
from sklearn import metrics



def score_function(gold_keys, predictions):
    # Load gold keys
    gk = open(gold_keys, "r")
    gk_lines = gk.readlines()
    gk_split = [line.split() for line in gk_lines]

    # Load predictions
    preds = open(predictions, "r")
    p_lines = preds.readlines()
    p_split = [line.split() for line in p_lines]

    assert len(gk_split) == len(p_split), "Number of gold keys doesn't match number of predictions!"
    n_predictions = len(p_split)

    correct_predictions = 0
    gk_sense_keys = []
    p_sense_keys = []
    for i in range(n_predictions):
        sent_id_gk = gk_split[i][0]
        sent_id_p = p_split[i][0]
        gk_i = gk_split[i][1]
        p_i = p_split[i][1]

        assert sent_id_gk == sent_id_p, "The order of the sentences have been shuffled. Sort before scoring."

        if gk_i == p_i:
            correct_predictions += 1

        gk_sense_keys.append(gk_i)
        p_sense_keys.append(p_i)

    P = metrics.precision_score(gk_sense_keys, p_sense_keys, average="weighted")
    R = metrics.recall_score(gk_sense_keys, p_sense_keys, average="weighted")
    F1 = metrics.f1_score(gk_sense_keys, p_sense_keys, average="weighted")
    prop_correct = correct_predictions/n_predictions

    print(f"Precision score: {P}")
    print(f"Recall score: {R}")
    print(f"F1 score: {F1}")
    print(f"Proportion of correct predictions: {prop_correct}")

if __name__ == "__main__":
    # ====== Set up args ======
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold_keys",
        required=True,
        help="Path to the file containing the gold keys."
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the file containing the model predictions."
    )

    args = parser.parse_args()

    # ====== Run score function ======
    score_function(args.gold_keys, args.predictions)