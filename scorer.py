import argparse

import regex as re
from sklearn import metrics


def score_function(gold_keys, predictions, ave_method):
    # ====== Load in data ======
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

    # ====== Loop through data to get stuff needed for calculating metrics ======
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

    # ====== Extract dataset name ======
    dataset_name = str(args.predictions).split("/")[-1]
    dataset_name = re.sub("_predictions.txt", "", dataset_name)

    # ====== Calculate metrics ======
    P = metrics.precision_score(gk_sense_keys, p_sense_keys, average=ave_method)
    R = metrics.recall_score(gk_sense_keys, p_sense_keys, average=ave_method)
    F1 = metrics.f1_score(gk_sense_keys, p_sense_keys, average=ave_method)
    prop_correct = (correct_predictions / n_predictions) * 100

    # ====== Print out results ======
    _print(dataset_name, P, R, F1, prop_correct)


def _print(dataset_name, P, R, F1, prop_correct):
    # ====== Print out results ======
    print(f"\n================ Performance for {dataset_name} ================")
    print(f"Precision score:{' ' * (len(dataset_name)+28)}{round(P, 4)}")
    print(f"{'-' * (len(dataset_name)+50)}")
    print(f"Recall score:{' ' * (len(dataset_name)+31)}{round(R, 4)}")
    print(f"{'-' * (len(dataset_name)+50)}")
    print(f"F1 score:{' ' * (len(dataset_name)+35)}{round(F1, 4)}")
    print(f"{'-' * (len(dataset_name)+50)}")
    print(f"Percentage of correct predictions:{' ' * (len(dataset_name)+9)}{round(prop_correct, 2)} %")
    print(f"{'=' * 50}{'=' * len(dataset_name)}\n")

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

    parser.add_argument(
        "--average_method",
        default="micro",
        help="The method used for averaging multi-label performance scores."
    )


    args = parser.parse_args()
    # ====== Run score function ======
    score_function(args.gold_keys, args.predictions, args.average_method)