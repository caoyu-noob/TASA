import json
import re
import string

from collections import OrderedDict
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate_adversarial(predictions, labels):
    orig_f1_score = 0.0
    orig_exact_match_score = 0.0
    adv_f1_scores = {}  # Map from original ID to F1 score
    adv_exact_match_scores = {}  # Map from original ID to exact match score
    adv_ids = {}
    all_ids = set()  # Set of all original IDs
    for i in range(len(labels)):
        orig_id = labels[i]["id"].split("-")[0]
        all_ids.add(orig_id)
        ground_truths = labels[i]["answers"]["text"]
        prediction = predictions[i]["prediction_text"]
        cur_exact_match = metric_max_over_ground_truths(exact_match_score,
                                                        prediction, ground_truths)
        cur_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        if orig_id == labels[i]["id"]:
            # This is an original example
            orig_f1_score += cur_f1
            orig_exact_match_score += cur_exact_match
            if orig_id not in adv_f1_scores:
                # Haven't seen adversarial example yet, so use original for adversary
                adv_ids[orig_id] = orig_id
                adv_f1_scores[orig_id] = cur_f1
                adv_exact_match_scores[orig_id] = cur_exact_match
        else:
            # This is an adversarial example
            if (orig_id not in adv_f1_scores or adv_ids[orig_id] == orig_id
                    or adv_f1_scores[orig_id] > cur_f1):
                # Always override if currently adversary currently using orig_id
                adv_ids[orig_id] = labels[i]["id"]
                adv_f1_scores[orig_id] = cur_f1
                adv_exact_match_scores[orig_id] = cur_exact_match
    orig_f1 = 100.0 * orig_f1_score / len(all_ids)
    orig_exact_match = 100.0 * orig_exact_match_score / len(all_ids)
    adv_exact_match = 100.0 * sum(adv_exact_match_scores.values()) / len(all_ids)
    adv_f1 = 100.0 * sum(adv_f1_scores.values()) / len(all_ids)
    return OrderedDict([
      ('orig_exact_match', orig_exact_match),
      ('orig_f1', orig_f1),
      ('adv_exact_match', adv_exact_match),
      ('adv_f1', adv_f1),
    ])