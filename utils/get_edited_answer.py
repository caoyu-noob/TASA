import json
import pickle
import sys
from metrics.squad.evaluate import f1_score

# input_file, output_file = sys.argv[1], sys.argv[2]

# with open(input_file, "rb") as f:
#     adv_track = pickle.load(f)
# answer_dict = {}
# for track in adv_track:
#     if track.__contains__("edited_answers") and len(track["edited_answers"]) > 0:
#         answer_dict[track["id"]] = track["edited_answers"][0]
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(answer_dict, f)

pred_file, answer_file = sys.argv[1], sys.argv[2]

with open(pred_file, "r", encoding="utf-8") as f:
    pred_dict = json.load(f)
with open(answer_file, "r", encoding="utf-8") as f:
    answer_dict = json.load(f)
f1s = []
for k, v in answer_dict.items():
    new_id = k + "-1"
    if pred_dict.__contains__(new_id):
        pred_answer = pred_dict[new_id]
        f1s.append(f1_score(pred_answer, v))
print(sum(f1s) / len(f1s))
