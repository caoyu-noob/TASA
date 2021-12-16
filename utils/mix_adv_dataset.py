import json
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_train_dataset",
        type=str,
        required=True,
        help="The dataset of the original train dataset",
    )
    parser.add_argument(
        "--adv_train_dataset",
        type=str,
        required=True,
        help="The dataset of the adversarial train dataset",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--mix_ratio",
        type=int,
        default=50
    )

    args = parser.parse_args()
    with open(args.orig_train_dataset, "r", encoding="utf-8") as f:
        orig_data = json.load(f)
    with open(args.adv_train_dataset, "r", encoding="utf-8") as f:
        adv_data = json.load(f)
    adv_num = min(adv_data["len"], int(orig_data["len"] * args.mix_ratio / 100))
    all_adv_ids = []
    for doc in adv_data["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                all_adv_ids.append(qa["id"].split("-")[0])
    random.shuffle(all_adv_ids)
    select_adv_ids = set(all_adv_ids[:adv_num])
    adv_id_sample_dict = {}
    for doc in adv_data["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                if select_adv_ids.__contains__(qa["id"].split("-")[0]):
                    qa["answers"] = [qa["answers"][0]]
                    adv_id_sample_dict[qa["id"].split("-")[0]] = {"context": para["context"], "qas": [qa]}
    for di, doc in enumerate(orig_data["data"]):
        added_paragraphs = []
        for pi, para in enumerate(doc["paragraphs"]):
            new_qas = []
            for qa in para["qas"]:
                if qa["id"] not in select_adv_ids:
                    new_qas.append(qa)
                else:
                    added_paragraphs.append(adv_id_sample_dict[qa["id"]])
            orig_data["data"][di]["paragraphs"][pi]["qas"] = new_qas
        orig_data["data"][di]["paragraphs"].extend(added_paragraphs)
    cnt = 0
    for doc in orig_data["data"]:
        for para in doc["paragraphs"]:
            cnt += len(para["qas"])
    with open(args.output_file_name, "w", encoding="utf-8") as f:
        json.dump(orig_data, f)
    print("1")