import json
import nltk
from tqdm import tqdm
from model.ppl_evaluator import PPLEvaluator

# evalute number of grammar errors
# files = ["../data/newsqa_adv/newsqa_dev-fooler.json",
#          "../data/newsqa_adv/newsqa_dev-fooler-bidaf.json",
#          "../data/newsqa_adv/newsqa_dev-T3-BERT.json",
#          "../data/newsqa_adv/newsqa_dev-T3-bidaf-target.json"]
files = ["../AttackResults/squad_bert_without_nouns/adv_dataset.json"]
for file_name in files:
    with open(file_name, "r") as f:
        d = json.load(f)["data"]
    grammar_diffs = []
    contexts = []
    for doc in d:
        for para in doc["paragraphs"]:
            contexts.append(para["context"])
    ppl_evaluator = PPLEvaluator("../../model_hubs/gpt2_small", 8)

    ppls = []
    for i, c in enumerate(tqdm(contexts)):
        ppl = ppl_evaluator.calculate_ppl([c])
        ppls.extend(ppl)
    print(file_name)
    print("Avg PPL: ", sum(ppls) / len(ppls))