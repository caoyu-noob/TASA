import language_tool_python
import json
import nltk
from tqdm import tqdm

# evalute number of grammar errors
tool = language_tool_python.LanguageTool('en-US')
files = [
         "../AttackResults/squad_bert_adv_beam9/adv_dataset_beam9.json"]
for file_name in files:
    with open(file_name, "r") as f:
        d = json.load(f)["data"]
    grammar_diffs = []
    contexts = []
    for doc in d:
        for para in doc["paragraphs"]:
            contexts.append(para["context"])
    error_cnt = 0
    word_cnt = 0
    for c in tqdm(contexts):
        errors = tool.check(c)
        miss_cnt = 0
        word_cnt += len(nltk.word_tokenize(c))
        if len(errors) > 0:
            for e in errors:
                if e.ruleId == "UPPERCASE_SENTENCE_START":
                    miss_cnt += 1
        error_cnt += (len(errors) - miss_cnt)
    print(file_name)
    print("number of grammar: ", error_cnt / (word_cnt / 100))