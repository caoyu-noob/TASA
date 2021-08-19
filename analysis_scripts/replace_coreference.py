import json
import spacy
import neuralcoref

from tqdm import tqdm

# def get_replacement_and_indices(coreferences, paragraph_indices):
#     replacements, replacements_indices = [], []
#     '''The replacement priority,
#     pronoun(e.g., it)/ noun phrase(the city) will be replaced with named entity if exists
#     otherwise the pronoun will be replaced with noun phrase'''
#     for coref_list, index in tqdm(zip(coreferences, paragraph_indices)):
#         for coref_set in coref_list:
#             '''The coreference set should contain noun so the pronoun can be replaced'''
#             '''In candidate list, the label
#                 0 = the target is a pronoun/single det (e.g. these),
#                 1 = the target is a pronoun$
#                 2 = the target is a named entity
#                 3 = the target is a noun phrase
#                 4 = the target is in other format'''
#             targets = []
#             processed_spans = list(nlp.pipe([c[0] for c in coref_set]))
#             for i, span in enumerate(processed_spans):
#                 if len(span.ents) > 0:
#                     targets.append((coref_set[i], 2))
#                 else:
#                     if len(span) == 1:
#                         if span[0].tag_ in ["PRP", "DT"]:
#                             targets.append((coref_set[i], 0))
#                         elif span[0].tag_ == "PRP$":
#                             targets.append((coref_set[i], 1))
#                         else:
#                             targets.append((coref_set[i], 4))
#                     else:
#                         if span[0].tag_ == "DT":
#                             targets.append((coref_set[i], 3))
#                         else:
#                             targets.append((coref_set[i], 4))
#             targets = sorted(targets, key=lambda x: x[1])
#             label_set = set([x[1] for x in targets])
#             if (
#                     (label_set.__contains__(0) or label_set.__contains__(1)) and
#                     (label_set.__contains__(2) or label_set.__contains__(3))):
#                 replacements.append(targets)
#                 replacements_indices.append(index)
#     return replacements, replacements_indices

def get_all_answer_positions(qas):
    answer_set = set()
    for qa in qas:
        for answer in qa["answers"]:
            answer_set.add((answer["answer_start"], answer["answer_start"] + len(answer["text"])))
    return answer_set

def get_replacements(mentions, all_answer_pos):
    targets = []
    for mention in mentions:
        if len(mention.ents) > 0:
            targets.append((mention, 2))
        else:
            if len(mention) == 1:
                if mention[0].tag_ in ["PRP", "DT"]:
                    targets.append((mention, 0))
                elif mention[0].tag_ == "PRP$":
                    targets.append((mention, 1))
                else:
                    targets.append((mention, 4))
            else:
                if mention[0].tag_ == "DT":
                    targets.append((mention, 3))
                else:
                    targets.append((mention, 4))
    label_set = set([x[1] for x in targets])
    targets = sorted(targets, key=lambda x: x[1])
    targets_pos = [(t[0].start_char, t[0].end_char) for t in targets]
    if (
                        (label_set.__contains__(0) or label_set.__contains__(1)) and
                        (label_set.__contains__(2) or label_set.__contains__(3))):
        '''replacement overlapping with an answer is not acceptable'''
        for target_pos in targets_pos:
            for answer_pos in all_answer_pos:
                if (answer_pos[0] >= target_pos[0] and answer_pos[1] <= target_pos[1]) or \
                        (answer_pos[0] <= target_pos[0] and answer_pos[1] > target_pos[0]) or \
                        (answer_pos[0] < target_pos[1] and answer_pos[1] >= target_pos[1]):
                    return None
        return targets
    return None

def replace_coreferences(paragraph_text, replacements, offsets):
    replace_targets, replace_ne, replace_noun = [], {}, {}
    for replacement in replacements:
        if replacement[1] in [0, 1]:
            replace_targets.append(replacement)
        elif replacement[1] == 2:
            if not replace_ne.__contains__(replacement[0].text):
                replace_ne[replacement[0].text] = 0
            replace_ne[replacement[0].text] += 1
        elif replacement[1] == 3:
            if not replace_noun.__contains__(replacement[0].lower_):
                replace_noun[replacement[0].lower_] = 0
            replace_noun[replacement[0].lower_] += 1
    if len(replace_ne) > 0:
        replace_ne = sorted(replace_ne.items(), key=lambda x: x[1], reverse=True)
        candidate = replace_ne[0][0]
    elif len(replace_noun) > 0:
        replace_noun = sorted(replace_noun.items(), key=lambda x: x[1], reverse=True)
        candidate = replace_noun[0][0]
    replace_targets = sorted(replace_targets, key=lambda x: x[0].end_char)
    for target in replace_targets:
        cur_offset, offset_i = 0, 0
        while offset_i < len(offsets):
            if offsets[offset_i][0] > target[0].start_char and offset_i > 0:
                break
            offset_i += 1
        if len(offsets) > 0 and offsets[offset_i - 1][0] < target[0].start_char:
            cur_offset = offsets[offset_i - 1][1]
        else:
            offset_i = 0

        candidate_text = candidate
        if target[1] == 1:
            if candidate_text[-1] == "s":
                candidate_text = candidate_text + "\'"
            else:
                candidate_text = candidate_text + "\'s"
        if target[0].text.isupper():
            candidate_text = candidate_text.capitalize()
        paragraph_text = paragraph_text[:target[0].start_char + cur_offset] + candidate_text + \
                         paragraph_text[target[0].end_char + cur_offset:]
        new_offset = cur_offset + len(candidate_text) - len(target[0].text)
        offsets.insert(offset_i, [target[0].end_char, new_offset])
        for i in range(offset_i + 1, len(offsets)):
            offsets[i][1] += (len(candidate_text) - len(target[0].text))
    return paragraph_text, offsets, len(replace_targets)

def correct_answer(qas, offsets, paragraph_text):
    for i, qa in enumerate(qas):
        new_answers = []
        for j, answer in enumerate(qa["answers"]):
            answer_start = answer["answer_start"]
            answer_text = answer["text"]
            cur_offset = 0
            m = 0
            while m < len(offsets):
                if offsets[m][0] > answer_start and m > 0:
                    break
                m += 1
            if offsets[m-1][0] < answer_start:
                cur_offset = offsets[m-1][1]
            assert paragraph_text[answer_start + cur_offset: answer_start + cur_offset + len(answer_text)] == answer_text
            new_answers.append({"answer_start": answer_start + cur_offset, "text": answer["text"]})
        qas[i]["answers"] = new_answers
    return qas

nlp = spacy.load("en_core_web_md")
neuralcoref.add_to_pipe(nlp)
with open("../data/squad/dev-v1.1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

replace_cnt = 0
for doc_i, doc in enumerate(tqdm(data["data"])):
    for p_i, paragraph in enumerate(tqdm(doc["paragraphs"])):
        processed = nlp(paragraph["context"])
        if processed._.has_coref:
            offsets = []
            paragraph_text = processed.text
            all_answer_pos = sorted(get_all_answer_positions(paragraph["qas"]))
            for cluster in processed._.coref_clusters:
                replacements = get_replacements(cluster.mentions, all_answer_pos)
                if replacements:
                    paragraph_text, offsets, cnt = replace_coreferences(paragraph_text, replacements, offsets)
                    replace_cnt += cnt
            if len(offsets) > 0:
                new_qas = correct_answer(paragraph["qas"], offsets, paragraph_text)
                data["data"][doc_i]["paragraphs"][p_i]["context"] = paragraph_text
                data["data"][doc_i]["paragraphs"][p_i]["qas"] = new_qas

with open("../corefs.json", "r", encoding="utf-8") as f:
    corefs = json.load(f)


coreferences = corefs['corefs']
paragraph_indices = corefs['index']
replacements, replacements_indices = get_replacement_and_indices(coreferences, paragraph_indices)



print("111")