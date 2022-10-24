# TASA: Twin Answer Sentences Attack for Adversarial Context Generation in Question Answering
Implementation for EMNLP 2022 paper TASA by Yu Cao, Dianqi Li, Meng Fang, Tianyi Zhou, Jun Gao, Yibing Zhan and Dacheng 
Tao.

## Environment

You need Python>=3.7.0

I have put some main packages required under `requirements.txt` in the root directory.

Your may still encounter dependency missing errors, please fix them according to the system output.

We use a single 16GB V100 GPU or a single 12GB RTX 3080Ti GPU in our experiments. At least 12GB GPU memory is needed.

## Preparation

#### 1 You need to download the following models
1. [USE model](https://tfhub.dev/google/universal-sentence-encoder/4), put it under `USE_PATH`
2. [Small size GPT2 model](https://huggingface.co/gpt2/), put it under `GPT2_PATH`
3. [BERT base uncased model](https://huggingface.co/bert-base-uncased), put it under `BERT_PATH`
4. [SpanBERT large cased model](https://huggingface.co/SpanBERT/spanbert-large-cased), put it under `SPANBERT_PATH`
5. [RoBerta base model](https://huggingface.co/roberta-base), put it under `ROBERTA_PATH`
6. [GLoVe 6B 100d embedding](https://nlp.stanford.edu/data/glove.6B.zip), put it under `GLOVE_PATH`

#### 2 Download QA datasets 
Put them under `./data/DATASET_NAME`, an example is given in `./data/squad/`, where 
you need to edit a python file `DATASET_NAME.py` as the dataloader. In `squad.py` we use `dev-v1.1.json` for
both training and dev sets.

For datasets from [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019), including NewsQA, Natural Questions, 
HotpotQA, and TriviaQA, using `./utility_scripts/convert_mrqa_to_squad.py` to convert these datasets into SQuAD format.

#### 3 Train a sample answerable determine model

Use `utility_scrips/get_no_answer_dataset.py` to obtain training samples with unanswerable samples for the dataset.
You will get two JSON files named `DATASET_NAME_TRAIN.json_no_answer` and `DATASET_NAME_DEV.json_no_answer`, using 
them to create a new directory `./data/DATASET_NAME_NO_ANSWER` as the dataset path for training.

Then train the RoBerta using the following command
```
python train_squad.py \
--model_name_or_path ROBERTA_PATH \
--dataset_name ./data/DATASET_NAME_NO_ANSWER \
--output_dir DETERMINE_MODEL_PATH \
--version_2_with_negative 1 \
```
Obtain the determine model under `DETERMINE_MODEL_PATH`

#### 4 Train the victim model $F(\cdot)$
Train the BERT model using the following command and obtain the trained BERT under `TRAINED_BERT_PATH`
```
python train_squad.py \
--model_name_or_path BERT_PATH \
--dataset_name ./data/DATASET_NAME \
--output_dir TRAINED_BERT_PATH \
--version_2_with_negative 0 \
```
Train the SpanBERT model using the following command and obtain the trained SpanBERT under `TRAINED_SPANBERT_PATH`
```
python train_squad.py \
--model_name_or_path SPANBERT_PATH \
--max_seq_length 512 \
--do_lower_case 0 \
--learning_rate 2e-5 \
--dataset_name ./data/DATASET_NAME \
--output_dir TRAINED_SPANBERT_PATH \
--version_2_with_negative 0 \
```
Train the BiDAF model using the following command and obtain the trained BiDAF under `TRAINED_BIDAF_PATH`
```
python train_bidaf.py \
--config_file ./bidaf/bidaf.jsonnet \
--save_path TRAINED_BIDAF_PATH \
--train_file ./data/DATASET_NAME/TRAIN_FILE.json \
--dev_file ./data/DATASET_NAME/DEV_FILE.json \
--cache_file ./data/DATASET_NAME/cache.bin \
--vocab_file ./data/TRAINED_BIDAF_PATH/vocabulary \
--passage_length_limit 800 \
--num_gradient_accumulation_steps 2 \
```

#### 5 Get the coreference file for dev set
Using the following command to get the file `COREFERENCE_FILE` containing coreference relationship of the target attack dataset
```
python ./utility_scripts/get_coreference.py \
--input_file ./data/DATASET_NAME/DEV_FILE.json \
--output_file COREFERENCE_FILE \
```

#### 6 Get named entity dictionary and POS tag dictionary for current dataset
Using the script `./utility/extact_entities_pos_vocab.py`. You can set the input dataset JSON files within the script,
and the output file paths to get `ENT_DICT` and `POS_VOCAB_DICT`, which will be used for candidate sampling in attack.

## Attack
Use the follow command to attack the BERT model and obtain the adversarial samples in `BERT_ATTACK_OUTPUT`
```
python TASA.py \
--target_dataset_file ./data/DATASET_NAME/DEV_FILE.json \
--target_model TRAINED_BERT_PATH \
--target_model_type bert \
--output_dir BERT_ATTACK_OUTPUT \
--target_dataset_type squad
--ent_dict_file ENT_DICT \
--coreference_file COREFERENCE_FILE \
--pos_vocab_dict_file POS_VOCAB_DICT \
--USE_model_path USE_PATH \
--ppl_model_path GPT2_PATH \
--determine_model_path DETERMINE_MODEL_PATH \
--beam_size 5 \
```

Use the follow command to attack the SpanBERT model and obtain the adversarial samples in `SPANBERT_ATTACK_OUTPUT`
```
python TASA.py \
--target_dataset_file ./data/DATASET_NAME/DEV_FILE.json \
--target_model TRAINED_BERT_PATH \
--target_model_type spanbert \
--output_dir SPANBERT_ATTACK_OUTPUT \
--target_dataset_type squad
--ent_dict_file ENT_DICT \
--coreference_file COREFERENCE_FILE \
--pos_vocab_dict_file POS_VOCAB_DICT \
--USE_model_path USE_PATH \
--ppl_model_path GPT2_PATH \
--determine_model_path DETERMINE_MODEL_PATH \
--beam_size 5 \
```

Similarly, use the following command to attack the BiDAF model and obtain the adversarial samples in `BIDAF_ATTACK_OUTPUT`
```
python TASA.py \
--target_dataset_file ./data/DATASET_NAME/DEV_FILE.json \
--target_model TRAINED_BIDAF_PATH \
--target_model_type bidaf \
--output_dir BIDAF_ATTACK_OUTPUT \
--target_dataset_type squad
--ent_dict_file ENT_DICT \
--coreference_file COREFERENCE_FILE \
--pos_vocab_dict_file POS_VOCAB_DICT \
--USE_model_path USE_PATH \
--ppl_model_path GPT2_PATH \
--determine_model_path DETERMINE_MODEL_PATH \
--beam_size 5 \
```

### Test adversarial samples
You an use the following command to test the performance of models on the adversarial samples
```
python train_squad.py \
--model_name_or_path TRAINED_BERT_PATH \
--dataset_name ./data/ADVERSARIAL_DATA \
--output_dir TRAINED_BERT_PATH \
--do_predict \
```
Or
```
python train_squad.py \
--model_name_or_path TRAINED_SPANBERT_PATH \
--dataset_name ./data/ADVERSARIAL_DATA \
--output_dir TRAINED_BERT_PATH \
--max_seq_length 512 \
--do_lower_case 0 \
--do_predict \
```
Or
```
python train_bidaf.py \
--config_file ./bidaf/bidaf.jsonnet \
--save_path debugger_train \
--dev_file ./data/ADVERSARIAL_DATA/ADVERSARIAL.json
--cache_file ./data/ADVERSARIAL_DATA/cache_test.bin
--vocab_file ./TRAINED_BIDAF_PATH/vocabulary
--passage_length_limit 800 \
--do_predict \
--model_path TRAINED_BIDAF_PATH \
--prediction_file squad_bidaf_predicitons.json \
```