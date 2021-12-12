import argparse

from bidaf.train import train_model
from bidaf.squad_qa import SquadQAReader
from allennlp.common import Params

parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
parser.add_argument(
    "--config_file",
    type=str,
    default=None,
    help="The config file to train the BiDAF model.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="The path to save the results",
)
parser.add_argument(
    "--train_file",
    type=str,
    default=None,
    help="The file path for training",
)
parser.add_argument(
    "--dev_file",
    type=str,
    default=None,
    help="The file path for validation",
)
parser.add_argument(
    "--cache_file",
    type=str,
)
parser.add_argument(
    "--vocab_file",
    type=str,
)
parser.add_argument(
    "--passage_length_limit",
    type=int,
    default=None
)
parser.add_argument(
    "--num_gradient_accumulation_steps",
    type=int,
    default=None
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None
)
parser.add_argument(
    "--do_predict",
    action="store_true",
    help="whether do prediction using existed model"
)
parser.add_argument(
    "--model_path",
    default=None,
    type=str,
    help="The path for the loaded model"
)
parser.add_argument(
    "--prediction_file",
    default=None,
    type=str,
    help="The path for the prediction dict file"
)


args = parser.parse_args()
config_file = args.config_file
serialization_dir = args.save_path

params = Params.from_file(config_file, "")
params.params["train_data_path"] = args.train_file
params.params["validation_data_path"] = args.dev_file
params.params["cache_file"] = args.cache_file
params.params["vocab_file"] = args.vocab_file
if args.passage_length_limit is not None:
    params.params["dataset_reader"]["passage_length_limit"] = args.passage_length_limit
if args.num_gradient_accumulation_steps is not None:
    params.params["trainer"]["num_gradient_accumulation_steps"] = args.num_gradient_accumulation_steps
if args.batch_size is not None:
    params.params["data_loader"]["batch_sampler"]["batch_size"] = args.batch_size
if args.do_predict:
    params.params["do_predict"] = True
else:
    params.params["do_predict"] = False
params.params["model_path"] = args.model_path
train_model(params, args, serialization_dir, force=True)