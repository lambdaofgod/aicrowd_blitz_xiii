import argparse
import pathlib
import shutil

from deepsense_vision.helpers import get_timestamp_string
from deepsense_vision.io import load_yaml, save_yaml
from deepsense_vision.models import MODEL_TYPES, Model
from deepsense_vision.models.base_config import BaseConfig
from deepsense_vision.neptune_manager import NeptuneManager


def run_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        required=True,
        choices=MODEL_TYPES,
        help="Select model type.",
    )
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to yaml file with model config.",
    )
    parser.add_argument(
        "--experiment_dirpath",
        help="Path to experiment directory. "
        "If not set, default value is taken from model config.",
    )
    parser.add_argument(
        "--train_metadata_path",
        help="Path to training metadata file. "
        "If not set, default value is taken from model config.",
    )
    parser.add_argument(
        "--valid_metadata_path",
        help="Path to validation metadata file. "
        "If not set, default value is taken from model config.",
    )
    parser.add_argument(
        "--neptune_project_name",
        help="Set your project name to create experiment in neptune. "
        "If not set, default value is taken from model config. "
        "If omitted completely, neptune is ignored.",
    )
    parser.add_argument(
        "--neptune_experiment_name",
        help="Name used as experiment name in neptune. "
        "If not set, default value is taken from model config.",
    )
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="Whether to prepend timestamp to the experiment folder name.",
    )
    args = parser.parse_args()
    return args


def setup_config(args: argparse.Namespace, config: dict) -> None:
    if args.experiment_dirpath:
        config["experiment_dirpath"] = args.experiment_dirpath
    if args.train_metadata_path:
        config["datasets"]["train"]["metadata_path"] = args.train_metadata_path
    if args.valid_metadata_path:
        config["datasets"]["valid"]["metadata_path"] = args.valid_metadata_path
    if args.neptune_project_name:
        config["neptune_project_name"] = args.neptune_project_name
    if args.neptune_experiment_name:
        config["neptune_experiment_name"] = args.neptune_experiment_name
    if args.timestamped or config.get("timestamped", None):
        timestamp = get_timestamp_string()
        experiment_dirpath = pathlib.Path(config["experiment_dirpath"])
        experiment_dirpath = experiment_dirpath.parent / (
            timestamp + "_" + experiment_dirpath.name
        )
        config["experiment_dirpath"] = str(experiment_dirpath)


def setup_experiment_directory(
    model_config: BaseConfig,
) -> None:
    experiment_dirpath = pathlib.Path(model_config.experiment_dirpath)
    print(f"Creating experiment dirpath {experiment_dirpath}...")
    experiment_dirpath.mkdir(parents=True)
    save_yaml(
        data=model_config.to_dict(),
        filepath=str(experiment_dirpath / "config.json"),
    )
    src_train_metadata_path = pathlib.Path(model_config.datasets.train.metadata_path)
    dst_train_metadata_filename = "train_metadata" + src_train_metadata_path.suffix
    dst_train_metadata_path = experiment_dirpath / dst_train_metadata_filename
    shutil.copy(
        src=str(src_train_metadata_path),
        dst=str(dst_train_metadata_path),
    )
    src_valid_metadata_path = pathlib.Path(model_config.datasets.valid.metadata_path)
    dst_valid_metadata_filename = "train_metadata" + src_valid_metadata_path.suffix
    dst_valid_metadata_path = experiment_dirpath / dst_valid_metadata_filename
    shutil.copy(
        src=str(src_valid_metadata_path),
        dst=str(dst_valid_metadata_path),
    )
    print(f"Experiment dirpath {experiment_dirpath} created successfully!")


def main_train(args: argparse.Namespace) -> None:
    config = load_yaml(args.config_path)
    setup_config(args, config)
    model = Model(model_type=args.model_type)
    model_config = model.parse_config(config)
    print(model_config)
    setup_experiment_directory(
        model_config=model_config,
    )
    datasets = model.get_datasets(config=model_config)
    train_dataset = datasets["train"]
    valid_dataset = datasets["valid"]
    with NeptuneManager(
        neptune_project_name=model_config.neptune_project_name,
        experiment_name=model_config.neptune_experiment_name,
        experiment_params=model_config.to_dict(),
    ) as neptune_manager:
        model.train(
            config=model_config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            neptune_manager=neptune_manager,
        )


def main() -> None:
    args = run_argparse()
    main_train(args)


if __name__ == "__main__":
    main()
