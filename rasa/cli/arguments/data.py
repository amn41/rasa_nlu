import argparse
from typing import Text

from rasa.cli.arguments.default_arguments import (
    add_nlu_data_param,
    add_out_param,
    add_data_param,
    add_domain_param,
)
from rasa.shared.constants import DEFAULT_CONVERTED_DATA_PATH, DEFAULT_DOMAIN_PATH


def set_convert_arguments(parser: argparse.ArgumentParser, data_type: Text) -> None:
    """Sets convert command arguments."""
    parser.add_argument(
        "-f",
        "--format",
        default="yaml",
        choices=["json", "yaml"],
        help="Output format the training data should be converted into.",
    )

    add_data_param(parser, required=True, data_type=data_type)

    add_out_param(
        parser,
        default=DEFAULT_CONVERTED_DATA_PATH,
        help_text="File (for `json`) or existing path (for `yaml`) "
        "where to save training data in Rasa format.",
    )

    parser.add_argument("-l", "--language", default="en", help="Language of data.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["watson"],
        help="source file type. Currently only watson is possible",
    )


def set_split_arguments(parser: argparse.ArgumentParser) -> None:
    add_nlu_data_param(parser, help_text="File or folder containing your NLU data.")

    parser.add_argument(
        "--training-fraction",
        type=float,
        default=0.8,
        help="Percentage of the data which should be in the training data.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed to generate the same train/test split.",
    )

    add_out_param(
        parser,
        default="train_test_split",
        help_text="Directory where the split files should be stored.",
    )


def set_validator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fail-on-warnings",
        default=False,
        action="store_true",
        help="Fail validation on warnings and errors. "
        "If omitted only errors will result in a non zero exit code.",
    )
    add_domain_param(parser)
    add_data_param(parser)


def set_migrate_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets migrate command arguments."""
    add_domain_param(parser)

    add_out_param(
        parser,
        default=DEFAULT_DOMAIN_PATH,
        help_text="Path (for `yaml`) where to save migrated domain in Rasa 3.0 format.",
    )
