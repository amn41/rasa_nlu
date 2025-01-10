import pytest
from _pytest.tmpdir import TempPathFactory
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError

import rasa.shared.utils.io
import rasa.utils.io as io_utils


@pytest.mark.parametrize("actual_path", ["", "file.json", "file"])
def test_file_path_validator_with_invalid_paths(actual_path):
    test_error_message = actual_path

    validator = io_utils.file_type_validator([".yml"], test_error_message)

    document = Document(actual_path)
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == test_error_message


@pytest.mark.parametrize("actual_path", ["domain.yml", "lala.yaml"])
def test_file_path_validator_with_valid_paths(actual_path):
    validator = io_utils.file_type_validator([".yml", ".yaml"], "error message")

    document = Document(actual_path)
    # If the path is valid there shouldn't be an exception
    assert validator.validate(document) is None


@pytest.mark.parametrize("user_input", ["", "   ", "\t", "\n"])
def test_non_empty_text_validator_with_empty_input(user_input):
    test_error_message = "enter something"

    validator = io_utils.not_empty_validator(test_error_message)

    document = Document(user_input)
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == test_error_message


@pytest.mark.parametrize("user_input", ["utter_greet", "greet", "Hi there!"])
def test_non_empty_text_validator_with_valid_input(user_input):
    validator = io_utils.not_empty_validator("error message")

    document = Document(user_input)
    # If there is input there shouldn't be an exception
    assert validator.validate(document) is None


def test_create_validator_from_callable():
    def is_valid(user_input) -> None:
        return user_input == "this passes"

    error_message = "try again"

    validator = io_utils.create_validator(is_valid, error_message)

    document = Document("this passes")
    assert validator.validate(document) is None

    document = Document("this doesn't")
    with pytest.raises(ValidationError) as e:
        validator.validate(document)

    assert e.value.message == error_message


def test_empty_directories_are_equal(tmp_path_factory: TempPathFactory):
    dir1 = tmp_path_factory.mktemp("dir1")
    dir2 = tmp_path_factory.mktemp("dir2")

    assert rasa.utils.io.are_directories_equal(dir1, dir2)


def test_directories_are_equal(tmp_path_factory: TempPathFactory):
    dir1 = tmp_path_factory.mktemp("dir1")
    (dir1 / "file.txt").write_text("Hello!")

    dir2 = tmp_path_factory.mktemp("dir2")
    (dir2 / "file.txt").write_text("Hello!")

    assert rasa.utils.io.are_directories_equal(dir1, dir2)


def test_directories_are_equal_sub_dir(tmp_path_factory: TempPathFactory):
    dir1 = tmp_path_factory.mktemp("dir1")
    (dir1 / "dir").mkdir()
    (dir1 / "dir" / "file.txt").write_text("Hello!")

    dir2 = tmp_path_factory.mktemp("dir2")
    (dir2 / "dir").mkdir()
    (dir2 / "dir" / "file.txt").write_text("Hello!")

    assert rasa.utils.io.are_directories_equal(dir1, dir2)


def test_directories_are_equal_different_file_content(
    tmp_path_factory: TempPathFactory,
):
    dir1 = tmp_path_factory.mktemp("dir1")
    (dir1 / "file.txt").write_text("Hello!")

    dir2 = tmp_path_factory.mktemp("dir2")
    (dir2 / "file.txt").write_text("Bye!")

    assert not rasa.utils.io.are_directories_equal(dir1, dir2)


def test_directories_are_equal_extra_file(tmp_path_factory: TempPathFactory):
    dir1 = tmp_path_factory.mktemp("dir1")
    (dir1 / "file.txt").write_text("Hello!")

    dir2 = tmp_path_factory.mktemp("dir2")
    (dir2 / "file.txt").write_text("Hello!")
    (dir2 / "file2.txt").touch()

    assert not rasa.utils.io.are_directories_equal(dir1, dir2)


def test_directories_are_equal_different_file_content_sub_dir(
    tmp_path_factory: TempPathFactory,
):
    dir1 = tmp_path_factory.mktemp("dir1")
    (dir1 / "dir").mkdir()
    (dir1 / "dir" / "file.txt").write_text("Hello!")

    dir2 = tmp_path_factory.mktemp("dir2")
    (dir2 / "dir").mkdir()
    (dir2 / "dir" / "file.txt").write_text("Bye!")

    assert not rasa.utils.io.are_directories_equal(dir1, dir2)
