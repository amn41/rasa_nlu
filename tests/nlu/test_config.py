import os
from typing import Text, List
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.shared.exceptions import YamlSyntaxException
from rasa.shared.importers import autoconfig
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu import config
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.constants import COMPONENT_INDEX
from rasa.shared.nlu.constants import TRAINABLE_EXTRACTORS
from tests.nlu.utilities import write_file_config


def test_blank_config(blank_config):
    file_config = {}
    f = write_file_config(file_config)
    final_config = config.load(f.name)

    assert final_config.as_dict() == blank_config.as_dict()


def test_invalid_config_json(tmp_path):
    file_config = """pipeline: [pretrained_embeddings_spacy"""  # invalid yaml

    f = tmp_path / "tmp_config_file.json"
    f.write_text(file_config)

    with pytest.raises(YamlSyntaxException):
        config.load(str(f))


def test_default_config_file():
    final_config = config.RasaNLUModelConfig()
    assert len(final_config) > 1


def test_set_attr_on_component():
    _config = RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyTokenizer"},
                {"name": "SpacyFeaturizer"},
                {"name": "DIETClassifier"},
            ],
        }
    )
    idx_classifier = _config.component_names.index("DIETClassifier")
    idx_tokenizer = _config.component_names.index("SpacyTokenizer")

    _config.set_component_attr(idx_classifier, epochs=10)

    assert _config.for_component(idx_tokenizer) == {
        "name": "SpacyTokenizer",
        COMPONENT_INDEX: idx_tokenizer,
    }
    assert _config.for_component(idx_classifier) == {
        "name": "DIETClassifier",
        "epochs": 10,
        COMPONENT_INDEX: idx_classifier,
    }


def test_override_defaults_supervised_embeddings_pipeline():
    builder = ComponentBuilder()

    _config = RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyTokenizer"},
                {"name": "SpacyFeaturizer", "pooling": "max"},
                {
                    "name": "DIETClassifier",
                    "epochs": 10,
                    "hidden_layers_sizes": {"text": [256, 128]},
                },
            ],
        }
    )

    idx_featurizer = _config.component_names.index("SpacyFeaturizer")
    idx_classifier = _config.component_names.index("DIETClassifier")

    component1 = builder.create_component(
        _config.for_component(idx_featurizer), _config
    )
    assert component1.component_config["pooling"] == "max"

    component2 = builder.create_component(
        _config.for_component(idx_classifier), _config
    )
    assert component2.component_config["epochs"] == 10
    assert (
        component2.defaults["hidden_layers_sizes"].keys()
        == component2.component_config["hidden_layers_sizes"].keys()
    )


def config_files_in(config_directory: Text):
    return [
        os.path.join(config_directory, f)
        for f in os.listdir(config_directory)
        if os.path.isfile(os.path.join(config_directory, f))
    ]


@pytest.mark.parametrize(
    "config_file",
    config_files_in("data/configs_for_docs") + config_files_in("docker/configs"),
)
async def test_train_docker_and_docs_configs(
    config_file: Text, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(autoconfig, "_dump_config", Mock())
    importer = RasaFileImporter(config_file=config_file)
    imported_config = importer.get_config()

    loaded_config = config.load(imported_config)

    assert len(loaded_config.component_names) > 1
    assert loaded_config.language == imported_config["language"]


# TODO: This should be tested by a validation component
@pytest.mark.parametrize(
    "config_path, data_path, expected_warning_excerpts",
    [
        (
            "data/test_config/config_supervised_embeddings.yml",
            "data/examples/rasa",
            ["add a 'ResponseSelector'"],
        ),
        (
            "data/test_config/config_spacy_entity_extractor.yml",
            "data/test/duplicate_intents_yaml/demo-rasa-intents-2.yml",
            [f"add one of {TRAINABLE_EXTRACTORS}"],
        ),
        (
            "data/test_config/config_crf_no_regex.yml",
            "data/test/duplicate_intents_yaml/demo-rasa-intents-2.yml",
            ["training data with regexes", "include a 'RegexFeaturizer'"],
        ),
        (
            "data/test_config/config_crf_no_regex.yml",
            "data/test/lookup_tables/lookup_table.json",
            ["training data consisting of lookup tables", "add a 'RegexFeaturizer'"],
        ),
        (
            "data/test_config/config_spacy_entity_extractor.yml",
            "data/test/lookup_tables/lookup_table.json",
            [
                "add a 'DIETClassifier' or a 'CRFEntityExtractor' "
                "with the 'pattern' feature"
            ],
        ),
        (
            "data/test_config/config_crf_no_pattern_feature.yml",
            "data/test/lookup_tables/lookup_table.yml",
            "your NLU pipeline's 'CRFEntityExtractor' does not include "
            "the 'pattern' feature",
        ),
        (
            "data/test_config/config_crf_no_synonyms.yml",
            "data/test/synonyms_only.yml",
            ["add an 'EntitySynonymMapper'"],
        ),
        (
            "data/test_config/config_embedding_intent_response_selector.yml",
            "data/test/demo-rasa-composite-entities.yml",
            ["include either 'DIETClassifier' or 'CRFEntityExtractor'"],
        ),
    ],
)
def test_validate_required_components_from_data(
    config_path: Text, data_path: Text, expected_warning_excerpts: List[Text]
):
    loaded_config = config.load(config_path)
    # trainer = Trainer(loaded_config)
    # training_data = rasa.shared.nlu.training_data.loading.load_data(data_path)
    # with pytest.warns(UserWarning) as record:
    #     components.validate_required_components_from_data(
    #         trainer.pipeline, training_data
    #     )
    # assert len(record) == 1
    # assert all(
    #     [excerpt in record[0].message.args[0]] for excerpt in expected_warning_excerpts
    # )
