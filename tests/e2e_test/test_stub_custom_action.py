from typing import Dict, Any

import pytest

from rasa.e2e_test.constants import (
    TEST_FILE_NAME,
    TEST_CASE_NAME,
    KEY_STUB_CUSTOM_ACTIONS,
)
from rasa.e2e_test.stub_custom_action import (
    StubCustomAction,
    get_stub_custom_action_key,
    get_stub_custom_action,
)
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def endpoint_config(
    test_file_name: str,
    test_case_name: str,
    stub_data: Dict[str, Any],
    action_test_file_stub: StubCustomAction,
    action_test_case_stub: StubCustomAction,
    action_name_test_file_with_separator: str,
    action_name_test_case_with_separator: str,
) -> EndpointConfig:
    return EndpointConfig(
        url="http://localhost:5055/webhook",
        **{
            TEST_FILE_NAME: test_file_name,
            TEST_CASE_NAME: test_case_name,
            KEY_STUB_CUSTOM_ACTIONS: {
                action_name_test_file_with_separator: action_test_file_stub,
                action_name_test_case_with_separator: action_test_case_stub,
            },
        },
    )


def test_stub_custom_action_from_dict(
    stub_data: Dict[str, Any], action_name_test_file: str
):
    stub_action = StubCustomAction.from_dict(action_name_test_file, stub_data)
    assert stub_action.action_name == action_name_test_file
    assert stub_action.events == stub_data["events"]
    assert stub_action.responses == stub_data["responses"]


def test_stub_custom_action_as_dict(
    stub_data: Dict[str, Any], action_name_test_file: str
):
    stub_action = StubCustomAction.from_dict(action_name_test_file, stub_data)
    assert stub_action.as_dict() == stub_data


def test_get_stub_custom_action_key(
    action_name_test_file_with_separator: str,
    test_file_name: str,
    action_name_test_file: str,
):
    assert (
        get_stub_custom_action_key(test_file_name, action_name_test_file)
        == action_name_test_file_with_separator
    )


def test_get_stub_test_file_custom_action(
    endpoint_config: EndpointConfig,
    stub_data: Dict[str, Any],
    action_name_test_file: str,
):
    stub_action = get_stub_custom_action(endpoint_config, action_name_test_file)
    assert stub_action.action_name == action_name_test_file
    assert stub_action.events == stub_data["events"]
    assert stub_action.responses == stub_data["responses"]


def test_get_stub_test_case_custom_action(
    endpoint_config: EndpointConfig,
    stub_data: Dict[str, Any],
    action_name_test_case: str,
    action_name_test_case_with_separator: str,
):
    stub_action = get_stub_custom_action(endpoint_config, action_name_test_case_with_separator)
    assert stub_action.action_name == action_name_test_case
    assert stub_action.events == stub_data["events"]
    assert stub_action.responses == stub_data["responses"]


def test_get_stub_custom_action_fallback(endpoint_config):
    new_action_name = "new_action"
    stub_action = get_stub_custom_action(endpoint_config, new_action_name)
    assert stub_action is None
