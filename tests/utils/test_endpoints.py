import structlog
from pathlib import Path
from typing import Text, Optional, Union
from unittest.mock import Mock

import pytest
from aioresponses import aioresponses

from rasa.shared.exceptions import FileNotFoundException
from tests.utilities import latest_request, json_of_latest_request
import rasa.utils.endpoints as endpoint_utils


@pytest.mark.parametrize(
    "base, subpath, expected_result",
    [
        ("https://example.com", None, "https://example.com"),
        ("https://example.com/test", None, "https://example.com/test"),
        ("https://example.com/", None, "https://example.com/"),
        ("https://example.com/", "test", "https://example.com/test"),
        ("https://example.com/", "test/", "https://example.com/test/"),
        (
            "http://duckling.rasa.com:8000",
            "/parse",
            "http://duckling.rasa.com:8000/parse",
        ),
        (
            "http://duckling.rasa.com:8000/",
            "/parse",
            "http://duckling.rasa.com:8000/parse",
        ),
    ],
)
def test_concat_url(base, subpath, expected_result):
    assert endpoint_utils.concat_url(base, subpath) == expected_result


def test_warning_for_base_paths_with_trailing_slash():
    test_path = "base/"
    with structlog.testing.capture_logs() as caplog:
        assert endpoint_utils.concat_url(test_path, None) == test_path

    assert len(caplog) == 1
    assert caplog[0]["event"] == "endpoint.concat_url.trailing_slash"
    assert caplog[0]["log_level"] == "debug"


async def test_endpoint_config():
    with aioresponses() as mocked:
        endpoint = endpoint_utils.EndpointConfig(
            "https://example.com/",
            params={"A": "B"},
            headers={"X-Powered-By": "Rasa"},
            basic_auth={"username": "user", "password": "pass"},
            token="mytoken",
            token_name="letoken",
            type="redis",
            port=6379,
            db=0,
            password="password",
            timeout=30000,
        )

        mocked.post(
            "https://example.com/test?A=B&P=1&letoken=mytoken",
            payload={"ok": True},
            repeat=True,
            status=200,
        )

        await endpoint.request(
            "post",
            subpath="test",
            content_type="application/text",
            json={"c": "d"},
            params={"P": "1"},
        )

        r = latest_request(
            mocked, "post", "https://example.com/test?A=B&P=1&letoken=mytoken"
        )

        assert r

        assert json_of_latest_request(r) == {"c": "d"}
        assert r[-1].kwargs.get("params", {}).get("A") == "B"
        assert r[-1].kwargs.get("params", {}).get("P") == "1"
        assert r[-1].kwargs.get("params", {}).get("letoken") == "mytoken"

        # unfortunately, the mock library won't report any headers stored on
        # the session object, so we need to verify them separately
        async with endpoint.session() as s:
            assert s._default_headers.get("X-Powered-By") == "Rasa"
            assert s._default_auth.login == "user"
            assert s._default_auth.password == "pass"


async def test_endpoint_config_with_cafile(tmp_path: Path):
    cafile = "data/test_endpoints/cert.pem"

    with aioresponses() as mocked:
        endpoint = endpoint_utils.EndpointConfig(
            "https://example.com/", cafile=str(cafile)
        )

        mocked.post("https://example.com/", status=200)

        await endpoint.request("post")

        request = latest_request(mocked, "post", "https://example.com/")[-1]

        ssl_context = request.kwargs["ssl"]
        certs = ssl_context.get_ca_certs()
        assert certs[0]["subject"][4][0] == ("organizationalUnitName", "rasa")


async def test_endpoint_config_with_non_existent_cafile(tmp_path: Path):
    cafile = "data/test_endpoints/no_file.pem"

    endpoint = endpoint_utils.EndpointConfig("https://example.com/", cafile=str(cafile))

    with pytest.raises(FileNotFoundException):
        await endpoint.request("post")


def test_endpoint_config_default_token_name():
    test_data = {"url": "http://test", "token": "token"}

    actual = endpoint_utils.EndpointConfig.from_dict(test_data)

    assert actual.token_name == "token"


def test_endpoint_config_custom_token_name():
    test_data = {"url": "http://test", "token": "token", "token_name": "test_token"}

    actual = endpoint_utils.EndpointConfig.from_dict(test_data)

    assert actual.token_name == "test_token"


async def test_request_non_json_response():
    with aioresponses() as mocked:
        endpoint = endpoint_utils.EndpointConfig("https://example.com/")

        mocked.post(
            "https://example.com/test",
            payload="ok",
            content_type="application/text",
            status=200,
        )

        response = await endpoint.request("post", subpath="test")

        assert not response


@pytest.mark.parametrize(
    "filename, endpoint_type",
    [("data/test_endpoints/example_endpoints.yml", "tracker_store")],
)
def test_read_endpoint_config(filename: Text, endpoint_type: Text):
    conf = endpoint_utils.read_endpoint_config(filename, endpoint_type)
    assert isinstance(conf, endpoint_utils.EndpointConfig)


@pytest.mark.parametrize(
    "endpoint_type, cafile",
    [("action_endpoint", "./some_test_file"), ("tracker_store", None)],
)
def test_read_endpoint_config_with_cafile(endpoint_type: Text, cafile: Optional[Text]):
    conf = endpoint_utils.read_endpoint_config(
        "data/test_endpoints/example_endpoints.yml", endpoint_type
    )
    assert conf.cafile == cafile


@pytest.mark.parametrize(
    "filename, endpoint_type",
    [
        ("", "tracker_store"),
        ("data/test_endpoints/example_endpoints.yml", "stuff"),
        ("data/test_endpoints/example_endpoints.yml", "empty"),
        ("/unknown/path.yml", "tracker_store"),
    ],
)
def test_read_endpoint_config_not_found(filename: Text, endpoint_type: Text):
    conf = endpoint_utils.read_endpoint_config(filename, endpoint_type)
    assert conf is None


@pytest.mark.parametrize(
    "value, default, expected_result",
    [
        (None, True, True),
        (False, True, False),
        ("false", True, False),
        ("true", False, True),
    ],
)
def test_bool_arg(
    value: Optional[Union[bool, str]], default: bool, expected_result: bool
):
    request = Mock()
    request.args = {}
    if value is not None:
        request.args = {"key": value}
    assert endpoint_utils.bool_arg(request, "key", default) == expected_result


@pytest.mark.parametrize(
    "value, default, expected_result",
    [(None, 0.5, 0.5), (0.5, None, 0.5), ("0.5", 0, 0.5), ("a", 0.5, 0.5)],
)
def test_float_arg(
    value: Optional[Union[float, str]], default: float, expected_result: float
):
    request = Mock()
    request.args = {}
    if value is not None:
        request.args = {"key": value}
    assert endpoint_utils.float_arg(request, "key", default) == expected_result


@pytest.mark.parametrize(
    "value, default, expected_result",
    [(None, 0, 0), (1, 0, 1), ("1", 0, 1), ("a", 0, 0)],
)
def test_int_arg(value: Optional[Union[int, str]], default: int, expected_result: int):
    request = Mock()
    request.args = {}
    if value is not None:
        request.args = {"key": value}
    assert endpoint_utils.int_arg(request, "key", default) == expected_result
