import aiohttp

import copy
import logging
import structlog

from typing import Text, Dict, Any, Optional

from rasa.core import constants
from rasa.core.channels import UserMessage
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


class RasaNLUHttpInterpreter:
    """Allows for an HTTP endpoint to be used to parse messages."""

    def __init__(self, endpoint_config: Optional[EndpointConfig] = None) -> None:
        """Initializes a `RasaNLUHttpInterpreter`."""
        self.session = aiohttp.ClientSession()
        if endpoint_config:
            self.endpoint_config = endpoint_config
        else:
            self.endpoint_config = EndpointConfig(constants.DEFAULT_SERVER_URL)

    async def parse(self, message: UserMessage) -> Dict[Text, Any]:
        """Parse a text message.

        Return a default value if the parsing of the text failed.
        """
        default_return = {
            "intent": {INTENT_NAME_KEY: "", "confidence": 0.0},
            "entities": [],
            "text": "",
        }

        result = await self._rasa_http_parse(message.text, message.message_id)
        return result if result is not None else default_return

    async def _rasa_http_parse(
        self, text: Text, message_id: Optional[Text] = None
    ) -> Optional[Dict[Text, Any]]:
        """Send a text message to a running rasa NLU http server.

        Return `None` on failure.
        """
        if not self.endpoint_config or self.endpoint_config.url is None:
            structlogger.error(
                "http.parse.text",
                text=copy.deepcopy(text),
                event_info="No rasa NLU server specified!",
            )
            return None

        params = {
            "token": self.endpoint_config.token,
            "text": text,
            "message_id": message_id,
        }

        if self.endpoint_config.url.endswith("/"):
            url = self.endpoint_config.url + "model/parse"
        else:
            url = self.endpoint_config.url + "/model/parse"

        # noinspection PyBroadException
        try:
            async with self.session.post(url, json=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    response_text = await resp.text()
                    structlogger.error(
                        "http.parse.text.failure",
                        text=copy.deepcopy(text),
                        response_text=copy.deepcopy(response_text),
                    )
                    return None
        except Exception:  # skipcq: PYL-W0703
            # need to catch all possible exceptions when doing http requests
            # (timeouts, value errors, parser errors, ...)
            structlogger.exception(
                "http.parse.text.exception",
                text=copy.deepcopy(text),
            )
            return None
