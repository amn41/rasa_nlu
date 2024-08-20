import typing
from typing import (
    Any,
    Dict,
    Text,
    Optional,
)

import structlog

from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from rasa.e2e_test.stub_custom_action import StubCustomAction

structlogger = structlog.get_logger(__name__)


class E2EStubCustomActionExecutor(CustomActionExecutor):
    def __init__(
        self,
        action_name: str,
        action_endpoint: EndpointConfig,
    ):
        """Initializes the e2e stub custom action executor.

        Args:
            action_name: Name of the custom action.
            action_endpoint: The endpoint to execute custom actions.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint
        self.stub_custom_action = self.get_stub_custom_action()

    def get_stub_custom_action(self) -> "StubCustomAction":
        from rasa.e2e_test.stub_custom_action import get_stub_custom_action

        stub_custom_action = get_stub_custom_action(
            self.action_endpoint, self.action_name
        )

        if stub_custom_action:
            return stub_custom_action

        # TODO Update message below with reference to the docs
        raise RasaException(f"Action `{self.action_name}` has not been stubbed.")

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
    ) -> Dict[Text, Any]:
        structlogger.debug(
            "action.e2e_stub_custom_action_executor.run",
            action_name=self.action_name,
        )
        return self.stub_custom_action.as_dict()
