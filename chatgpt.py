import os
from typing import Any, List

from erniebot_agent.memory.messages import Message
from openai import AzureOpenAI


class ChatGPT:
    def __init__(
        self,
    ):
        self.client = AzureOpenAI(
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
            api_version="2023-07-01-preview",
            # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment_name = os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo-16k"
        )

    async def chat(
        self,
        messages: List[Message],
        *,
        stream: bool = False,
        **kwargs: Any,
    ):
        """Asynchronously chats with the ERNIE Bot model.

        Args:
            messages (List[Message]): A list of messages.
            stream (bool): Whether to use streaming generation. Defaults to False.
            **kwargs: Keyword arguments, such as `top_p`, `temperature`, `penalty_score`, and `system`.

        Returns:
            If `stream` is False, returns a single message.
            If `stream` is True, returns an asynchronous iterator of message chunks.
        """

        if isinstance(messages[0], Message):
            messages = [item.to_dict() for item in messages]
        if "system" in kwargs:
            messages = [{"role": "system", "content": kwargs.pop("system")}] + messages

        if "response_format" in kwargs:
            kwargs.pop("response_format")

        response = self.client.chat.completions.create(
            model=self.deployment_name, messages=messages, **kwargs
        )
        result = response.choices[0].message

        return result
