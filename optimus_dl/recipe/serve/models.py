from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message sender (e.g., 'user', 'assistant', 'system').
        content: The content of the message.
    """

    role: str | None = None
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    """Request body for the chat completion API.

    Attributes:
        model: ID of the model to use.
        messages: A list of messages comprising the conversation so far.
        max_tokens: The maximum number of tokens to generate in the chat completion.
        temperature: What sampling temperature to use, between 0 and 2.
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        stream: If set, partial message deltas will be sent.
    """

    model: str = "optimus-dl-model"
    messages: list[dict]  # Use dict to allow flexibility or define strict message model
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int | None = Field(default=None, ge=1)
    stream: bool = False


class CompletionRequest(BaseModel):
    """Request body for the text completion API.

    Attributes:
        model: ID of the model to use.
        prompt: The prompt(s) to generate completions for.
        max_tokens: The maximum number of tokens to generate in the completion.
        temperature: What sampling temperature to use, between 0 and 2.
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        stream: If set, partial message deltas will be sent.
    """

    model: str = "optimus-dl-model"
    prompt: str | list[str]
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int | None = Field(default=None, ge=1)
    stream: bool = False


class Choice(BaseModel):
    """A single completion choice.

    Attributes:
        index: The index of the choice in the list of choices.
        text: The generated text.
        logprobs: Log probabilities of the token choices (optional).
        finish_reason: The reason the model stopped generating tokens.
    """

    index: int
    text: str
    logprobs: dict | None = None
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    """Response object for the text completion API.

    Attributes:
        id: A unique identifier for the completion.
        object: The object type, which is always "text_completion".
        created: The Unix timestamp (in seconds) of when the completion was created.
        model: The model used for completion.
        choices: The list of completion choices.
        usage: Usage statistics for the completion request.
    """

    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[Choice]
    usage: dict | None = None


class ChatChoice(BaseModel):
    """A single chat completion choice.

    Attributes:
        index: The index of the choice in the list of choices.
        message: The generated message.
        finish_reason: The reason the model stopped generating tokens.
    """

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Response object for the chat completion API.

    Attributes:
        id: A unique identifier for the chat completion.
        object: The object type, which is always "chat.completion".
        created: The Unix timestamp (in seconds) of when the chat completion was created.
        model: The model used for the chat completion.
        choices: The list of chat completion choices.
        usage: Usage statistics for the completion request.
    """

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict | None = None


# Streaming Models


class Delta(BaseModel):
    """A partial message delta for streaming responses.

    Attributes:
        role: The role of the message sender.
        content: The content of the message delta.
    """

    role: str | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    """A single chat completion chunk choice.

    Attributes:
        index: The index of the choice in the list of choices.
        delta: The message delta.
        finish_reason: The reason the model stopped generating tokens.
    """

    index: int
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Represents a streamed chunk of a chat completion response.

    Attributes:
        id: A unique identifier for the chat completion.
        object: The object type, which is always "chat.completion.chunk".
        created: The Unix timestamp (in seconds) of when the chat completion was created.
        model: The model used for the chat completion.
        choices: The list of chat completion choices.
    """

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatChunkChoice]


class CompletionChunkChoice(BaseModel):
    """A single text completion chunk choice.

    Attributes:
        index: The index of the choice in the list of choices.
        text: The text chunk.
        logprobs: Log probabilities of the token choices (optional).
        finish_reason: The reason the model stopped generating tokens.
    """

    index: int
    text: str
    logprobs: dict | None = None
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    """Represents a streamed chunk of a text completion response.

    Attributes:
        id: A unique identifier for the completion.
        object: The object type, which is always "text_completion".
        created: The Unix timestamp (in seconds) of when the completion was created.
        model: The model used for the completion.
        choices: The list of completion choices.
    """

    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionChunkChoice]
