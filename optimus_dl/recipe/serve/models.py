from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "optimus-dl-model"
    messages: list[dict]  # Use dict to allow flexibility or define strict message model
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int | None = Field(default=None, ge=1)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "optimus-dl-model"
    prompt: str | list[str]
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int | None = Field(default=None, ge=1)
    stream: bool = False


class Choice(BaseModel):
    index: int
    text: str
    logprobs: dict | None = None
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[Choice]
    usage: dict | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict | None = None


# Streaming Models


class Delta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatChunkChoice]


class CompletionChunkChoice(BaseModel):
    index: int
    text: str
    logprobs: dict | None = None
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionChunkChoice]
