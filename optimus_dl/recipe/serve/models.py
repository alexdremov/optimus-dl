from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "optimus-dl-model"
    messages: List[dict]  # Use dict to allow flexibility or define strict message model
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "optimus-dl-model"
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    stream: bool = False


class Choice(BaseModel):
    index: int
    text: str
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[dict] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[dict] = None


# Streaming Models


class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatChunkChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChatChunkChoice]


class CompletionChunkChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChunkChoice]
