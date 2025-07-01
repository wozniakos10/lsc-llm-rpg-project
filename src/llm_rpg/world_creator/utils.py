
from enum import Enum

class WrongResponseError(Exception):
    pass

class MessageReciever(Enum):
    USER = 1
    LLM = 2