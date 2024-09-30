from abc import ABC


class BaseModel(ABC):
    # ensures the subclass has the name property
    def __init__(self):
        self.name = "base"

    def __str__(self):
        return self.name
