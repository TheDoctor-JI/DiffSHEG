from enum import Enum

class EnumWithValEq(Enum):
    def __eq__(self, other):
        #Override for enum comparison by value
        if(other is None):
            return False
        else:
            return self.value == other.value