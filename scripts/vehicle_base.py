from abc import ABC, abstractmethod

class VehicleBase(ABC):
    def __init__(self, name, x, y, yaw, color = 'k', length = 5, width = 2,
                 safe_length = 8, safe_width = 2.4):
        self.name: str = name
        self.x: float = x
        self.y: float = y
        self.yaw: float = yaw
        self.v: float = 0
        self.length: float = length
        self.width: float = width
        self.color:str = color
        self.safe_length = safe_length
        self.safe_width = safe_width

    @abstractmethod
    def get_box2d(self, tar_offset):
        pass

    @abstractmethod
    def get_safezone(self, tar_offset):
        pass
