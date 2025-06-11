from typing import List

class MemoryBank():
    def __init__(self):
        self.universe: str= None
        self.place: str= None
        self.time: str= None
        # self.lenght: str=None

        # Place context
        self.map_position: str= None
        self.place_history :List[str]= []
        
        # Plot context
        self.current_plot: str= None
        self.whole_plot: str= None
        self.history: List[str]= []

    