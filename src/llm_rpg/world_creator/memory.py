from typing import List

class MemoryBank():
    def __init__(self):
        self.universe: str= None
        self.place: str= None
        self.time: str= None
        # self.lenght: str=None

        # Place context
        self.map_description: str=None
        self.map_position: str= None
        
        # Plot context
        self.current_plot: str= None
        self.history: str= None
        self.quest :str = None

    def to_string(self, place_context=False, plot_context=False) -> str:
        '''
        returns memory from what players explored so far.
        Vanilla return is just basic information, but plot and context can be added for better 
        '''
        
        result = [('uniwersum', self.universe),
                  ('miejsce akcji', self.place),
                  ('czas akcji', self.time)]
        
        if place_context:
            result.append(('lokalizacja', self.map_description))
            result.append(('gracze znajdują się obecnie w', self.map_position))


        if plot_context:
            result.append(('ostatnie wydarzenie', self.current_plot))
            result.append(('cała historia to', self.history))
            result.append(('celem przygody jest', self.quest))

        return '\n'.join(map(lambda x: f"{x[0]} : {x[1]}", result))

    