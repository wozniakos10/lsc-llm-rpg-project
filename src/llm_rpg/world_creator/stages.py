from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from src.llm_rpg.world_creator.dungeon_master import DungeonMaster
from src.llm_rpg.world_creator.utils import WrongPromptError


class StageType(Enum):
    World_building = 1
    Creation_of_world = 2
    Creation_of_plot = 3
    Exploration = 4
    Fight = 5
    End = 6

class MessageReciever(Enum):
    USER = 1
    LLM = 2

class Stage(ABC):
    def __init__(self, dungeon_master: DungeonMaster):
        self.stage_type: StageType = None
        self.dm = dungeon_master

    @abstractmethod
    def start(self) -> Tuple[MessageReciever, str]:
        '''function initializing this stage, might be some introduction to what is happening'''
        pass

    @abstractmethod
    def convert_user_input(self, user_input)  -> Tuple[str, Dict]:
        '''converts input given from user in a way that LLM will now what to do with this information, instead having raw data returns 
        - `template` for PromptTemplate
        - `dict`
        '''
        pass

    @abstractmethod
    def update(self, response) -> Tuple[StageType, str]:
        '''function that reacts to what LLM gave in return to input'''
        pass


class WorldBuildingStage(Stage):
    stage_type = StageType.World_building
    template = """
Na podstawie poniższego tekstu wypisz trzy informacje, jeśli można je wywnioskować. 
Wypełnij tylko nawiasy kwadratowe — podaj własne, wywnioskowane określenia. 
Jeśli którejś informacji nie da się ustalić, pomiń ją całkowicie (nie wypisuj jej).
Odpowiedzi ustaw w podany sposób, w osobnych linijkach każda, 

Miejsce akcji: [odpowiedź]
Czas akcji: [odpowiedź]
Uniwersum: [odpowiedź]

Teskt do wywnioskowania:
{question}"""

    def start(self):
        return (MessageReciever.USER,
                "Podaj krótki opis historii, wokół której chciałbyś rozpocząć przygodę RPG! Podaj świat, miejsce i czas akcji, a resztą zajmę się ja.")

    def convert_user_input(self, user_input) -> Tuple[str, Dict]:
        formatting = {'question' : user_input}
        return self.template, formatting
        
    def update(self, response:str) -> Tuple[StageType, str]:
        response_check = response.lower()

        # check if model returns something real, not trash
        if not ('miejsce' in response_check and
            'czas' in response_check and
            'uniwersum' in response_check):
            raise WrongPromptError
        
        # update memory
        memory = self.dm.memory
        for line in response_check.split('\n'):
            if 'miejsce' in line:
                memory.place = line
            elif 'czas' in line:
                memory.time = line
            elif 'uniwersum' in line:
                memory.universe = line

        # check, if something is missing
        missing = []
        for name, elem in zip(['universum', 'miejsce', 'czas'], [memory.universe, memory.place, memory.time, memory.lenght]):
            if elem is None:
               missing.append(name)
        
        if missing:
            return self.stage_type, f"Brakuje opisu {', '.join(missing)}.\n. Spróbuj jeszcze raz."
        
        else:
            return (StageType.Creation_of_world, 
                    f"""Następujące dane zostały pozyskane 
                    miejsce akcji: {memory.place}
                    czas akcji: {memory.time}
                    uniwersum: {memory.universe}""")
