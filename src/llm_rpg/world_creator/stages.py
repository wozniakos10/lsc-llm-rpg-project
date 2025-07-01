from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from llm_rpg.world_creator.memory import MemoryBank
from llm_rpg.world_creator.utils import WrongResponseError, MessageReciever


class StageType(Enum):
    World_building = 1
    Creation_of_world = 2
    Creation_of_plot = 3
    Exploration = 4
    Fight = 5
    End = 6


class Stage(ABC):
    def __init__(self, dungeon_master_memory: MemoryBank):
        self.stage_type: StageType = None
        self.memory = dungeon_master_memory

    @abstractmethod
    def start(self) -> Tuple[MessageReciever, str | Tuple[str, Dict]]:
        """function initializing this stage, might be some introduction to what is happening"""
        pass

    @abstractmethod
    def convert_user_input(self, user_input) -> Tuple[str, Dict]:
        """converts input given from user in a way that LLM will now what to do with this information, instead having raw data returns
        - `template` for PromptTemplate
        - `dict`
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, response) -> Tuple[StageType, str]:
        """function that reacts to what LLM gave in return to input"""
        pass

    @abstractmethod
    def save_update(self):
        """after updating, if user is satisfied with what LLM responded, changes (mostly to memory) are happening"""
        pass


class WorldBuildingStage(Stage):
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

    def __init__(self, dungeon_master_memory):
        super().__init__(dungeon_master_memory)
        self.changed_memory = {"place": None, "universe": None, "time": None}
        self.stage_type = StageType.World_building

    def start(self):
        return (
            MessageReciever.USER,
            "Podaj krótki opis historii, wokół której chciałbyś rozpocząć przygodę RPG! Podaj świat, miejsce i czas akcji, a resztą zajmę się ja.",
        )

    def convert_user_input(self, user_input) -> Tuple[str, Dict]:
        formatting = {"question": user_input}
        return self.template, formatting

    def update(self, response: str) -> Tuple[StageType, str]:
        response_check = response.lower()

        # check if model returns something real, not trash
        if not (
            "miejsce" in response_check
            and "czas" in response_check
            and "uniwersum" in response_check
        ):
            raise WrongResponseError

        # update memory
        for line, line_check in zip(response.split("\n"), response_check.split("\n")):
            if "miejsce" in line_check:
                self.changed_memory["place"] = (
                    line[line.find(":") + 1 :] if ":" in line else line
                )
            elif "czas" in line_check:
                self.changed_memory["time"] = (
                    line[line.find(":") + 1 :] if ":" in line else line
                )
            elif "uniwersum" in line_check:
                self.changed_memory["universe"] = (
                    line[line.find(":") + 1 :] if ":" in line else line
                )

        # check, if something is missing
        missing = []
        for name, elem in zip(
            ["universum", "miejsce", "czas"], ["universe", "place", "time"]
        ):
            if self.changed_memory[elem] is None:
                missing.append(name)

        if missing:
            return (
                self.stage_type,
                f"Brakuje opisu {', '.join(missing)}.\n. Spróbuj jeszcze raz.",
            )

        else:
            return (
                StageType.Creation_of_world,
                f"""Następujące dane zostały pozyskane 
                    miejsce akcji: {self.changed_memory["place"]}
                    czas akcji: {self.changed_memory["time"]}
                    uniwersum: {self.changed_memory["universe"]}""",
            )

    def save_update(self):
        for key, value in self.changed_memory.items():
            if value is not None:
                setattr(self.memory, key, value)


class PlaceCreationStage(Stage):
    template = """
Scena musi zawierać te informacje:
{place} 
{universe}.
Scena musi nadawać się do fabuły gry RPG Dungeons and Dragons
Opisz nie wielki obszar, opisz scenerię.
Nie generuj instrukcji jak wykonać taki świat.
Nie opisuj bohaterów. Nie powtarzaj się. Podaj tylko kilka zdań treściwych.
Twoja odpowiedź ma składać się jedynie z opisu według schematu:

Opis scenerii:
[Tutaj masz napiszać opis i nic więcej]
"""

    def __init__(self, dungeon_master_memory):
        super().__init__(dungeon_master_memory)
        self.stage_type = StageType.Creation_of_world

    def start(self):
        memory = self.memory
        formatting = {"place": memory.place, "universe": memory.universe}
        self.created_place: str = None

        return (MessageReciever.LLM, (self.template, formatting))

    def convert_user_input(self, user_input) -> Tuple[str, Dict]:
        pass

    def update(self, response: str):
        response_check = response.lower()
        if "opis" not in response_check and "scener" not in response_check:
            raise WrongResponseError
        self.created_place = response

        return StageType.Creation_of_plot, response

    def save_update(self):
        self.memory.map_description = self.created_place[:]


class PlotCreationStage(Stage):
    template = """
Jesteś mistrzem gry RPG.
Wygeneruj tylko jedną wersję, bez powtórzeń
Na podstawie poniższych informacji stwórz **jedną krótką fabułę** oraz jasno zdefiniowany cel przygody.
- Świat: {universe}
- Miejsce akcji: {map}

Twoja odpowiedź ma mieć dokładnie dwie części. Nie powtarzaj sekcji historii ani celu więcej niż raz.
Użyj tylko poniższego formatu:
Historia: [tekst]
Cel: [tekst]
"""

    def __init__(self, dungeon_master_memory):
        super().__init__(dungeon_master_memory)
        self.stage_type = StageType.Creation_of_plot

    def start(self):
        memory = self.memory
        formatting = {"map": memory.map_description, "universe": memory.universe}

        self.history = None
        self.quest = None

        return (MessageReciever.LLM, (self.template, formatting))

    def update(self, response: str):
        if not self._read_llm_response(response):
            raise WrongResponseError

        return StageType.Exploration, f"Historia: {self.history}\nCel:{self.quest}\n"

    def convert_user_input(self, user_input) -> Tuple[str, Dict]:
        pass

    def save_update(self):
        self.memory.history = self.history[:]
        self.memory.quest = self.quest[:]

    def _read_llm_response(self, response) -> bool:
        """function to read response and convert it to usable state, check if main aim of asked prompt is achieved"""
        split_plot = response.split("\n")
        history_index = None
        quest_index = None

        history = None
        quest = None
        for line_idx, line in enumerate(split_plot):
            if line.lower().find("historia") == 0:
                if history_index is not None and quest_index is not None:
                    history = split_plot[history_index + 1 : quest_index]
                    quest = split_plot[quest_index + 1 : line_idx]

                    break
                history_index = line_idx
            elif line.lower().find("cel") == 0:
                quest_index = line_idx

        if history is None and history_index is not None and quest_index is not None:
            history = split_plot[history_index:quest_index]
            quest = split_plot[quest_index:line_idx]

        if history is None:
            return False
        self.history = "\n".join(filter(lambda x: x, history))
        self.quest = "\n".join(filter(lambda x: x, quest))

        return True


class ExplorationStage(Stage):
    starting_template = """
Jesteś mistrzem gry RPG.
Na podstawie poniższych informacji stwórz **kilka punktów** które znajduje się w obecnej lokacji i mogą być ciekawe dla graczy:
- Świat: {universe}
- Miejsce akcji: {place}
- Historia: {plot}
- Cel: {quest}

Wymyśl lokację, a następnie kilka misji związanych z tą lokacją.
Jedna misja wokół musi być powiązane z głównym celem misji. Musi być to mały krok do ostatecznego celu.
Wypisz je w liście, jedna pod drugą, tak aby było je łatwo odczytać.
"""

    continuing_template = """
Jesteś mistrzem gry RPG.
Na podstawie poniższej listy misji:
{current_plot}

Wygeneruj ciąg dalszy historii. Nie powtarzaj się, tamte zdarzenia to już przeszłość.
Przy generowaniu najważniejszym punktem jest ta akcja:
{action}
"""
    information_at_the_end = "Podaj akcję, którą wykonuje bohater, oraz czy mu się udała, a ja powiem jego konsekwencję"

    def __init__(self, dungeon_master_memory):
        super().__init__(dungeon_master_memory)

    def start(self):
        memory = self.memory
        formatting = {
            "place": memory.map_description,
            "universe": memory.universe,
            "history": memory.history,
            "quest": memory.quest,
        }
        self.current_plot = None

        return (MessageReciever.LLM, (self.starting_template, formatting))

    def convert_user_input(self, user_input):
        memory = self.memory
        formatting = {"current_plot": memory.current_plot, "action": user_input}
        return (MessageReciever.LLM, (self.starting_template, formatting))

    def update(self, response):
        if "misja" not in response.lower() or "zadanie" not in response.lower():
            raise WrongResponseError()

        self.current_plot = response
        return (MessageReciever.USER, response)

    def save_update(self):
        self.memory.current_plot = self.current_plot[:]
