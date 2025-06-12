import llm_rpg.world_creator.stages as stages
from src.llm_rpg.world_creator.memory import MemoryBank
from src.llm_rpg.world_creator.utils import MessageReciever

from typing import Dict, Tuple


class DungeonMaster():
    '''
    class working as a middle man between players prompts and real prompts.
    Creates good prompts for model, as it have little to no memory of what happends.
    '''
    def __init__(self):
        
        self.stage : stages.Stage = None
        self.reset()
        
        self.stages_dict: Dict[stages.StageType : stages.Stage] = {
            stages.StageType.World_building : stages.WorldBuildingStage(self.memory),
            stages.StageType.Creation_of_world : stages.PlaceCreationStage(self.memory),
            stages.StageType.Creation_of_plot : stages.PlotCreationStage(self.memory),
            stages.StageType.Exploration : stages.ExplorationStage(self.memory),

        }
    
    def start(self):
        return self.refresh()

    def track_last_used(func):
        def wrapper(self, *args, **kwargs):
            self.last_used = (func, args, kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    

    def reset(self):
        '''resets DM to starting state'''
        self.stage = None
        self.memory = MemoryBank()
        self.new_stage = stages.StageType.World_building
        self.last_used = None

    @track_last_used
    def handle_input(self, user_message) -> Tuple[MessageReciever, Tuple[str, Dict]]:
        return (MessageReciever.LLM, self.stage.convert_user_input(user_message))
    
    def update(self, response) ->  Tuple[MessageReciever, str]:
        '''
        Updates current stage, memory and adds information to response

        Can throw WrongResponseError, when LLM returns something that doesn't match stage's .update() policy
        '''
        self.new_stage, response = self.stage.update(response)
        return (MessageReciever.USER, response)
    
    def check_if_refresh(self):
        return self.stage.stage_type != self.new_stage
    
    def refresh(self) -> Tuple[MessageReciever, str | Tuple[str, Dict]]:
        stage :stages.Stage= self.stages_dict[self.new_stage]
        self.stage = stage
        reciever, message = self._refresh(stage)
        return reciever, message 
    
    @track_last_used
    def _refresh(self, new_stage):
        return new_stage.start()

    def accept(self):
        self.stage.save_update()

    def retry(self):
        func, args, kwargs = self.last_used
        return func(self, *args, **kwargs)
        
        
        


