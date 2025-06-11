import llm_rpg.world_creator.stages as stages
from src.llm_rpg.world_creator.memory import MemoryBank
from typing import Dict


class DungeonMaster():
    '''
    class working as a middle man between players prompts and real prompts.
    Creates good prompts for model, as it have little to no memory of what happends.
    '''
    def __init__(self):
        self.stages_dict: Dict[stages.StageType : stages.Stage] = {
            stages.StageType.World_building : stages.WorldBuildingStage(),
            stages.StageType.Creation_of_world : None,
            stages.StageType.Creation_of_plot : None,
            stages.StageType.Exploration : None,
            stages.StageType.Fight : None,
            stages.StageType.End : None

        }
        self.stage : stages.Stage = None
        self.reset()
        self.refresh()
    

    def reset(self):
        '''resets DM to starting state'''
        self.stage = None
        self.memory = MemoryBank()
        self.new_stage = stages.StageType.World_building

    def handle_input(self, user_message) -> str:
        return self.stage.convert_user_input(user_message)
    
    def update(self, response) -> str:
        '''
        Updates current stage, memory and adds information to response

        Can throw WrongResponseError, when LLM returns something that doesn't match stage's .update() policy
        '''
        self.new_stage, response = self.stage.update(response)
        return response

    def refresh(self):
        if self.new_stage != self.stage.stage_type:
            stage :stages.Stage= self.stages_dict[self.new_stage]
            reciever, message = stage.start()
            self.stage = stage

            if reciever == stages.MessageReciever.USER:
                pass #TODO
            else:
                pass
        
        
        


