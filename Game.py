import numpy as np
import torch
class game:
    def __init__(self, env):
        from Enviroment import environment
        self.env = env
        self.done = False
        self.boss_action = None
        self.state = []
        self.sDict = self.env.update()
        self.player_action = None
        self.reward = None
        self.current_turn = 0
        self.max_projectiles = round(env.width/5) # 10/2 = Slowest projectile speed as to estimate max projectiles / number of entities

    def reset(self):
        self.state = self.env.resett()
        if self.state is None:
            raise ValueError("Environment reset returned None. Ensure the environment is properly initialized.")
        return self.state
    def get_entity(self,isBoss):
        return self.env.get_env_entity(isBoss)
    def get_environment(self):
        """
        Get the current state of the environment.
        """
        return self.env

    def _flatten_values(self,value):
        if isinstance(value, (int, float)):
            return [float(value)]
        elif isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        elif value is None:
            return []
        else:
            raise TypeError(f"Unsupported value type: {type(value)}")

    def flatten_dict(self):
        vec = []
        for i in range(2):
            if i < len(self.state["entities"]):
                e = self.state["entities"][i]
                for key in ["x", "y", "move", "direction", "hp", "attack", "boss", "size"]:
                    vec.extend(self._flatten_values(e[key]))
            else:
                vec.extend([0.0] * 10)  # padding

        for i in range(self.max_projectiles):
            if i < len(self.state["attacks"]):
                a = self.state["attacks"][i]
                for key in ["x", "y", "damage", "speed", "hp", "size", "direction", "current_cooldown", "cooldown",
                            "attack_id"]:
                    vec.extend(self._flatten_values(a[key]))
            else:
                vec.extend([0.0] * 11)  # padding

        vec.append(float(self.state["score"]))
        vec.append(float(self.state["done"]))
        vec.append(float(self.state["turn"]))

        return torch.tensor(vec, dtype=torch.float32)

    def process_turn(self):
        state_dict = self.env.update()
        self.sDict = state_dict
        self.state = state_dict
        self.state = self.flatten_dict()
        self.boss_action = [entity["attack"] for entity in state_dict.get("entities", []) if entity.get("boss") == 1]
        self.player_action = [entity["attack"] for entity in state_dict.get("entities", []) if entity.get("boss") != 1]
        self.reward = state_dict.get("score", 0)
        self.done = state_dict.get("done", 0)
        self.current_turn = state_dict.get("turn", 0)
        return self.state, self.reward, self.done, self.boss_action, self.player_action, self.current_turn,
