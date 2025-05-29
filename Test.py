from Simulation.AI.PPO_Worker import PPOTrainer, ActorCritic, RolloutBuffer
from Simulation.AI.Training import Trainer
from Game import game
from Enviroment import environment
from AI.Rule_Based import RuleBased

# Initialize environment, model, and PPO trainer
env = environment(1024, 1024)
game = game(env)
game.reset()
game.process_turn()
bossAI = RuleBased(game,1)
print("bossAI")
playerAI = RuleBased(game,0)
print("playerAI")

game.process_turn()
print()
print(game.state)
while not game.done:
    game.process_turn()
    playerAI.apply_rules()  # Apply player AI rules
    bossAI.apply_rules()  # Apply boss AI rules
    if game.current_turn % 1000 == 0:  # Example condition to stop the loop
        break
print(game.env.attacks)
print(f"Player Position: ({playerAI.character.x}, {playerAI.character.y}), Boss Position: ({bossAI.character.x}, {bossAI.character.y})")
print(f"Player Attack: {playerAI.character.attack}, Boss Attack: {playerAI.character.attack}")