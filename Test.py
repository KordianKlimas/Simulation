from AI.PPO_Worker import PPOTrainer, ActorCritic, RolloutBuffer
from AI.Training import Trainer
from Game import game
from Enviroment import environment
from AI.Rule_Based import RuleBased
from GameStatistics import gameStatistics
# Initialize environment, model, and PPO trainer

from game_enteties.Attacks import regular_attack, fireball, ice_shard
from game_enteties.Character import player, boss

it_settings = { 
    "map_width": 1024,
    "map_height": 1024,
    "player": {
        "max_hp": 100,
        "min_hp": 0,
        "max_speed": 10,
        "min_speed": 0,
        "max_size": 20,
        "min_size": 0,
       
    },
    "player_attacks": {
        "max_attack_cooldown": 10,
        "min_attack_cooldown": 0,
    },
    "boss": {
        "max_hp": 500,
        "min_hp": 0,
        "max_speed": 5,
        "min_speed": 0,
        "max_size": 30,
        "min_size": 0,
    },
    "attacks": {
        "max_player_projectiles": 10,
        "max_boss_projectiles": 10,
        "max_attack_hp": 20,
        "min_attack_hp": 0,
        "max_attack_cooldown": 10,
        "min_attack_cooldown": 0,
        "max_attack_damage": 40,
        "min_attack_damage": 0,
        "max_attack_size": 10,
        "min_attack_size": 0,
        "max_attack_speed": 8,
        "min_attack_speed": 0,
    },
    # References to existing objects
    "attack_objects": {
        "regular_attack": regular_attack,
        "fireball": fireball,
        "ice_shard": ice_shard,
    },
    "character_objects": {
        "player": player,
        "boss": boss,
    },
    # For compatibility with GameStatistics TODO clean up 
    "boss_attacks": {
        "max_player_projectiles": 10,
        "max_boss_projectiles": 10,
        "max_attack_hp": 20,
        "max_attack_cooldown": 10,
        "max_attack_damage": 40,
        "max_attack_size": 10,
        "max_attack_speed": 8,
    },
    "attack_stats": {}
}

env = environment(it_settings["map_height"], it_settings["map_width"])
game_instance = game(env)
game_instance.reset()
bossAI = RuleBased(game_instance, 1)
print("bossAI")
playerAI = RuleBased(game_instance, 0)
print("playerAI")

print()
print(game_instance.state)

# Initialize GameStatistics
gameStats = gameStatistics(game_instance.sDict, it_settings)

# Main loop
clock = 0
debug_prints = 0  # Counter for debug prints

while not game_instance.done:
    # Process the game turn and get the new state
    state, reward, done, boss_action, player_action, current_turn = game_instance.process_turn()

    # Log the current tick, gather statistics, extract the feature vector for GRU 
    gameStats.log_game_state(game_instance.sDict)
    feature_vector = gameStats.extract_feature_vector()

    # Apply AIs
    playerAI.apply_rules()
    bossAI.apply_rules()

    clock += 1  # Increment clock

    if game_instance.current_turn % 1000 == 0:  # Example condition to stop the loop
        if debug_prints < 3:
           gameStats.debug_print_state()
           debug_prints += 1
        break

#print(game_instance.env.attacks)
#print(f"Player Position: ({playerAI.character.x}, {playerAI.character.y}), Boss Position: ({bossAI.character.x}, {bossAI.character.y})")
#print(f"Player Attack: {playerAI.character.attack}, Boss Attack: {playerAI.character.attack}")