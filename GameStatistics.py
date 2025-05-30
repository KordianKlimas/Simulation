import numpy as np
import json

class gameStatistics:
    def __init__(self, game_state, it_settings):
        """Initializes session, clock, data storage, and gathers all game variables"""
        self.current_session_id = 0
        self.logged_data_history = []  # a list of sessions, each session is a list of tick records
        self._start_new_session_list()
        self.log_game_state(game_state)
        self.it_settings = it_settings

    def _start_new_session_list(self):
        """Creates a new session list in the history matrix."""
        self.logged_data_history.append([])

    def start_new_session(self):
        """Increments session ID and starts a new session in the matrix."""
        self.current_session_id += 1
        self._start_new_session_list()

    def log_game_state(self, game_state):
        """Logs data for the current clock tick, advancing the clock.
        Auto-starts session 1 if no session has been explicitly started.
        """
        if self.current_session_id == 0 and len(self.logged_data_history[0]) == 0:
            self.start_new_session()  # Automatically start the first session

        tick_record = {
            "session_id": self.current_session_id,
            "tick": game_state.get("current_turn"),
            "data": game_state
        }
        self.logged_data_history[self.current_session_id].append(tick_record)

    def extract_feature_vector(self, interval_ticks=100, dodge_distance=20):
        """
        Extracts a feature vector from the latest game state for GRU input.
        All values are standardized using self.it_settings.
        Uses logged_data_history for previous ticks.
        """
        iv = self.it_settings
        session = self.logged_data_history[self.current_session_id]
        if not session:
            return np.zeros(10, dtype=np.float32)  # Return a zero vector if no data

        current_tick_record = session[-1]
        current_game_state = current_tick_record["data"]
        
        previous_game_state = session[-2]["data"] if len(session) > 1 else None

        # --- Apply dodge detection before stats extraction ---
        self.check_player_dodge(current_game_state, previous_game_state, dodge_distance=dodge_distance)

        fireball_obj = self.it_settings["attack_objects"]["fireball"]
        ice_shard_obj = self.it_settings["attack_objects"]["ice_shard"]

        # --- 1. Stats (aggregated over intervals, e.g., 5s, 20s, whole game) ---
        (
            player_hp_change,
            boss_hp_change,
            player_dodge_rate,
            boss_attack_frequency,
            number_of_attacks_prevented_by_player,
            time_since_boss_last_took_damage_from_player
        ) = self.calculate_interval_stats(interval_ticks)

        max_boss_proj = iv["boss_attacks"].get("max_boss_projectiles", 1)
        max_player_proj = iv["boss_attacks"].get("max_player_projectiles", 1)

        # --- 2. Current State ---
        active_player_projectile_count = sum(1 for a in current_game_state.get("attacks", []) if a.get("boss", 0) == 0) / iv["boss_attacks"].get("max_player_projectiles", 1)
        active_boss_projectile_count = (
            sum(1 for a in current_game_state.get("attacks", []) if a.get("boss", 0) == 1) / iv["boss_attacks"].get("max_boss_projectiles")
            if iv["boss_attacks"].get("max_boss_projectiles") else 0.0
        )

        # --- 3. Player ---
        player_x_position = current_game_state.get("player", {}).get("x", 0) / iv["map_width"]
        player_y_position = current_game_state.get("player", {}).get("y", 0) / iv["map_height"]
        player_hp_current = current_game_state.get("player", {}).get("hp", 0) / iv["player"].get("max_hp", 1)
        player_direction_x = current_game_state.get("player", {}).get("direction", [0, 0])[0]
        player_direction_y = current_game_state.get("player", {}).get("direction", [0, 0])[1]
        player_is_moving_flag = 1 if current_game_state.get("player", {}).get("move", [0, 0]) != [0, 0] else 0
        player_attack_cooldown_progress = current_game_state.get("player", {}).get("attack_cooldown", 0) / iv["player_attacks"].get("max_attack_cooldown", 1)

        # --- 4. Boss ---
        boss_x_position = current_game_state.get("boss", {}).get("x", 0) / iv["map_width"]
        boss_y_position = current_game_state.get("boss", {}).get("y", 0) / iv["map_height"]
        boss_hp_current = current_game_state.get("boss", {}).get("hp", 0) / iv["boss"].get("max_hp", 1)
        boss_direction_x = current_game_state.get("boss", {}).get("direction", [0, 0])[0]
        boss_direction_y = current_game_state.get("boss", {}).get("direction", [0, 0])[1]
        boss_is_moving_flag = 1 if current_game_state.get("boss", {}).get("move", [0, 0]) != [0, 0] else 0
        boss_ability_1_cooldown_progress = fireball_obj.current_cooldown / fireball_obj.cooldown if fireball_obj.cooldown else 0.0
        boss_ability_2_cooldown_progress = ice_shard_obj.current_cooldown / ice_shard_obj.cooldown if ice_shard_obj.cooldown else 0.0

        # --- 5. Player's Bullets ---
        player_proj_features = []
        # --- 6. Boss Projectile Features ---
        boss_proj_features = []

        # --- Parse current's state attacks ---
        attacks = current_game_state.get("attacks", [])
        attack_stats = iv.get("attack_stats", {})
        boss_attacks = iv.get("boss_attacks", {})
        max_attack_speed = boss_attacks.get("max_attack_speed", 1)
        max_attack_size = boss_attacks.get("max_attack_size", 1)
        max_attack_damage = boss_attacks.get("max_attack_damage", 1)
        max_attack_hp = boss_attacks.get("max_attack_hp", 1)
        for attack in attacks:
            aid = attack.get("attack_id", 0)
            # Use per-attack stats if available, else global
            max_hp = attack_stats.get(aid, {}).get("max_hp", max_attack_hp)
            max_speed = attack_stats.get(aid, {}).get("max_speed", max_attack_speed)
            max_size = attack_stats.get(aid, {}).get("max_size", max_attack_size)
            max_damage = attack_stats.get(aid, {}).get("max_damage", max_attack_damage)

            if attack.get("boss", 0) == 0:
                # Player projectile
                player_proj_features.append([
                    aid,
                    attack.get("x", 0) / iv["map_width"],
                    attack.get("y", 0) / iv["map_height"],
                    attack.get("direction", [0, 0])[0],
                    attack.get("direction", [0, 0])[1],
                    attack.get("speed", 0) / max_speed if max_speed else 0.0,
                    attack.get("size", 0) / max_size if max_size else 0.0,
                    attack.get("damage", 0) / max_damage if max_damage else 0.0,
                ])
            else:
                # Boss projectile
                boss_proj_features.append([
                    aid,
                    attack.get("x", 0) / iv["map_width"],
                    attack.get("y", 0) / iv["map_height"],
                    attack.get("direction", [0, 0])[0],
                    attack.get("direction", [0, 0])[1],
                    attack.get("speed", 0) / max_speed if max_speed else 0.0,
                    attack.get("size", 0) / max_size if max_size else 0.0,
                    attack.get("hp", 0) / max_hp if max_hp else 0.0,
                    attack.get("damage", 0) / max_damage if max_damage else 0.0,
                ])

        # Pad/cut projectiles to fixed length for GRU input
        player_proj_features = (player_proj_features + [[0]*9]*max_player_proj)[:max_player_proj]
        boss_proj_features = (boss_proj_features + [[0]*10]*max_boss_proj)[:max_boss_proj]

        # --- Compose feature vector ---
        feature_vector = [
            # 1. Stats 
            player_hp_change,
            boss_hp_change,
            player_dodge_rate,
            boss_attack_frequency,
            number_of_attacks_prevented_by_player,
            time_since_boss_last_took_damage_from_player,
            active_player_projectile_count,
            active_boss_projectile_count,

            # 2. Player
            player_x_position,
            player_y_position,
            player_hp_current,
            player_direction_x,
            player_direction_y,
            player_is_moving_flag,
            player_attack_cooldown_progress,
            # 3. Boss
            boss_x_position,
            boss_y_position,
            boss_hp_current,
            boss_direction_x,
            boss_direction_y,
            boss_is_moving_flag,
            boss_ability_1_cooldown_progress,
            boss_ability_2_cooldown_progress,

            # 4. attacks 

        ]

        # 4 Player s Bullets ( for each existing attack )  
            #  player_proj_N_type_id
            #  player_proj_N_x_position
            #  player_proj_N_y_position
            #  player_proj_N_direction_x
            #  player_proj_N_direction_y
            #  player_proj_N_speed
            #  player_proj_N_size
            #  player_proj_N_damage_potential
        # 5. Boss Projectile Features
            #  boss_proj_N_type_id
            #  boss_proj_N_x_position
            #  boss_proj_N_y_position
            #  boss_proj_N_direction_x
            #  boss_proj_N_direction_y
            #  boss_proj_N_speed
            #  boss_proj_N_size
            #  boss_proj_N_hp_current
            #  boss_proj_N_damage_potential
        # 4. Player's Bullets (flatten)
        for proj in player_proj_features:
            feature_vector.extend(proj)
        # 5. Boss Projectiles (flatten)
        for proj in boss_proj_features:
            feature_vector.extend(proj)

        # Convert to numpy array for GRU input
        return np.array(feature_vector, dtype=np.float32)

    def calculate_interval_stats(self, interval_ticks):
        """
        Calculates stats over the last `interval_ticks` ticks.
        If interval_ticks is None or exceeds available ticks, uses all available ticks.
        Returns:
            player_hp_change (normalized),
            boss_hp_change (normalized),
            player_dodge_rate,
            boss_attack_frequency,
            number_of_attacks_prevented_by_player,
            time_since_boss_last_took_damage_from_player
        """
        session = self.logged_data_history[self.current_session_id]
        if not session:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Use only the last `interval_ticks` records
        records = session[-interval_ticks:] if interval_ticks and interval_ticks < len(session) else session

        # Initialize
        player_hp_start = records[0]["data"]["player"]["hp"]
        boss_hp_start = records[0]["data"]["boss"]["hp"]
        player_hp_end = records[-1]["data"]["player"]["hp"]
        boss_hp_end = records[-1]["data"]["boss"]["hp"]

        # Normalized HP change (absolute change, not rate)
        max_player_hp = self.it_settings["player"].get("max_hp", 1)
        max_boss_hp = self.it_settings["boss"].get("max_hp", 1)
        player_hp_change = (player_hp_end - player_hp_start) / max_player_hp
        boss_hp_change = (boss_hp_end - boss_hp_start) / max_boss_hp

        # Dodge rate and attack frequency
        player_dodge_count = 0
        boss_attack_count = 0
        attacks_prevented = 0
        last_boss_damage_tick = None
        prev_boss_hp = boss_hp_start

        for i, rec in enumerate(records):
            data = rec["data"]
            boss_attack_count += sum(1 for a in data.get("attacks", []) if a.get("boss", 0) == 1)
            if data.get("player", {}).get("dodged", False):
                player_dodge_count += 1
            attacks_prevented += data.get("attacks_prevented", 0)
            boss_hp = data.get("boss", {}).get("hp", prev_boss_hp)
            if boss_hp < prev_boss_hp:
                last_boss_damage_tick = i
            prev_boss_hp = boss_hp

        ticks = len(records)
        player_dodge_rate = player_dodge_count / ticks
        boss_attack_frequency = boss_attack_count / ticks
        number_of_attacks_prevented_by_player = attacks_prevented / ticks
        if last_boss_damage_tick is not None:
            time_since_boss_last_took_damage_from_player = ticks - last_boss_damage_tick - 1
        else:
            time_since_boss_last_took_damage_from_player = ticks
        
        return (
            player_hp_change,
            boss_hp_change,
            player_dodge_rate,
            boss_attack_frequency,
            number_of_attacks_prevented_by_player,
            time_since_boss_last_took_damage_from_player
        )

    def check_player_dodge(self, current_game_state, previous_game_state, dodge_distance=20):
        """
        Checks if the player dodged a bullet in the current tick.
        Sets current_game_state["player"]["dodged"] = True if a dodge is detected.
        """
        if not previous_game_state:
            current_game_state["player"]["dodged"] = False
            return

        player = current_game_state.get("player", {})
        prev_player = previous_game_state.get("player", {})
        player_pos = np.array([player.get("x", 0), player.get("y", 0)])
        player_dir = np.array(player.get("direction", [0, 0]))
        prev_player_dir = np.array(prev_player.get("direction", [0, 0]))

        # Check for direction change 
        changed_direction = not np.array_equal(player_dir, prev_player_dir) and np.linalg.norm(player_dir) > 0

        # Check all boss projectiles
        dodged = False
        for attack in current_game_state.get("attacks", []):
            if attack.get("boss", 0) == 1:
                bullet_pos = np.array([attack.get("x", 0), attack.get("y", 0)])
                distance = np.linalg.norm(player_pos - bullet_pos)
                # Was the bullet close enough?
                if distance <= dodge_distance:
                    # Did the player change direction and not get hit?
                    if changed_direction:
                        # Optionally: check if player's HP did not decrease
                        if player.get("hp", 0) == prev_player.get("hp", 0):
                            dodged = True
                            break

        current_game_state["player"]["dodged"] = dodged

    def debug_print_state(self):
        """
        Prints the current game state with all variable names, values, normalization calculations, and normalized values.
        Also prints the final feature vector with variable names.
        """
        session = self.logged_data_history[self.current_session_id]
        if not session:
            print("No state to debug.")
            return

        current_tick_record = session[-1]
        current_game_state = current_tick_record["data"]
        iv = self.it_settings

        print("\n--- DEBUG GAME STATE (JSON) ---")
        print(json.dumps(current_game_state, indent=2))

        print("\n--- Normalized Values & Calculations ---")

        # Player
        player = current_game_state.get("player", {})
        max_player_hp = iv["player"].get("max_hp", 1)
        player_x_norm = player.get("x", 0) / iv["map_width"]
        player_y_norm = player.get("y", 0) / iv["map_height"]
        player_hp_norm = player.get("hp", 0) / max_player_hp
        player_dir_x = player.get("direction", [0, 0])[0]
        player_dir_y = player.get("direction", [0, 0])[1]
        player_is_moving = 1 if player.get("move", [0, 0]) != [0, 0] else 0
        player_attack_cooldown = player.get("attack_cooldown", 0)
        max_attack_cooldown = iv["player_attacks"].get("max_attack_cooldown", 1)
        player_attack_cooldown_norm = player_attack_cooldown / max_attack_cooldown

        print(f"player_x_position: {player.get('x', 0)} / {iv['map_width']} = {player_x_norm}")
        print(f"player_y_position: {player.get('y', 0)} / {iv['map_height']} = {player_y_norm}")
        print(f"player_hp_current: {player.get('hp', 0)} / {max_player_hp} = {player_hp_norm}")
        print(f"player_direction_x: {player_dir_x}")
        print(f"player_direction_y: {player_dir_y}")
        print(f"player_is_moving_flag: {player_is_moving}")
        print(f"player_attack_cooldown_progress: {player_attack_cooldown} / {max_attack_cooldown} = {player_attack_cooldown_norm}")

        # Boss
        boss = current_game_state.get("boss", {})
        max_boss_hp = iv["boss"].get("max_hp", 1)
        boss_x_norm = boss.get("x", 0) / iv["map_width"]
        boss_y_norm = boss.get("y", 0) / iv["map_height"]
        boss_hp_norm = boss.get("hp", 0) / max_boss_hp
        boss_dir_x = boss.get("direction", [0, 0])[0]
        boss_dir_y = boss.get("direction", [0, 0])[1]
        boss_is_moving = 1 if boss.get("move", [0, 0]) != [0, 0] else 0

        print(f"boss_x_position: {boss.get('x', 0)} / {iv['map_width']} = {boss_x_norm}")
        print(f"boss_y_position: {boss.get('y', 0)} / {iv['map_height']} = {boss_y_norm}")
        print(f"boss_hp_current: {boss.get('hp', 0)} / {max_boss_hp} = {boss_hp_norm}")
        print(f"boss_direction_x: {boss_dir_x}")
        print(f"boss_direction_y: {boss_dir_y}")
        print(f"boss_is_moving_flag: {boss_is_moving}")

        # Projectiles
        print("\nPlayer Projectiles:")
        for attack in current_game_state.get("attacks", []):
            if attack.get("boss", 0) == 0:
                aid = attack.get("attack_id", 0)
                max_speed = iv["boss_attacks"].get("max_attack_speed", 1)
                max_size = iv["boss_attacks"].get("max_attack_size", 1)
                max_damage = iv["boss_attacks"].get("max_attack_damage", 1)
                print(f"  id: {aid}, x: {attack.get('x', 0)}, y: {attack.get('y', 0)}, "
                      f"speed: {attack.get('speed', 0)} / {max_speed} = {attack.get('speed', 0)/max_speed if max_speed else 0}, "
                      f"size: {attack.get('size', 0)} / {max_size} = {attack.get('size', 0)/max_size if max_size else 0}, "
                      f"damage: {attack.get('damage', 0)} / {max_damage} = {attack.get('damage', 0)/max_damage if max_damage else 0}")

        print("\nBoss Projectiles:")
        for attack in current_game_state.get("attacks", []):
            if attack.get("boss", 0) == 1:
                aid = attack.get("attack_id", 0)
                max_speed = iv["boss_attacks"].get("max_attack_speed", 1)
                max_size = iv["boss_attacks"].get("max_attack_size", 1)
                max_damage = iv["boss_attacks"].get("max_attack_damage", 1)
                max_hp = iv["boss_attacks"].get("max_attack_hp", 1)
                print(f"  id: {aid}, x: {attack.get('x', 0)}, y: {attack.get('y', 0)}, "
                      f"speed: {attack.get('speed', 0)} / {max_speed} = {attack.get('speed', 0)/max_speed if max_speed else 0}, "
                      f"size: {attack.get('size', 0)} / {max_size} = {attack.get('size', 0)/max_size if max_size else 0}, "
                      f"hp: {attack.get('hp', 0)} / {max_hp} = {attack.get('hp', 0)/max_hp if max_hp else 0}, "
                      f"damage: {attack.get('damage', 0)} / {max_damage} = {attack.get('damage', 0)/max_damage if max_damage else 0}")

        # --- Feature Vector with Names ---
        print("\n--- Feature Vector ---")
        # You may want to keep a list of variable names in the same order as your feature vector
        feature_names = [
            "player_hp_change",
            "boss_hp_change",
            "player_dodge_rate",
            "boss_attack_frequency",
            "number_of_attacks_prevented_by_player",
            "time_since_boss_last_took_damage_from_player",
            "active_player_projectile_count",
            "active_boss_projectile_count",
            "player_x_position",
            "player_y_position",
            "player_hp_current",
            "player_direction_x",
            "player_direction_y",
            "player_is_moving_flag",
            "player_attack_cooldown_progress",
            "boss_x_position",
            "boss_y_position",
            "boss_hp_current",
            "boss_direction_x",
            "boss_direction_y",
            "boss_is_moving_flag",
            "boss_ability_1_cooldown_progress",
            "boss_ability_2_cooldown_progress",
            # ...plus one name per each element in your flattened projectile features...
        ]

        # Get the actual feature vector
        feature_vector = self.extract_feature_vector()
        # Print with names (for the first N features)
        for i, name in enumerate(feature_names):
            if i < len(feature_vector):
                print(f"{name}: {feature_vector[i]}")
        # Print the rest (projectile features) as a flat list
        if len(feature_vector) > len(feature_names):
            print("...projectile features (flattened):", feature_vector[len(feature_names):])

        print("\n--- End Debug ---\n")

# New assumptions:
# Diffrent boss attacks will be treated 

# Feature vector
# 1. Stats 
#  [in intervals of 5 seconds  ( can be extended to  more intervals 5sec, 20 sec and whole game )]
#  player_hp_change_rate,  
#  boss_hp_change_rate
#  player_dodge_rate, 
#  boss_attack_frequency
#  active_player_projectile_count  
#  active_boss_projectile_count 

#  [in current state]
#  number_of_attacks_prevented_by_player ( destroyed bullets)
#  active_player_projectile_count
#  active_boss_projectile_count
#  time_since_boss_last_took_damage_from_player

# 2 Player
#  player_x_position
#  player_y_position
#  player_hp_current
#  player_direction_x
#  player_direction_y
#  player_is_moving_flag
#  player_attack_cooldown_progress
# 3 Boss
#  boss_x_position
#  boss_y_position
#  boss_damage_taken
#  boss_direction_x
#  boss_direction_y
#  boss_is_moving_flag
#  boss_ability_1_cooldown_progress
#  boss_ability_2_cooldown_progress
# 4 Player s Bullets ( for each existing attack )  
#  player_proj_N_type_id
#  player_proj_N_x_position
#  player_proj_N_y_position
#  player_proj_N_direction_x
#  player_proj_N_direction_y
#  player_proj_N_speed
#  player_proj_N_size
#  player_proj_N_damage_potential
# 5. Boss Projectile Features
#  boss_proj_N_type_id
#  boss_proj_N_x_position
#  boss_proj_N_y_position
#  boss_proj_N_direction_x
#  boss_proj_N_direction_y
#  boss_proj_N_speed
#  boss_proj_N_size
#  boss_proj_N_hp_current
#  boss_proj_N_damage_potential

# For optimal training the values outside of [-1,1] interval will be normalized to that interval ( due to usage of tanh by GRU )


# Output variables:
# desired_player_hp_change_rate
# desired_played_dodge_rate
# desired_player_damage_output_rate_to_boss
# desired_time_since_boss_last_took_damage_from_player


# GRU + PPO Output variables:
#  new_boss_projectile_multiplier
#  new_boss_attack_frequency_multiplier
#  new_boss_ability_1_cooldown_duration
#  new_boss_ability_2_cooldown_duration
#  new_boss_damage_output_multiplier
#  new_boss_proj_N_multiplier
#  new_boss_proj_N_multiplier
#  new_boss_proj_N_hp_multiplier
#  new_boss_movement_multiplier
#  new_max_boss_projectiles




# for reward function we want to keep player hp decrease simillarl to boss hp decrease
# and player dodge rate similar but smaller to boss attack frequency