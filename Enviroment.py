from torchgen.executorch.api.et_cpp import return_type

class environment:
    def __init__(self, width=1024, height=1024):
        self.width = width
        self.height = height
        self.done = False
        self.entities = []
        self.attacks = []
        self.health_drops= []
        self.score = 0
        self.turn = 0
    def resett(self):
        from Characters.Character import player, boss
        self.entities.clear()
        self.attacks.clear()
        self.score = 0
        self.health_drops.clear()
        self.done = False
        self.add_entity(player)
        self.add_entity(boss)
        self.turn = 0
        print("Environment has been reset.")
        return self.encode_state(self.entities, self.attacks, self.score, self.done, self.turn)


    def add_entity(self, entity):
        if 0 <= entity.x < self.width and 0 <= entity.y < self.height:
            self.entities.append(entity)
        else:
            raise ValueError("Entity position is out of bounds.")

    def get_env_entity(self, entity):
        """Get an entity by its name."""
        for e in self.entities:
            if e.boss  == entity:
                return e
        return None

    def add_attack(self, attack_id, entity):
        """Add an attack entity to the environment based on the character's current attack."""
        from Characters.Attacks import Attack
        # Find the attack in the entity's attack list
        for attack in entity.attacks:
            if attack.attack_id == attack_id:
                # Create a new attack object with the same properties
                new_attack = Attack(
                    damage=attack.damage,
                    speed=attack.speed,
                    cooldown=attack.cooldown,
                    attack_id=attack.attack_id,
                    size=attack.size,
                    boss=entity.boss,
                    hp=attack.hp
                )
                # Set the attack's initial position, direction, and creator
                new_attack.x = entity.x
                new_attack.y = entity.y
                new_attack.direction = entity.direction
                new_attack.creator = entity  # Reference to the entity that created the attack
                # Add the attack to the environment
                self.attacks.append(new_attack)
                print(f"Attack created: {new_attack.attack_id} at ({new_attack.x}, {new_attack.y}) with direction {new_attack.direction}")

    def decrease_cooldowns(self):
        """Decrease the current cooldown for all attacks in the environment."""
        for entity in self.entities:
            for attack in entity.attacks:
                if attack.current_cooldown > 0:
                     attack.current_cooldown -= 1

    def add_health_drop(self, health_drops):
        """Add a health drop entity to the environment."""
        if 0 <= health_drops.x < self.width and 0 <= health_drops.y < self.height:
            self.health_drops.append(health_drops)
        else:
            raise ValueError("Health drop position is out of bounds.")

    def check_collision(self, entity1, entity2):
        """Check if two entities collide based on their positions and sizes."""
        return (
                entity1.x + entity1.size > entity2.x - entity2.size
                and entity1.x - entity1.size < entity2.x + entity2.size
                and entity1.y + entity1.size > entity2.y - entity2.size
                and entity1.y - entity1.size < entity2.y + entity2.size
        )

    def check_collision_dimensions(self, entity1, entity1_x, entity1_y, entity2):
        """Check if two entities collide based on their positions and sizes."""
        return (
                entity1_x + entity1.size > entity2.x - entity2.size
                and entity1_x - entity1.size < entity2.x + entity2.size
                and entity1_y + entity1.size > entity2.y - entity2.size
                and entity1_y - entity1.size < entity2.y + entity2.size
        )

    def update_attacks(self):
        """Update the movement of all attacks and handle collisions."""
        for attack in self.attacks[:]:  # Iterate over a copy of the list to allow modifications
            # Move the attack based on its direction and speed
            attack.x += attack.direction[0] * attack.speed
            attack.y += attack.direction[1] * attack.speed

            # Print the updated coordinates of the attack
            print(f"Attack {attack.attack_id} updated: ({attack.x}, {attack.y})")

            # Check for collisions with entities
            for entity in self.entities:
                # Ensure the attack does not hit the entity that created it
                if attack.creator == entity:
                    continue

                if self.check_collision(attack, entity):
                    entity.take_damage(attack.damage)  # Apply damage to the entity
                    if entity.boss == 1:  # If the entity is the boss, add score proportional to the damage
                        self.score += attack.damage
                        print(f"Score increased by {attack.damage}. Total score: {self.score}")
                    self.attacks.remove(attack)  # Remove attack if its HP is 0
                    print(f"Collision detected: {entity} hit by attack at ({attack.x}, {attack.y})")
                    if entity.hp <= 0:
                        print(f"{entity} has been defeated!")
                        break
                    break

            # Check for collisions with other attacks
            for other_attack in self.attacks[:]:
                # Ensure projectiles from the same creator type do not collide
                if attack.creator.boss == other_attack.creator.boss:
                    continue

                if self.check_collision(attack, other_attack):
                    # Handle projectile collision
                    print(f"Projectile collision: Attack {attack.attack_id} hit Attack {other_attack.attack_id}")
                    if attack.damage >= other_attack.hp:
                        self.attacks.remove(other_attack)
                    else:
                        other_attack.hp -= attack.damage
                    if other_attack.damage >= attack.hp:
                        self.attacks.remove(attack)
                    else:
                        attack.hp -= other_attack.damage
                        break

            # Remove the attack if it goes out of bounds
            if attack.x < 0 or attack.x >= self.width or attack.y < 0 or attack.y >= self.height:
                print(f"Attack {attack.attack_id} went out of bounds at ({attack.x}, {attack.y})")
                self.attacks.remove(attack)

    def do_action_entities(self):
        for entity in self.entities:
            # Handle movement
            if isinstance(entity.move, list):  # Ensure the entity has a valid move
                new_x = entity.x + entity.move[0] * entity.speed
                new_y = entity.y + entity.move[1] * entity.speed

                # Check for collisions with other entities
                collision = any(
                    self.check_collision_dimensions(entity, new_x, new_y, other_entity)
                    for other_entity in self.entities
                    if other_entity != entity
                )

                # Check if the movement would go out of bounds
                out_of_bounds = (new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height)

                if collision:
                    print(f"Collision detected: {entity.boss} cannot move to ({new_x}, {new_y})")
                if out_of_bounds:
                    print(f"Entity {entity.boss} is out of bounds at ({new_x}, {new_y})")

                # Only update position if there is no collision and the movement is within bounds
                if not collision and not out_of_bounds:
                    entity.x = new_x
                    entity.y = new_y
                else:
                    # Reset the move and direction if movement is invalid
                    entity.move = [0, 0]

            # Handle attack creation
            if isinstance(entity.attack, int) and entity.attack != 0:
                # Create the attack regardless of whether others of the same type exist
                for attack in entity.attacks:
                    if attack.attack_id == entity.attack:
                        self.add_attack(attack.attack_id, entity)
                        print(f"Attack created: {attack.attack_id}")
                        break
                # Reset the entity's attack to prevent repeated creation in the same turn
                entity.attack = None

    def encode_state(self,entities, attacks, score, done, turn):
        done = 1 if done else 0
        for entity in entities:
            if entity.boss :
                entity.boss = 1
            else: entity.boss = 0
        """Encode the state into a structured format for AI processing."""
        state = {
            "entities": [
                {"x": entity.x, "y": entity.y, "move": entity.move, "direction": entity.direction, "hp": entity.hp, "attack": entity.attack,
                 "boss": entity.boss, "size": entity.size}
                for entity in entities
            ],
            "attacks": [
                {"x": attack.x, "y": attack.y, "damage": attack.damage, "speed": attack.speed, "hp": attack.hp,
                 "size": attack.size, "direction": attack.direction, "current_cooldown": attack.current_cooldown,
                 "cooldown": attack.cooldown, "attack_id": attack.attack_id}
                for attack in (attacks if attacks is not None else [])
            ],
            "score": score,
            "done": done,
            "turn": turn
        }
        return state
    def update(self):
        self.score = 0
        self.decrease_cooldowns()
        self.turn += 1

        self.update_attacks()

        self.do_action_entities()

        # Reset movement for all entities after processing the round
        for entity in self.entities:
            if self.turn% 5 == 0:
               entity.move = [0, 0]

        # Check if the game is over (all entities of one type are dead)
        if all(entity.boss == 0 for entity in self.entities) or all(entity.boss == 1 for entity in self.entities):
            self.done = True

        return self.encode_state(self.entities, self.attacks, self.score, self.done, self.turn)



