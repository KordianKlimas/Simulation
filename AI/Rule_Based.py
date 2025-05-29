import random

class RuleBased:
    def __init__(self, game, is_boss):
        self.game = game
        self.character = game.get_entity(is_boss)  # Get the character (0 for player, 1 for boss)
        self.is_boss = is_boss
        self.rules = self.create_rules()

    def apply_rules(self):
        """Apply the rules for the character."""
        rules = self.rules[1] if self.is_boss else self.rules[0]  # Boss or player rules
        for rule in rules:
            if rule['condition']():
                for action in rule['actions']:
                    if callable(action):
                        action()
                        if not rule['condition']():  # Recheck condition
                            break
                break

    def move_randomly(self):
        """Move in a random direction."""
        directions = ["move_up", "move_down", "move_left", "move_right"]
        direction = random.choice(directions)
        getattr(self.character, direction)()

    def move_towards(self, target):
        """Move towards a target character."""
        dx = target.x - self.character.x
        dy = target.y - self.character.y
        if abs(dx) > abs(dy):
            if dx > 0:
                self.character.move_right()
            else:
                self.character.move_left()
        else:
            if dy > 0:
                self.character.move_down()
            else:
                self.character.move_up()

    def move_away_from(self, target):
        """Move away from a target character."""
        dx = self.character.x - target.x
        dy = self.character.y - target.y
        if abs(dx) > abs(dy):
            if dx > 0:
                self.character.move_left()
            else:
                self.character.move_right()
        else:
            if dy > 0:
                self.character.move_up()
            else:
                self.character.move_down()

    def attack_target(self, target):
        """Attack the target if possible."""
        valid_attacks = [attack for attack in self.character.attacks if attack.current_cooldown <= 0]
        if valid_attacks:
            attack = random.choice(valid_attacks)
            self.character.create_attack(attack)

    def create_rules(self):
        """Define rules for the player and boss."""
        def is_within_range(entity, target, range_x, range_y):
            """Check if the target is within range, considering the sizes of both entities."""
            effective_range_x = range_x + entity.size + target.size
            effective_range_y = range_y + entity.size + target.size
            return abs(entity.x - target.x) <= effective_range_x and abs(entity.y - target.y) <= effective_range_y

        player = self.game.get_entity(False)  # Get player object
        boss = self.game.get_entity(True)  # Get boss object

        player_rules = [
            {
                "condition": lambda: is_within_range(self.character, boss, 100, 100),
                "actions": [lambda: self.attack_target(boss)]
            },
            {
                "condition": lambda: is_within_range(self.character, boss, 20, 20),
                "actions": [lambda: self.move_away_from(boss)]
            },
            {
                "condition": lambda: random.random() < 0.1,
                "actions": [self.move_randomly]
            }
        ]

        boss_rules = [
            {
                "condition": lambda: is_within_range(self.character, player, 20, 20),
                "actions": [lambda: self.attack_target(player)]
            },
            {
                "condition": lambda: not is_within_range(self.character, player, 10, 10),
                "actions": [lambda: self.move_towards(player)]
            },
            {
                "condition": lambda: random.random() < 0.1,
                "actions": [self.move_randomly]
            }
        ]

        return player_rules, boss_rules