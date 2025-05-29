from Simulation.Characters.Attacks import regular_attack
from Simulation.Characters.Attacks import fireball
from Simulation.Characters.Attacks import ice_shard
from Simulation.Characters.Attacks import non_attack

class Character:
    def __init__(self, x, y, hp, size,speed, boss, attacks):
        self.x = x  # X-coordinate position
        self.y = y  # Y-coordinate position
        self.hp = hp  # Health points
        self.speed = speed  # Movement speed
        self.boss = boss
        self.attacks = attacks
        self.attack = None
        self.move = [0, 0]
        self.direction = [1,0]
        self.size= size
        self.attacks_ids = [attack.attack_id for attack in attacks]
    def move_up(self):
        """Move the character by dx and dy, scaled by movement speed."""
        self.direction = [0,-1]
        self.move = [0,-1]
    def move_down(self):
        """Move the character by dx and dy, scaled by movement speed."""
        self.direction = [0,1]
        self.move = [0,1]
    def move_left(self):
        """Move the character by dx and dy, scaled by movement speed."""
        self.direction = [-1,0]
        self.move = [-1,0]
    def move_right(self):
        """Move the character by dx and dy, scaled by movement speed."""
        self.direction = [1,0]
        self.move = [1,0]

    def take_damage(self, damage):
        """Reduce the character's HP by the damage amount."""
        self.hp -= damage

    def create_attack(self, attack):
        """Attack another character."""
        if attack in self.attacks:
            if attack.current_cooldown <= 0:
                attack.x = self.x  # Set the attack's initial position to the character's position
                attack.y = self.y
                attack.direction = self.direction  # Set the attack's direction to the character's current direction
                self.attack = attack.attack_id
                attack.current_cooldown = attack.cooldown  # Reset cooldown
                print(f"Attack {attack.attack_id} created at ({attack.x}, {attack.y}) with direction {attack.direction}")
            else:
                print(f"{type(attack).__name__} is on cooldown for {attack.current_cooldown} turns.")
        else:
            print("Invalid attack.")

# CreateΩthe player character
player = Character(x=0, y=64, hp=100,size=10, speed=5, boss=0, attacks=[regular_attack])

# Create the boss character
boss = Character(x=127, y=64, hp=500, size = 25, speed=2, boss=1, attacks=[fireball, ice_shard])

