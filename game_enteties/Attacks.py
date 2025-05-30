# Example Attack class definition
class Attack:
    def __init__(self, damage, speed, cooldown, attack_id, size, boss=0, hp=0):
        self.damage = damage
        self.speed = speed
        self.cooldown = cooldown
        self.current_cooldown = 0
        self.boss = boss
        self.hp = hp
        self.direction = []
        self.x = 0
        self.y = 0
        self.size = size
        self.attack_id = attack_id
        self.creator = None  # Reference to the entity that created the attack

non_attack = Attack( damage=0, speed =0, cooldown=0,attack_id=0,size= 0,boss=0, hp=0)
regular_attack = Attack( damage=10, speed =2, cooldown=2,attack_id=1,size= 5,boss=0, hp=5)
fireball = Attack( damage=40, speed= 5,cooldown=10,attack_id=2, size = 10,boss=1, hp=20)
ice_shard = Attack( damage=20,speed=8, cooldown=5,attack_id=3,size = 5,boss=1, hp=10)

