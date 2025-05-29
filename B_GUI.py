import tkinter as tk
from Simulation.Game import game
from Simulation.Enviroment import environment
from Simulation.Characters.Character import player, boss
from Simulation.Characters.Attacks import fireball, ice_shard, regular_attack

class BasicGUI:
    def __init__(self, game_instance):
        self.game = game_instance
        self.root = tk.Tk()
        self.root.title("Basic GUI")
        self.canvas = tk.Canvas(self.root, width=self.game.env.width, height=self.game.env.height, bg="white")
        self.canvas.pack()
        self.score = 0


        # Key bindings for player movement and attacks
        self.root.bind("w", lambda event: self.move_player("up"))
        self.root.bind("s", lambda event: self.move_player("down"))
        self.root.bind("a", lambda event: self.move_player("left"))
        self.root.bind("d", lambda event: self.move_player("right"))
        self.root.bind("q", lambda event: self.player_attack(regular_attack))

        # Key bindings for boss movement and attacks
        self.root.bind("i", lambda event: self.move_boss("up"))
        self.root.bind("k", lambda event: self.move_boss("down"))
        self.root.bind("j", lambda event: self.move_boss("left"))
        self.root.bind("l", lambda event: self.move_boss("right"))
        self.root.bind("o", lambda event: self.boss_attack(fireball))
        self.root.bind("p", lambda event: self.boss_attack(ice_shard))
        self.root.bind("u", lambda event: self.boss_attack(regular_attack))

        self.update_canvas()

    def move_player(self, direction):
        player_entity = self.game.get_entity(isBoss=0)
        if direction == "up":
            player_entity.move_up()
        elif direction == "down":
            player_entity.move_down()
        elif direction == "left":
            player_entity.move_left()
        elif direction == "right":
            player_entity.move_right()
        self.game.process_turn()  # Process the turn after the move
        self.update_canvas()

    def move_boss(self, direction):
        boss_entity = self.game.get_entity(isBoss=1)
        if direction == "up":
            boss_entity.move_up()
        elif direction == "down":
            boss_entity.move_down()
        elif direction == "left":
            boss_entity.move_left()
        elif direction == "right":
            boss_entity.move_right()
        self.game.process_turn()  # Process the turn after the move
        self.update_canvas()

    def player_attack(self, attack):
        player_entity = self.game.get_entity(isBoss=0)
        if attack in player_entity.attacks:
            player_entity.create_attack(attack)
            self.game.process_turn()  # Process the turn after the attack
            self.update_canvas()

    def boss_attack(self, attack):
        boss_entity = self.game.get_entity(isBoss=1)
        if attack in boss_entity.attacks:
            boss_entity.create_attack(attack)
            self.game.process_turn()  # Process the turn after the attack
            self.update_canvas()

    def reset_game(self):
        """Reset the game when the reset button is clicked."""
        self.game.reset()
        self.update_canvas()

    def update_canvas(self):
        self.canvas.delete("all")

        # Draw entities
        for entity in self.game.env.entities:
            color = "blue" if entity.boss == 0 else "red"
            # Draw the square for the entity
            self.canvas.create_rectangle(
                entity.x - entity.size, entity.y - entity.size,
                entity.x + entity.size, entity.y + entity.size,
                fill=color
            )

            # Draw the triangle to indicate direction
            triangle_size = entity.size // 2
            triangle_color = "white" if entity.boss == 0 else "black"  # White for player, black for boss
            if entity.direction == [1, 0]:  # Facing right
                triangle_coords = [
                    entity.x + entity.size, entity.y,
                    entity.x + entity.size - triangle_size, entity.y - triangle_size,
                    entity.x + entity.size - triangle_size, entity.y + triangle_size
                ]
            elif entity.direction == [-1, 0]:  # Facing left
                triangle_coords = [
                    entity.x - entity.size, entity.y,
                    entity.x - entity.size + triangle_size, entity.y - triangle_size,
                    entity.x - entity.size + triangle_size, entity.y + triangle_size
                ]
            elif entity.direction == [0, -1]:  # Facing up
                triangle_coords = [
                    entity.x, entity.y - entity.size,
                    entity.x - triangle_size, entity.y - entity.size + triangle_size,
                    entity.x + triangle_size, entity.y - entity.size + triangle_size
                ]
            elif entity.direction == [0, 1]:  # Facing down
                triangle_coords = [
                    entity.x, entity.y + entity.size,
                    entity.x - triangle_size, entity.y + entity.size - triangle_size,
                    entity.x + triangle_size, entity.y + entity.size - triangle_size
                ]
            else:
                triangle_coords = []  # Default to no triangle if direction is invalid

            if triangle_coords:
                self.canvas.create_polygon(triangle_coords, fill=triangle_color)

        # Draw attacks
        for attack in self.game.env.attacks:
            if attack.attack_id == regular_attack.attack_id:
                color = "green"
            elif attack.attack_id == fireball.attack_id:
                color = "orange"
            elif attack.attack_id == ice_shard.attack_id:
                color = "cyan"
            else:
                color = "black"

            self.canvas.create_oval(
                attack.x - attack.size, attack.y - attack.size,
                attack.x + attack.size, attack.y + attack.size,
                fill=color
            )

        # Display player and boss HP
        player_entity = self.game.get_entity(isBoss=0)
        boss_entity = self.game.get_entity(isBoss=1)
        self.canvas.create_text(
            10, 10, anchor="nw", text=f"Player HP: {player_entity.hp}", fill="black", font=("Arial", 12)
        )
        self.canvas.create_text(
            self.game.env.width - 10, 10, anchor="ne", text=f"Boss HP: {boss_entity.hp}", fill="black", font=("Arial", 12)
        )

        # Display attack cooldowns
        y_offset = 30
        for attack in player_entity.attacks:
            self.canvas.create_text(
                10, y_offset, anchor="nw",
                text=f"{type(attack).__name__} Cooldown: {attack.current_cooldown}",
                fill="black", font=("Arial", 12)
            )
            y_offset += 20

        for attack in boss_entity.attacks:
            self.canvas.create_text(
                self.game.env.width - 10, y_offset, anchor="ne",
                text=f"{type(attack).__name__} Cooldown: {attack.current_cooldown}",
                fill="black", font=("Arial", 12)
            )
            y_offset += 20

        # Display score in the middle of the screen
        self.score +=self.game.env.score
        self.canvas.create_text(
            self.game.env.width // 2, 10, anchor="n",
            text=f"Score: {self.score}", fill="black", font=("Arial", 16)
        )

        # Display turn count in the bottom-right corner
        self.canvas.create_text(
            self.game.env.width - 10, self.game.env.height - 10, anchor="se",
            text=f"Turn: {self.game.env.turn}", fill="black", font=("Arial", 12)
        )

        # Add reset button if the game is over
        if self.game.done:
            reset_button = tk.Button(self.root, text="Reset Game", command=self.reset_game)
            reset_button.pack()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    env = environment()
    env.resett()  # Initialize the environment
    game_instance = game(env)  # Create a game instance
    gui = BasicGUI(game_instance)
    gui.run()
