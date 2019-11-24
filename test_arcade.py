import arcade
import random
import sys

# Constants
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
SCREEN_TITLE = "Freeze Tag"
MOVEMENT_SPEED = 5
FREE_AGENTS = 4
TOGGLE_FREE_AGENT = False
TIME_LIMIT = 20
""" Reward: -1 if a tagger tags an agent, +1 if an agent is unfrozen, bonus at end for # of agents left alive """
# Iterate through free agents for movement, abstract left, right, up, down

class FreezeTagGame(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        self.free_agents = arcade.SpriteList()
        self.tagger = None
        self.wall_list = arcade.SpriteList()
        self.frozen_agents = arcade.SpriteList()
        self.toggled_agent = None
        f = open("tagger_input.txt", "r")
        self.commands = f.read().split(" \n")[:-1] # Reading commands from file
        f.close()

        self.index = 0
        self.reward = 0
        self.model = None

    def setup(self):
        # Creating tagger
        self.tagger = arcade.Sprite("tagger.png")
        self.tagger.center_x = SCREEN_WIDTH / 2
        self.tagger.center_y = SCREEN_HEIGHT / 2

        # Creating free agents
        for i in range(FREE_AGENTS):
            free_agent = arcade.Sprite("free_agent.png")
            free_agent.center_x = SCREEN_HEIGHT - 100 * (i + 1)
            free_agent.center_y = 800

            self.free_agents.append(free_agent)

        # Initialize controllable free agent for testing
        if TOGGLE_FREE_AGENT:
            self.toggled_agent = self.free_agents.sprite_list[0]

        # Creating the rows of wall
        for x in range(50, SCREEN_WIDTH - 50, 50):
            wall = arcade.Sprite("wall.png")
            wall.center_x = x
            wall.center_y = 200
            self.wall_list.append(wall)

        # Creating the rows of wall
        for x in range(50, SCREEN_WIDTH - 50, 50):
            wall = arcade.Sprite("wall.png")
            wall.center_x = x
            wall.center_y = SCREEN_HEIGHT - 50
            self.wall_list.append(wall)

        # Creating the columns of wall
        for y in range(200, SCREEN_HEIGHT - 50, 50):
            wall = arcade.Sprite("wall.png")
            wall.center_x = 50
            wall.center_y = y
            self.wall_list.append(wall)

        # Creating the columns of wall
        for y in range(200, SCREEN_HEIGHT, 50):
            wall = arcade.Sprite("wall.png")
            wall.center_x = SCREEN_WIDTH - 50
            wall.center_y = y
            self.wall_list.append(wall)

        self.tagger_physics = arcade.PhysicsEngineSimple(self.tagger, self.wall_list)

        arcade.set_background_color(arcade.color.WHITE)

    def on_draw(self):
        """ Render the screen. """

        arcade.start_render()
        self.wall_list.draw()
        self.tagger.draw()
        self.free_agents.draw()

    # General movement methods
    def move_left(self, agent):
        agent.change_x = -MOVEMENT_SPEED

    def move_right(self, agent):
        agent.change_x = MOVEMENT_SPEED

    def move_up(self, agent):
        agent.change_y = MOVEMENT_SPEED

    def move_down(self, agent):
        agent.change_y = -MOVEMENT_SPEED

    def on_key_press(self, key, modifiers):
#        if key == arcade.key.K:
#            self.tagger.change_y = MOVEMENT_SPEED
#        elif key == arcade.key.J:
#            self.tagger.change_y = -MOVEMENT_SPEED
#        elif key == arcade.key.H:
#            self.tagger.change_x = -MOVEMENT_SPEED
#        elif key == arcade.key.L:
#            self.tagger.change_x = MOVEMENT_SPEED

        if TOGGLE_FREE_AGENT:
            if key == arcade.key.W:
                self.toggled_agent.change_y = MOVEMENT_SPEED
            elif key == arcade.key.S:
                self.toggled_agent.change_y = -MOVEMENT_SPEED
            elif key == arcade.key.A:
                self.toggled_agent.change_x = -MOVEMENT_SPEED
            elif key == arcade.key.D:
                self.toggled_agent.change_x = MOVEMENT_SPEED

    def on_key_release(self, key, modifiers):
#        if key == arcade.key.K or key == arcade.key.J:
#            self.tagger.change_y = 0
#        elif key == arcade.key.H or key == arcade.key.L:
#            self.tagger.change_x = 0

        if TOGGLE_FREE_AGENT:
            if key == arcade.key.W or key == arcade.key.S:
                self.toggled_agent.change_y = 0
            elif key == arcade.key.A or key == arcade.key.D:
                self.toggled_agent.change_x = 0

    def check_freeze(self):
        """ Checks if tagger can freeze an agent """
        agent_to_freeze = arcade.get_closest_sprite(self.tagger, self.free_agents)[0]
        x_pos = agent_to_freeze.center_x
        y_pos = agent_to_freeze.center_y

        if abs(x_pos - self.tagger.center_x) < 50 and abs(y_pos - self.tagger.center_y) < 50:
            self.frozen_agents.append(agent_to_freeze)
            self.reward -= 1

    def check_unfreeze(self):
        """ Checks if an unfrozen agent can 'rescue' a frozen agent """
        for free_agent in self.free_agents:
            if not free_agent in self.frozen_agents.sprite_list:
                if (len(self.frozen_agents.sprite_list) > 0):
                    agent_to_unfreeze = arcade.get_closest_sprite(free_agent, self.frozen_agents)[0]
                    x_pos = agent_to_unfreeze.center_x
                    y_pos = agent_to_unfreeze.center_y

                    if abs(x_pos - free_agent.center_x) < 50 and abs(y_pos - free_agent.center_y) < 50:
                        self.frozen_agents.remove(agent_to_unfreeze)
                        self.reward += 1

    def move_tagger(self, x):
        if x == "1":
            self.tagger.center_y += MOVEMENT_SPEED
        elif x == "2":
            self.tagger.center_x += MOVEMENT_SPEED
        elif x == "3":
            self.tagger.center_y += -MOVEMENT_SPEED
        elif x == "4":
            self.tagger.center_x += -MOVEMENT_SPEED

    def update(self, delta_time):
        """ Updates every frame of the game """

        # Game is over
        if self.index > TIME_LIMIT:
            for agent in self.free_agents.sprite_list:
                if agent not in self.frozen_agents.sprite_list:
                    self.reward += 1

            print("Final reward is", self.reward)
            sys.exit()

        if self.index < len(self.commands):
            self.move_tagger(self.commands[self.index])

        self.tagger_physics.update()
        self.check_freeze()
        self.check_unfreeze()
        self.index = self.index + 1

        # Seems to be a bottleneck for rendering game
        image = arcade.get_image()
        curr_state = list(image.getdata()) # List of (R, G, B, S) states as input to model

        self.model.input(curr_state)

        actions = self.model.actions()

        for free_agent in self.free_agents:
            if free_agent != self.toggled_agent:
                if free_agent in self.frozen_agents.sprite_list:
                    free_agent.change_x = 0
                    free_agent.change_y = 0
                else:
                    free_agent.change_x = random.randint(-MOVEMENT_SPEED, MOVEMENT_SPEED)
                    free_agent.change_y = random.randint(-MOVEMENT_SPEED, MOVEMENT_SPEED)

            arcade.PhysicsEngineSimple(free_agent, self.wall_list).update()

def main():
    window = FreezeTagGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()

if __name__ == "__main__":
    main()
