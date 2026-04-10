
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import pygame


WORLD_SIZE = 1
DIAMETER = 0.02
MAX_VELOCITY = 1
MAX_ACCELERATION = 1
MAX_STEPS = 20000
TIME_STEP = 0.001
SHADOW_DIST = 0.3 # ignore shadow


class RoboBuntingEnv(MultiAgentEnv):
    metadata = {"render_modes": ["human"], "name": "perceptual_crossing_v0", "render_fps": 30}

    def __init__(self, config=None,
                 world_size=WORLD_SIZE,
                 diameter=DIAMETER, 
                 max_steps=MAX_STEPS,
                 max_vel = MAX_VELOCITY,
                 max_acc = MAX_ACCELERATION,
                 time_step = TIME_STEP,
                 shadow_distance = SHADOW_DIST,
                 seed = None
                 ):
        super().__init__()

        # agents in environment; we store the states (position, velocity) with these keys
        self.possible_agents = ["p1", "p2"]

        # configure environment
        self.world_size = world_size
        self.diameter = diameter
        self.max_steps = max_steps
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.dt = time_step

        #self.shadow_distance = shadow_distance # ignore shadow for now
        self.hist_buffer = 4 # number of past steps to include in observation
        

        self.observation_space = {agent: gym.spaces.Box(low=-1, high=1, shape=(self.hist_buffer*4,), dtype=np.float32) for agent in self.possible_agents}
        self.action_space = {agent : gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) for agent in self.possible_agents}

        self.step_count = 0 # track number of steps in current episode
    

        # rendering screen
        self.render_mode = 'human'
        self.screen = None
        self.clock = None
        self.canvas_size = 600
        self.ring_radius = 200
        self.agent_ui_size = 30 # Visual size in pixels

    def reset(self, *, seed=None, options=None):
        '''
        Reset the environment to an initial state and return an initial observation and info dict.
        '''
        self.agents = self.possible_agents[:]

        self.pos = {agent: np.random.uniform(0, self.world_size) for agent in self.agents}
        self.vel = {agent: 0.0 for agent in self.agents}


        sensors = {a: 0.0 for a in self.agents}
        for i, a in enumerate(self.agents):
            other = self.agents[1 - i]

            targets = [
                self.pos[other],                               # the other 
                # (self.pos[other] + self.shadow_distance) % self.world_size, # The shadow
            ]
            
            if any(self.__distance(self.pos[a], t) <= self.diameter for t in targets):
                sensors[a] = 1.0 # contact sensor activated if in contact with other

        # memory buffer; initialize with current state
        self.acc_hist       = {agent: [0.0]*self.hist_buffer for agent in self.agents}
        self.vel_hist       = {agent: [0.0]*self.hist_buffer for agent in self.agents}
        self.pos_hist       = {agent: [self.pos[agent]/self.world_size]*self.hist_buffer for agent in self.agents}
        self.contact_hist   = {agent: [sensors[agent]]*self.hist_buffer for agent in self.agents}

        # calculate reward
        self.internal_rewards = {agent : 0.0 for agent in self.agents}


        observations = self.__get_obs()
        infos = {agent: self.__get_info() for agent in self.agents}

        self.step_count = 0

        return observations, infos
    
    
    def step(self, actions):
        self.step_count += 1 
        self.agents = self.possible_agents[:]

        # physics update; integrate acceleration to get velocity, then integrate velocity to get position
        for agent in self.agents:
            acc = actions[agent][0] * self.max_acc
            self.vel[agent] = np.clip(self.vel[agent] + acc * self.dt, -self.max_vel, self.max_vel)
            self.pos[agent] = (self.pos[agent] + self.vel[agent] * self.dt) % self.world_size

        # calculate the distance between agents and determine if they are in contact (overlap)
        dist = self.__distance(self.pos[self.possible_agents[0]], self.pos[self.possible_agents[1]])
        overlap = 0
        if dist < self.diameter:
            overlap = 1 - dist/(self.world_size)

        sensors = {a: 0.0 for a in self.agents}
        for i, a in enumerate(self.agents):
            other = self.agents[1 - i]
            targets = [
                self.pos[other],                               # the other 
                # (self.pos[other] + self.shadow_distance) % self.world_size, # The shadow
            ]
            
            if any(self.__distance(self.pos[a], t) <= self.diameter for t in targets):
                sensors[a] = 1.0

        # memory buffer; update memory with new values
        for agent in self.agents:
            acc = actions[agent][0]

            self.acc_hist[agent]        = [acc]                             + self.acc_hist[agent][0:self.hist_buffer-1] # add new values to buffer
            self.vel_hist[agent]        = [self.vel[agent]/self.max_vel]    + self.vel_hist[agent][0:self.hist_buffer-1]
            self.pos_hist[agent]        = [self.pos[agent]/self.world_size] + self.pos_hist[agent][0:self.hist_buffer-1]
            self.contact_hist[agent]    = [sensors[agent]]                  + self.contact_hist[agent][0:self.hist_buffer-1]

        

        # observations
        observations = self.__get_obs()

        # contact_shadow_t_ = {
        #     self.possible_agents[0] : self.__distance(self.pos[self.possible_agents[0]], (self.pos[self.possible_agents[1]] + self.shadow_distance) % self.world_size), # <= self.diameter,
        #     self.possible_agents[1] : self.__distance(self.pos[self.possible_agents[1]], (self.pos[self.possible_agents[0]] + self.shadow_distance) % self.world_size), #<= self.diameter
        # }

        for agent in self.agents:

            contact_t =  dist
            energy_t = (self.vel[agent]*self.vel[agent])/(self.max_vel*self.max_vel)

            self.internal_rewards[agent] = contact_t  - energy_t

        # reward contact
        rewards = self.internal_rewards

        # termination and info
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}

        terminations = {"__all__": self.step_count >= self.max_steps}
        infos = {agent: self.__get_info() for agent in self.agents}


        return observations, rewards, terminations, truncations, infos
    
    def __distance(self, a, b):
        diff = abs(a - b)
        return min(diff, self.world_size - diff)
    

    def __get_obs(self):
        obs = {}
        self.agents = self.possible_agents[:]

        for agent in self.agents:
            obs[agent] = np.array(self.contact_hist[agent] + self.pos_hist[agent] + self.vel_hist[agent] + self.acc_hist[agent])

        return obs

    
    def __get_info(self):
        info = { # info for debugging
            'p1_pos': self.pos[self.possible_agents[0]],
            'p2_pos': self.pos[self.possible_agents[1]],
            'p1_v': self.vel[self.possible_agents[0]],
            'p2_v': self.vel[self.possible_agents[1]],
            'contact': self.__get_obs()[self.possible_agents[0]][0]
        }
        return info
    

    def render(self, bar_len=20):
        '''
        Basic render function to visualize the positions of the agents on a 1D ring.
        P1 and P2 are the agents, represented on a line of length `bar_len`.
        '''

        p1_idx = int((self.pos[self.possible_agents[0]]/ self.world_size) * (bar_len - 1))
        p2_idx = int((self.pos[self.possible_agents[1]] / self.world_size) * (bar_len - 1))


        line = ["."] * bar_len
        line[p1_idx] = "P1"
        line[p2_idx] = "P2"

        # p1_s = (self.pos[self.possible_agents[0]] + self.shadow_distance) % self.world_size
        # p2_s = (self.pos[self.possible_agents[1]] + self.shadow_distance) % self.world_size
        # p1_s_idx = int((p1_s/ self.world_size) * (bar_len - 1))
        # p2_s_idx = int((p2_s/ self.world_size) * (bar_len - 1))
        # line[p1_s_idx] = "1s"
        # line[p2_s_idx] = "2s"
        
        print("".join(line))

    def render_screen(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.canvas_size, self.canvas_size))
            pygame.display.set_caption("Perceptual Crossing Environment")
            self.clock = pygame.time.Clock()

        # 1. Reset Canvas
        canvas = pygame.Surface((self.canvas_size, self.canvas_size))
        canvas.fill((255, 255, 255))
        
        # 2. Draw Environment Ring
        pygame.draw.circle(canvas, (220, 220, 220), (self.canvas_size/2, self.canvas_size/2), self.ring_radius, width=2)

        # 3. Render Agents and Shadows
        colors = {"p1": (190, 18, 27), "p2": (25, 82, 184)}
        is_inside = {"p1": False, "p2": True}
        offset = {"p1": 0, "p2": 0}
        for agent_id in self.possible_agents:
            pos = self.pos[agent_id]
            color = colors[agent_id]
            
            # Agent Physics -> Circle Geometry
            angle = (pos / self.world_size) * 2 * np.pi
            coords = (self.ring_radius + offset[agent_id]) * np.array([np.cos(angle), np.sin(angle)])
            self.__draw_robot(canvas, coords, color, angle, inside=is_inside[agent_id])
            
            # # Shadow Mapping
            # s_pos = (pos + self.shadow_distance) % self.world_size
            # s_angle = (s_pos / self.world_size) * 2 * np.pi
            # s_coords = (self.ring_radius + offset[agent_id]) * np.array([np.cos(s_angle), np.sin(s_angle)])
            # self.__draw_robot(canvas, s_coords, color, s_angle, is_shadow=True, inside=is_inside[agent_id])

        # 4. Update Display
        self.screen.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def __draw_robot(self, surface, center, color, angle, is_shadow=False, inside=False):
        """Internal helper to draw agents with sensors and rotation."""
        mid = np.array([self.agent_ui_size / 2, self.agent_ui_size / 2])
        agent_surf = pygame.Surface((self.agent_ui_size, self.agent_ui_size), pygame.SRCALPHA)
        
        # body & border
        alpha = 120 if is_shadow else 255
        pygame.draw.circle(agent_surf, (*color, alpha), mid, radius=self.agent_ui_size/2)
        pygame.draw.circle(agent_surf, (0, 0, 0), mid, radius=self.agent_ui_size/2, width=2)
        
        # sensor (yellow for real agent, Gray for shadow)
        sensor_color = (228, 213, 29) if not is_shadow else (150, 150, 150)
        pygame.draw.rect(agent_surf, sensor_color, [0, 0, self.agent_ui_size/3, self.agent_ui_size], border_radius=4)
        
        # rotate and blit to center
        ang_degree = np.degrees(-angle)
        if inside:
            ang_degree += 180
        
        rotated = pygame.transform.rotate(agent_surf, ang_degree)
        origin = np.array([self.canvas_size/2, self.canvas_size/2])
        blit_pos = (center + origin) - np.array(rotated.get_size()) / 2
        surface.blit(rotated, blit_pos)
