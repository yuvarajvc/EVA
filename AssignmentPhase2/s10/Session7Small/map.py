# Self Driving Car

# Importing the libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import torchvision.transforms as transforms

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen , ScreenManager

import pyscreenshot as ImageGrab
# Importing the Dqn object from our AI in ai.py
#from ai import Dqn, ObsSpaceNetwork
from ai import ReplayBuffer, Actor, Critic, TD3
from numpy import asarray


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
episode_reward = 0

torch.manual_seed(seed)
np.random.seed(seed)
state_dim = 5
action_dim = 1
max_action = 5
min_action = -5

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
#brain = Dqn(5,3,0.9)
action2rotation = [0,5,-5]
#spacenetwork = ObsSpaceNetwork()
policy = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
episode_timesteps = 0
done = True
t0 = time.time()

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
i = 0

def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global img
    #global ALPHA 
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1420
    goal_y = 622
    max_timesteps = 500000
    #ALPHA = 0.01
    global swap
    swap = 0
    
    
    


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    #sensor1_x = NumericProperty(0)
    #sensor1_y = NumericProperty(0)
    #sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    #sensor2_x = NumericProperty(0)
    #sensor2_y = NumericProperty(0)
    #sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    #sensor3_x = NumericProperty(0)
    #sensor3_y = NumericProperty(0)
    #sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    #signal1 = NumericProperty(0)
    #signal2 = NumericProperty(0)
    #signal3 = NumericProperty(0)

    def move(self, rotation):
        print("move")
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation


class Game(Widget):

    car = ObjectProperty(None)


    def serve_car(self):
        print("servecar")
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def get_obs(self, xx, yy):
        area = (self.car.x-100,self.car.y-100,self.car.x+100,self.car.y+100)
        iobj = img.rotate(90,expand=True).crop(area)
        #obj.save("xxafter" + str(i) +".png")
        img2 = PILImage.open("./images/car.png").resize((20,10))
        iobj.paste(img2.rotate(Vector(*self.car.velocity).angle((xx,yy)),expand=True),(90,95))
        iobj.thumbnail((28,28))
        #observationspace = spacenetwork.sceneforward(transforms.ToTensor()(iobj))
        return iobj

    def step(self, action, last_distance):
        global goal_x
        global goal_y
        global done
        self.car.move(action)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        done - False
        obs = self.get_obs(xx,yy)

        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                reward = 0.1
            # else:
            #     last_reward = last_reward +(-0.2)

        if self.car.x < 5:
            self.car.x = 5
            reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            reward = -1
            done = True
        if self.car.y < 5:
            self.car.y = 5
            reward = -1
            done = True
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            reward = -1
            done = True

        if distance < 25:
            done = True
            
        return obs, reward, done,distance
    
    def update(self,dt):

        #global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        #global i
        global last_reward
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global done
        global seed# Random seed number
        global start_timesteps  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
        global eval_freq  # How often the evaluation step is performed (after how many timesteps)
        global max_timesteps # Total number of iterations/timesteps
        global save_models  # Boolean checker whether or not to save the pre-trained model
        global expl_noise# Exploration noise - STD value of exploration Gaussian noise
        global batch_size  # Size of the batch
        global discount # Discount factor gamma, used in the calculation of the total discounted reward
        global tau # Target network update rate
        global policy_noise # STD of Gaussian noise added to the actions for the exploration purposes
        global noise_clip # Maximum value of the Gaussian noise added to the actions (policy)
        global policy_freq 
        global max_action
        global episode_reward
        global episode_timesteps

        

        longueur = self.width
        print("update1")
        largeur = self.height
        if first_update:
            init()
            print("update")
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            observationspace = self.get_obs(xx,yy)
 
        max_timesteps = 200
# We start the main loop over 500,000 timesteps
        #while total_timesteps < max_timesteps:
        print("total_timesteps:" + str(total_timesteps))
        if done:
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
            # When the training step is done, we reset the state of the environment
            #obs = env.reset()
            Game().serve_car()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        if total_timesteps < start_timesteps:
            #action = env.action_space.sample()
            action =  random.randrange(-5,5)*random.random()
            
            #self.car.move(action)  
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(np.array(observationspace))
    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(min_action, max_action)
                #self.car.move(action)  
        new_obs, reward, done, distance = self.step(action,last_distance)

        # We check if the episode is done
        #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

        # We increase the total reward
        episode_reward += reward
        
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((observationspace, new_obs, action, reward, done))
        
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        observationspace = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

        #action = brain.update(last_reward, observationspace)
        #scores.append(brain.score())
        #rotation = action2rotation[action]
        #self.car.move(rotation)
        
        last_distance = distance

    # def evaluate_policy(policy, eval_episodes=10):
    #     avg_reward = 0.
    #     for _ in range(eval_episodes):
    #         obs = env.reset()
    #         done = False
    #         while not done:
    #             action = policy.select_action(np.array(obs))
    #             obs, reward, done, _ = env.step(action)
    #             avg_reward += reward
    #     avg_reward /= eval_episodes
    #     print ("---------------------------------------")
    #     print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    #     print ("---------------------------------------")
    #     return avg_reward

    # def mkdir(base, name):
    #     path = os.path.join(base, name)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     return path
        

# Adding the painting tools

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        print("build")
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        #parent.update()
        
        return parent

# Running the whole thing
if __name__ == '__main__':
    print("1")
    CarApp().run()
