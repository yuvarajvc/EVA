Phase 2 Session 10 End Game Assignment

## Environment Stack
- Kivy (map.py and car.kv) - for the city map, car and define other environment conditions
- Pytorch (The T3D Model) ai.py file

## The Goal of the assignment is to :

1. Train a car to move around the city and reach a defined destination.
2. Car should follow the road and not hit sand or walls or water bodies.

## Parameters Used
1. Observation Space - Here we are using image of the car moving in the map as the observation space or state. The car location is know by car.x and car.y and surrounding pixels are cropped of the sand image. Then the car image from the folder is taken and resized to 10,20 as per the and overlayed on top of the car image using pixel calculations. Necessary rotation of the car is determined by Vector(*self.car.velocity).angle((xx,yy). This value is fed to the actor model  (CNN 28*28) as observation space or state. *(Note  I still need to tune the model and possible use kernels or 3*3)
*
The image processing logic is written under get_obs function in map.py

2. max action - Max action is the max magnitude of angle of rotation of the car. In our case we are keeping it as 5. 

3. The action dimension - The action dimension is between -5 0 5. To get continous values as output from the T3D algorithm, we have tanh at the last layer to multiply with max action. so the output will be a continous action space between -5 and 5 (angle of rotation of the car)

### Replay Buffer
In this assignment the replay buffer contains

1. current observation - The current state of the car in the road
2. new observation, - The state of the car after taking a particular action. 
3. action - The action taken through following steps
	random.randrange(-5,5)*random.random() until start time steps
	policy.select_action(np.array(observationspace)) after start time steps.after sufficient memory is built.
	The selected action is passed through self.step(action,last_distance) function, make the necessary action on the environment/game and get new_obs, reward and done flags.
4. reward. - The Reward for taking the specific action in the environment. *I'm still playing tuning with the different rewards to reward and penalize certain actions. This is work in progress.*
5. Done -  Flag to specify if the taking a particular action resulted in end of an episode or not. In the assignment environment, reaching the goal, reaching the walls or reaching a certain number of steps without any achievement of goal are considered as done = True.

## Accomplished activities:

1. Able to integrate kivy environment with T3D.
2. Able to crop the pixels around the car, overlay the car image on the cropped image and get the orientation of the car in the car image,
3. The car is running on the map.

## Issues faced:

1. The car is running and hitting the wall and getting struck.
2. The car is not going towards the goal.
3. Training is slow.

## Things Planning to do if given an extension

1. Determine correct rewards to penalize going towards the wall, going on the sand,living penalty etc. 
2. Determine correct rewards to incentivize staying on the road and making steps towards the goal.
3. Tune the actor model with correct weights.
4. pad the image to ensure the car going towards the border is handled correctly.
5. put a triangle on top the car so that orientation of the car towards the goal can be determined by the algorithm

   <u></u>