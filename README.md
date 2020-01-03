# Welcome to AID island

Your autonomous robot got stranded on a deserted island at a random position [x0, y0].
 
![logo](./assets/island.png)

Luckily, the programmer gave the robot the coordinates of the  coordinates [goal_x, goal_y] 
where the robot will find access to internet to call for help.

Please help the robot find its way to his goal.

## Motion Model
The robot can move in a 2D grid on the island. 
Its state is s =[x,y, goal_x, goal_y] and indicates its position on the map as well as also the goal location. 
At each time-step the robot can move to any adjacent or diagonal cell and will consume some amount of fuel.
For example, an action a=[1,1] will move diagonally up-right and consume sqrt(2) fuel.
 
As the robot was not made for maneuvering such a slopy terrain, it might fail its action with some unknown probability if the slope is too high, still consuming fuel.

If the robot falls into water or other terrain which is non-traversable, the robot is destroyed and needs to try again. 

## Task

Your task is to train a policy using reinforcement learning that will safely steer our robot to a random goal location, given a random start position

All the logic described above is already implemented in ```  aid_gym/island_env.py ```
and is not supposed to be changed.

You find an example usage of the API in ```main.py```. 
Please feel free to modify this to your needs.

We will not only judge the resulting algorithm, but also the design and the design choices while implementing.
Please provide some form of documentation explaining your design rationale.

 