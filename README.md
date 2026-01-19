# dPickleball Training Strategy
_____

The dPickleball Environment: https://github.com/dPickleball/dPickleBallEnv
Our code provide some training strategy for dpickleball!

_____
### Training Strategy:
1. Curriculum Learning 
2. World Model 
3. Parallel Env Training 
4. Parallel Env + Currciculum Learning (the best)

——————
### Description of Training Strategy:
1. Curriculum Learning: Once the specified score is reached, switch to the next course (alternate between left and right sides).
2. World Model: Build an internal model so that the agent can be trained within the model.
3. Parallel Env Training: Training in multiple environments
4. Parallel Env + Curriculum Learning: Techniques combining multiple environments and curriculum learning
All the code is implemented by Pytorch and the algorithm is based on PPO.



### Note:
1. After training, you can find the Competition Script on the provided website. Once you've set up the models for the left and right players, you can start the match.
2. If WandB is not needed, please delete the relevant code.
3. Our code is all in ipynb, so it can be run after the relevant packages are set up.

If you like our project, please give us a star. Thank you!
