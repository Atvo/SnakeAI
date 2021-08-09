# Reinforcement learning and Snake
This project is an experiment I did a few years ago (~2018). I tried to create a self-learning ML model that plays the classic Snake game. As an inspiration I used [this blog post](https://blog.coast.ai/using-reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-6e782cc7d4c6#.h62resb0o).

## Description
The goal of the experiment was to use reinforcement learning and create an "intelligent" agent that plays the classic mobile game Snake. Each time the snake is going towards the food object or eats the object, the agent gets a positive reward. If the snake hits the walls or itself and "dies", the agent gets a negative reward. As an input the agent uses "sensors" that can see right in front, left, right, and diagonally. These sensors detect how far away the food, walls, or the snake itself is. The output of these sensors is used as an input for a neural network that is trained and tries to predict which action (turn left, turn right, do nothing) maximizes the reward.

## Outcome
As expected at the start of the exploration phase the actions of the agent are random and the scores are low.
![At the start of the exploration phase actions are random](https://github.com/Atvo/SnakeAI/blob/main/snake_start.gif)

When the algorithm gradually moves from the exploration phase to the exploitation phase the behavior starts to seem much more rational. The following gif is from the end where the agent tries to choose the optimal action ~99.99% of the time.
![The agent behaves rationally and the snake eats food objects](https://github.com/Atvo/SnakeAI/blob/main/snake_end.gif)

However, because the agent lacks the "complete picture", it can sometimes choose actions that will result in situations where it is impossible to survive. One such example can be seen below in which the snake's tail creates a dead end and the agent loses. Because of situations like these the maximum scores were typically around 55, which is still considerably lower than a typical human player would get after some training.

![The snake's tail creates a dead end](https://github.com/Atvo/SnakeAI/blob/main/snake_fail.gif)

## Running the experiment

If you want to run the experiment (and tweak the internal parameters) by yourself you can do it by following these steps:

1. Clone the project
2. Create a virtual environment
3. Install Pip packages from the requirements.txt
4. Run the main.py script