import sys, pygame, time
import csv

import math
import random
import numpy as np
from nn import neural_net, LossHistory

gui = True

pygame.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)

area_size = area_width, area_height = 30, 20
cell_size = 10
screen_size = screen_width, screen_height = area_width * cell_size, area_height * cell_size
speed = [2, 2]
black = 0, 0, 0
white = 255, 255, 255
yellow = 255, 255, 0
blue = 0, 0, 255
green = 0, 255, 0
red = 255, 0, 0
screen_array = None

num_input = 6
gamma = 0.1
nn_param = [128, 128]
params = {
    "batchSize": 128,
    "buffer": 200000,
    "nn": nn_param
}
batchSize = params['batchSize']
buffer = params['buffer']

die_reward = -2000

t = 0

train_frames = 20000
observe = 1000  # Number of frames to observe before training.
epsilon = 1
data_collect = []
model = neural_net(num_input, nn_param)
replay = []  # stores tuples of (S, A, R, S').

loss_log = []
action_log = [0,0,0,0]
all_action_log = [0,0,0,0]

score = 0
score_sum = 0
games = 0

screen = pygame.display.set_mode(screen_size)

snake_obj = None
food_obj = None
direction = None
run_time = None
eat_time = None
games = 0.0

# Reset game status after the snake has crashed
def restart(snake_obj, food_obj, direction, score, run_time, games, eat_time):
	direction = random.choice(["up", "down", "right", "left"])
	
	# Initialize snake object
	snake_obj = [[random.randint(int(0.2 * (area_width-1)), int(0.8 * (area_width-1))), random.randint(int(0.2 * (area_height-1)), int(0.8 * (area_height-1)))]]
	if direction == "up":
		snake_obj.append([snake_obj[0][0],snake_obj[0][1]+1])
		snake_obj.append([snake_obj[0][0],snake_obj[0][1]+2])
	if direction == "down":
		snake_obj.append([snake_obj[0][0],snake_obj[0][1]-1])
		snake_obj.append([snake_obj[0][0],snake_obj[0][1]-2])
	if direction == "right":
		snake_obj.append([snake_obj[0][0]-1,snake_obj[0][1]])
		snake_obj.append([snake_obj[0][0]-2,snake_obj[0][1]])
	if direction == "left":
		snake_obj.append([snake_obj[0][0]+1,snake_obj[0][1]])
		snake_obj.append([snake_obj[0][0]+2,snake_obj[0][1]])

	# Initialize food object
	food_obj = None
	while food_obj == None or food_obj in snake_obj:
		food_obj = [random.randint(0, area_width - 1), random.randint(0, area_height - 1)]

	# Initialize score
	score = 0
	run_time = 0
	eat_time = 0
	games += 1

	return snake_obj, food_obj, direction, score, run_time, games, eat_time

# Get a list of the coordinates that the snake sees in a given direction
def get_ray(ray_dir, snake_obj, width, height):
	ray = []
	tmp_x = snake_obj[0][0]
	tmp_y = snake_obj[0][1]
	if ray_dir == "left":
		while tmp_x >= 0:
			ray.append([tmp_x, tmp_y])
			tmp_x -= 1
	if ray_dir == "upleft":
		while tmp_x >= 0 and tmp_y >= 0:
			ray.append([tmp_x, tmp_y])
			tmp_x -= 1
			tmp_y -= 1
	if ray_dir == "up":
		while tmp_y >= 0:
			ray.append([tmp_x, tmp_y])
			tmp_y -= 1
	if ray_dir == "upright":
		while tmp_x < width and tmp_y >= 0:
			ray.append([tmp_x, tmp_y])
			tmp_x += 1
			tmp_y -= 1
	if ray_dir == "right":
		while tmp_x < width:
			ray.append([tmp_x, tmp_y])
			tmp_x += 1
	if ray_dir == "downright":
		while tmp_x < width and tmp_y < height:
			ray.append([tmp_x, tmp_y])
			tmp_x += 1
			tmp_y += 1
	if ray_dir == "down":
		while tmp_y < height:
			ray.append([tmp_x, tmp_y])
			tmp_y += 1
	if ray_dir == "downleft":
		while tmp_x >= 0 and tmp_y < height:
			ray.append([tmp_x, tmp_y])
			tmp_x -= 1
			tmp_y += 1

	if len(ray) == 0:
		return None

	del ray[0]
	return ray

# Checks if the snake sees food, wall, or itself in a given set of coordinates and returns the distance from the snake's head
def get_reading(ray, snake_obj, food_obj, get_food):
	dist = 0

	if ray == None:
		return dist

	# Check how far away the food is (if the food)
	if get_food:
		for step in ray:
			dist += 1
			if step[0] == food_obj[0] and step[1] == food_obj[1]:
				return dist
			if step in snake_obj[:-1]:
				return 0
		return 0

	for step in ray:
		if step in snake_obj[:-1]:
			return dist
		dist += 1
	return dist

# Reference: https://github.com/harvitronix/reinforcement-learning-car/blob/6d1007410485ee7caaf0cace8d889c768a311a41/learning.py#L133
def process_minibatch2(minibatch, model):
	# by Microos, improve this batch processing function 
	#   and gain 50~60x faster speed (tested on GTX 1080)
	#   significantly increase the training FPS

	# instead of feeding data to the model one by one, 
	#   feed the whole batch is much more efficient

	mb_len = len(minibatch)

	old_states = np.zeros(shape=(mb_len, 6))
	actions = np.zeros(shape=(mb_len,))
	rewards = np.zeros(shape=(mb_len,))
	new_states = np.zeros(shape=(mb_len, 6))

	for i, m in enumerate(minibatch):
		old_state_m, action_m, reward_m, new_state_m = m
		old_states[i, :] = old_state_m[...]
		actions[i] = action_m
		rewards[i] = reward_m
		new_states[i, :] = new_state_m[...]

	old_qvals = model.predict(old_states, batch_size=mb_len)
	new_qvals = model.predict(new_states, batch_size=mb_len)

	maxQs = np.max(new_qvals, axis=1)
	y = old_qvals
	non_term_inds = np.where(rewards != die_reward)[0]
	term_inds = np.where(rewards == die_reward)[0]

	y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (gamma * maxQs[non_term_inds])
	y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

	X_train = old_states
	y_train = y
	return X_train, y_train

def write_to_file(filename, screen_array):
	f = open(filename, 'w')
	ret = ""
	for i in range(len(screen_array)):
		for k in range(len(screen_array[i])):
			ret += screen_array[i][k]
		ret += "\n"
	f.write(ret)
	f.close()


snake_obj, food_obj, direction, score, run_time, games, eat_time = restart(snake_obj, food_obj, direction, score, run_time, games, eat_time)
max_score = 0
max_time = 0
start_time = time.time()

while 1 and t < train_frames:
	t += 1
	run_time += 1
	eat_time += 1
	if run_time > max_time:
		max_time = run_time
	reward = 0

	if random.random() < epsilon or t < observe:
		action = np.random.randint(0, 3)  # random
	else:
		# Get Q values for each action.
		qval = model.predict(state, batch_size=1)
		action = (np.argmax(qval))
		action_log[action] += 1
	all_action_log[action] += 1

	# Give small negative reward each time the snake turns
	if action == 0:
		reward = -5
		if direction == "up":
			new_direction = "left"
		if direction == "right":
			new_direction = "up"
		if direction == "down":
			new_direction = "right"
		if direction == "left":
			new_direction = "down"
	elif action == 2:
		reward = -5
		if direction == "up":
			new_direction = "right"
		if direction == "right":
			new_direction = "down"
		if direction == "down":
			new_direction = "left"
		if direction == "left":
			new_direction = "up"
	else:
		new_direction = direction
	direction = new_direction

	## CONTROLS

	for event in pygame.event.get():
		if event.type == pygame.QUIT: sys.exit()
	if event.type == pygame.KEYDOWN:
		if event.key == pygame.K_UP and direction in ["left", "right"]:
			direction = "up"
		if event.key == pygame.K_DOWN and direction in ["left", "right"]:
			direction = "down"
		if event.key == pygame.K_LEFT and direction in ["up", "down"]:
			direction = "left"
		if event.key == pygame.K_RIGHT and direction in ["up", "down"]:
			direction = "right"

	## UPDATE STATE
	if direction == "up":
		snake_obj.insert(0, [snake_obj[0][0], snake_obj[0][1]-1])
	if direction == "down":
		snake_obj.insert(0, [snake_obj[0][0], snake_obj[0][1]+1])
	if direction == "left":
		snake_obj.insert(0, [snake_obj[0][0]-1, snake_obj[0][1]])
	if direction == "right":
		snake_obj.insert(0, [snake_obj[0][0]+1, snake_obj[0][1]])


	## STATE READING
	# Get ray coordinates
	ray_left = get_ray("left", snake_obj, area_width, area_height)
	ray_upleft = get_ray("upleft", snake_obj, area_width, area_height)
	ray_up = get_ray("up", snake_obj, area_width, area_height)
	ray_upright = get_ray("upright", snake_obj, area_width, area_height)
	ray_right = get_ray("right", snake_obj, area_width, area_height)
	ray_downright = get_ray("downright", snake_obj, area_width, area_height)
	ray_down = get_ray("down", snake_obj, area_width, area_height)
	ray_downleft = get_ray("downleft", snake_obj, area_width, area_height)

	# Check how far the wall is in each direction
	wall_read_left = get_reading(ray_left, snake_obj, food_obj, False)
	wall_read_upleft = get_reading(ray_upleft, snake_obj, food_obj, False)
	wall_read_up = get_reading(ray_up, snake_obj, food_obj, False)
	wall_read_upright = get_reading(ray_upright, snake_obj, food_obj, False)
	wall_read_right = get_reading(ray_right, snake_obj, food_obj, False)
	wall_read_downright = get_reading(ray_downright, snake_obj, food_obj, False)
	wall_read_down = get_reading(ray_down, snake_obj, food_obj, False)
	wall_read_downleft = get_reading(ray_downleft, snake_obj, food_obj, False)

	# Check how far the food is in each direction
	food_read_left = get_reading(ray_left, snake_obj, food_obj, True)
	food_read_upleft = get_reading(ray_upleft, snake_obj, food_obj, True)
	food_read_up = get_reading(ray_up, snake_obj, food_obj, True)
	food_read_upright = get_reading(ray_upright, snake_obj, food_obj, True)
	food_read_right = get_reading(ray_right, snake_obj, food_obj, True)
	food_read_downright = get_reading(ray_downright, snake_obj, food_obj, True)
	food_read_down = get_reading(ray_down, snake_obj, food_obj, True)
	food_read_downleft = get_reading(ray_downleft, snake_obj, food_obj, True)

	# Map the absolute food and wall readings to relative readings
	if direction == "up":
		wall_read_rel_left = wall_read_left
		wall_read_rel_straight = wall_read_up
		wall_read_rel_right = wall_read_right
		food_read_rel_left = food_read_left
		food_read_rel_straight = food_read_up
		food_read_rel_right = food_read_right
	if direction == "left":
		wall_read_rel_straight = wall_read_left
		wall_read_rel_left = wall_read_down
		wall_read_rel_right = wall_read_up
		food_read_rel_straight = food_read_left
		food_read_rel_left = food_read_down
		food_read_rel_right = food_read_up
	if direction == "down":
		wall_read_rel_straight = wall_read_down
		wall_read_rel_left = wall_read_right
		wall_read_rel_right = wall_read_left
		food_read_rel_straight = food_read_down
		food_read_rel_left = food_read_right
		food_read_rel_right = food_read_left
	if direction == "right":
		wall_read_rel_straight = wall_read_right
		wall_read_rel_left = wall_read_up
		wall_read_rel_right = wall_read_down
		food_read_rel_straight = food_read_right
		food_read_rel_left = food_read_up
		food_read_rel_right = food_read_down

	if t == 1:
		state = [wall_read_rel_left, wall_read_rel_straight, wall_read_rel_right, food_read_rel_left, food_read_rel_straight, food_read_rel_right]
		state = np.array([state])
	new_state = [wall_read_rel_left, wall_read_rel_straight, wall_read_rel_right, food_read_rel_left, food_read_rel_straight, food_read_rel_right]
	new_state = np.array([new_state])

	# Eat food: Increase score, generate new food object
	if snake_obj[0] == food_obj:
		score += 1
		score_sum += 1
		if score > max_score:
			max_score = score
		reward = max(1000 - eat_time, 200)
		food_obj = None
		eat_time = 0
		while food_obj == None or food_obj in snake_obj:
			food_obj = [random.randint(0, area_width -1), random.randint(0, area_height - 1)]
	# Continue without eating food
	else:
		# Give small reward if the food is straight in front of the snake
		if food_read_rel_straight > 0:
			reward += 100 - food_read_rel_straight
		del snake_obj[-1]

	# Die if snake hits itself
	if snake_obj[0] in snake_obj[1:]:
		reward = die_reward
		snake_obj, food_obj, direction, score, run_time, games, eat_time = restart(snake_obj, food_obj, direction, score, run_time, games, eat_time)
	# Die if snake hits wall
	if snake_obj[0][0] < 0 or snake_obj[0][1] < 0 or snake_obj[0][0] >=area_width or snake_obj[0][1] >= area_height:
		snake_obj, food_obj, direction, score, run_time, games, eat_time = restart(snake_obj, food_obj, direction, score, run_time, games, eat_time)
		reward = die_reward

	## Append to replay storage
	replay.append((state, action, reward, new_state))

	## If we're done observing, start training
	if t > observe:

		# If we've stored enough in our buffer, pop the oldest.
		if len(replay) > buffer:
		    replay.pop(0)

		# Randomly sample our experience replay memory
		minibatch = random.sample(replay, batchSize)

		# Get training values.
		X_train, y_train = process_minibatch2(minibatch, model)

		# Train the model on this batch.
		history = LossHistory()
		model.fit(
		    X_train, y_train, batch_size=batchSize,
		    epochs=1, verbose=0, callbacks=[history]
		)
		loss_log.append(history.losses)

	state = new_state

	# Decrement epsilon over time.
	if epsilon > 0.001 and t > observe:
	    epsilon -= (1.0/train_frames*2)


	# Print stats periodically
	if t % 5000 == 0:
		print("\nt: " + str(t))
		print("max_score: " + str(max_score))
		print("max_time: " + str(max_time))
		print("avg_score: " + str(score_sum / games))
		print("avg_time: " + str(t / games))
		print("run time: " + str(int(time.time() - start_time)))
		print("epsilon: " + str(epsilon))
		print("up,down,left,right")
		print("action_log: " + str(action_log))
		print("all_action_log: " + str(all_action_log))


	if gui:

		screen.fill(white)

		scoreSurface = myfont.render(str(score), False, (0, 0, 0))
		screen.blit(scoreSurface,(20,20))

		pygame.draw.rect(screen,red,(food_obj[0]*cell_size,food_obj[1]*cell_size,cell_size,cell_size ))
		
		pygame.draw.rect(screen,yellow,(snake_obj[0][0]*cell_size,snake_obj[0][1]*cell_size,cell_size,cell_size ))
		for snake_piece in snake_obj[1:]:
			pygame.draw.rect(screen,blue,(snake_piece[0]*cell_size,snake_piece[1]*cell_size,cell_size,cell_size ))

		pygame.display.flip()

