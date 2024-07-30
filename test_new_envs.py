from simpleenvs.envs import discrete_rooms 


env = discrete_rooms.SixRooms(movement_penalty=0)

state = env.reset()

for step in range(100):
    action = env.get_available_actions().sample()
    state, reward, terminal, info = env.step(action)

    if terminal:
        state = env.reset()

    env.render()

env.close()