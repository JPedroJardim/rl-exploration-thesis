from pyboy import PyBoy, WindowEvent
import keyboard

pyboy = PyBoy('../rom/pokemon_red.gb', window_type="SDL2", window_scale=3, debug=True, game_wrapper=True)
pyboy.set_emulation_speed(1)

poke_red = pyboy.game_wrapper()
poke_red.start_game()

#pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
for _ in range(10000):
    if keyboard.is_pressed('esc'):
        break
    elif keyboard.is_pressed('n'):
        pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        print('Pressed N')
    elif keyboard.is_pressed('m'):
        pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
        print('Pressed M')
    elif keyboard.is_pressed('enter'):
        pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    elif keyboard.is_pressed('backspace'):
        pyboy.send_input(WindowEvent.PRESS_BUTTON_SELECT)
    elif keyboard.is_pressed('left arrow'):
        pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
    elif keyboard.is_pressed('right arrow'):
        pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    elif keyboard.is_pressed('up arrow'):
        pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
    elif keyboard.is_pressed('down arrow'):
        pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    
    pyboy.tick()
    

pyboy.stop()


# example of config
env_config = {
                'headless': True, 
                'save_final_state': True, 
                'early_stop': False,
                'action_freq': 24, 
                'init_state': '../has_pokedex_nballs.state', 
                'max_steps': ep_length, 
                'print_rewards': True, 
                'save_video': False, 
                'fast_video': True, 
                'session_path': sess_path,
                'gb_path': '../PokemonRed.gb',
                'debug': False, 
                'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True,
                'extra_buttons': False
            }