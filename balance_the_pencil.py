"""
TODO tidy up this fucking mess of code, theres so much copy paste, let it be functions and files
"""
import pygame as pg
import sys
import numpy as np
import os

import pencil as pencil_
import handmade_perceptron as hp


def render_text(Surface, text, size, color, centre_position):
    base_font=pg.font.Font(None,size)
    txt_surface = base_font.render(text, True, color)
    txt_rect = txt_surface.get_rect()
    Surface.blit(txt_surface, ((centre_position[0]-txt_rect[2]/2),
                               (centre_position[1]-txt_rect[3]/2)))

def restart_game():
    x_pos = screen_width/2 - 15
    base = pg.Rect(x_pos, screen_height * 2/3, 30, 5)

    pencil = pencil_.Pencil(tip_position=[360, 480])
    stopwatch = Stopwatch()

    #starting condition, could make it random
    pencil.angular_velocity += 0.001
    return base, pencil, stopwatch

class Stopwatch():
    def __init__(self):
        self.start_time = pg.time.get_ticks()
        self.display_time = 0
        self.paused = False

    def time(self):
        if not self.paused:
            self.display_time = (pg.time.get_ticks() - self.start_time)/1000
        return self.display_time

def render_neural_network(Surface, neural_network, position, scale=1,
                          x_stretch=1, input_labels=[], output_labels=[]):
    # arrange the positions of the neurons
    x_positions = []
    y_positions = []
    i = 0
    layer_sizes = ([len(neural_network.inputs)]
                   +neural_network.hl_sizes
                   +[len(neural_network.outputs)])
    n_layers = len(layer_sizes)
    x_stretch *= 2/n_layers
    for layer_size in layer_sizes:
        if i != len(neural_network.biases):
            layer_size += 1 # add room for bias neuron
        x_positions.append(np.full((layer_size), i) * x_stretch)
        y_positions.append(np.arange(-(layer_size)/2, (layer_size)/2, 1))
        i += 1

    # plot the weights, l is layer, j is output, k is input (in general)
    for l in range(len(neural_network.weights)):
        for j in range(len(neural_network.weights[l])):
            for k in range(len(neural_network.weights[l][j])):
                weight = neural_network.weights[l][j][k]
                input_pos = [x_positions[l][k]*50*scale + position[0],
                             y_positions[l][k]*50*scale + position[1]]
                output_pos = [x_positions[l+1][j]*50*scale + position[0],
                              y_positions[l+1][j]*50*scale + position[1]]
                color=('red' if weight>0 else 'blue')
                pg.draw.line(screen, color, input_pos, output_pos,
                             width=int(abs(weight*5)))

    # plot the biases
    for l in range(len(neural_network.biases)):
        for j in range(len(neural_network.biases[l])):
            # a neuron with activation 1 connected with weight=bias
            bias = neural_network.biases[l][j]
            bias_pos = [(x_positions[l][-1])*50*scale + position[0],
                        (y_positions[l][-1])*50*scale + position[1]]
            output_pos = [x_positions[l+1][j]*50*scale + position[0],
                          y_positions[l+1][j]*50*scale + position[1]]
            color=('red' if bias>0 else 'blue')
            pg.draw.line(screen, color, bias_pos, output_pos,
                         width=int(abs(bias*5)))
            # plot bias neuron
            pg.draw.circle(Surface, (255,255,255), bias_pos, 5*scale)

    # plot the neurons
    for layer in range(len(layer_sizes)):
        for i in range(layer_sizes[layer]):
            x = x_positions[layer][i]*50*scale + position[0]
            y = y_positions[layer][i]*50*scale + position[1]
            activation = ([neural_network.inputs]
                          +neural_network.hl_activations
                          +[neural_network.outputs])[layer][i]
            a = int(hp.sigmoid(activation)*255)
            color = (a,a,a)
            pg.draw.circle(Surface, color, [x,y], 5*scale)

    # label neurons
    input_labels.append('bias')
    for i in range(len(input_labels)): # +1 for bias "neuron"
        x = x_positions[0][i]*50*scale + position[0] - 10*scale
        y = y_positions[0][i]*50*scale + position[1]

        label = input_labels[i]
        base_font=pg.font.Font(None, 16*scale)
        txt_surface = base_font.render(label, True, (255,255,255))
        txt_rect = txt_surface.get_rect()
        Surface.blit(txt_surface, [x-txt_rect[2], y-txt_rect[3]/2])

    for i in range(len(output_labels)):
        x = x_positions[-1][i]*50*scale + position[0] + 10*scale
        y = y_positions[-1][i]*50*scale + position[1]

        label = output_labels[i]
        base_font=pg.font.Font(None, 16*scale)
        #color = 'green' if neural_network.outputs[i]>0.5 else 'white'
        color = 'white'
        txt_surface = base_font.render(label, True, color)
        txt_rect = txt_surface.get_rect()
        Surface.blit(txt_surface, [x, y-txt_rect[3]/2])

def NEAT(best_ai, ai):
    parents_index = np.random.permutation(range(len(best_ai)))[0:2]
    mother = best_ai[parents_index[0]]
    father = best_ai[parents_index[1]]
    r = np.random.uniform(0,0.6667,4) # such that weight stays the same on average
    new_weights = (r[0]*np.asarray(father['weights'])
                   + r[1]*np.asarray(mother['weights'])
                   + np.random.normal(0,2,ai.weights[0].shape))
    new_biases = (r[2]*np.asarray(father['biases'])
                   + r[3]*np.asarray(mother['biases'])
                   + np.random.normal(0,2,ai.biases[0].shape))
    ai.set_weights(new_weights)
    ai.set_biases(new_biases)
    return

# General setup
pg.init()
clock = pg.time.Clock()

# Setting up the main window
screen_width = 720
screen_height = 720
screen = pg.display.set_mode((screen_width,screen_height))
pg.display.set_caption('Balance the pencil')
# background color
background_color = (100,100,150)
screen.fill(background_color)

# Game setup
x_speed = 10
base, pencil, stopwatch = restart_game()

# Setup AI
# inputs = angle, angular_velocity, outputs = [move_left, move_right]
ai = hp.Perceptron(3, 1)
ai.randomize()
best_time = 0
enable_AI = False
tick_speed = 60
show_controls = False
auto_restart = False
loop_screen = False
ticks = 0
use_NEAT = True
restarts = 0
zero_biases = True
enable_kicks = True

if zero_biases:
    ai.set_biases(np.zeros( (np.asarray(ai.biases).shape) ))

best_ai = []

while True:
    ticks += 1
    # AI controlling the stuff
    if enable_AI:
        ai.set_inputs([pencil.angle*5,
                       pencil.tip_speed/20,
                       (2*pencil.tip_position[0]/screen_width - 1)])
        x_movement = int((ai.outputs[0]-0.5)*20)
        pencil.move_tip(x_movement)
        base.x += x_movement
        # add kicks
        if enable_kicks and ticks%600 == 0: #every 10 seconds at 60fps
            kick_strength = 0.01*(ticks/600) # 0.01,0.02,0.03 etc.
            sign = 1 if np.random.random() < 0.5 else -1
            pencil.angular_velocity += sign*kick_strength
            print(f"Kick! Angular velocity peturbed by {sign*kick_strength:.3f}")
    else: # player controlled
        # Handling held down keys
        keys = pg.key.get_pressed()
        if keys[pg.K_RIGHT]:
            base.x += x_speed
            pencil.move_tip(x_speed)
        elif keys[pg.K_LEFT]:
            base.x -= x_speed
            pencil.move_tip(-x_speed)
        else:
            pencil.move_tip(0) # tip needs to know when its not moving

    # Handling input
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.KEYDOWN:
            # handling single key presses
            if event.key == pg.K_SPACE:
                # restart_game
                del base
                del pencil
                del stopwatch
                ticks = 0
                base, pencil, stopwatch = restart_game()
            if event.key == pg.K_r:
                # randomize ai
                ai.randomize()
                if zero_biases:
                    ai.set_biases(np.zeros( (np.asarray(ai.biases).shape) ))
                del base
                del pencil
                del stopwatch
                ticks = 0
                base, pencil, stopwatch = restart_game()
            if event.key == pg.K_d:
                # disable ai
                enable_AI = not enable_AI
            if event.key == pg.K_t:
                print(stopwatch.time(), tick_speed, ticks)
                print([pencil.angle*5,
                               pencil.angular_velocity*10,
                               (2*pencil.tip_position[0]/screen_width - 1)],
                               ai.outputs)
            if event.key == pg.K_y:
                tick_speed = 2 if tick_speed==60 else 60
            if event.key == pg.K_u:
                tick_speed = 'unlimited' if tick_speed==60 else 60
            if event.key == pg.K_TAB:
                show_controls = not show_controls
            if event.key == pg.K_q:
                pencil.angular_velocity += np.random.normal(0, 1e-2, 1)[0]
            if event.key == pg.K_i:
                auto_restart = not auto_restart
            if event.key == pg.K_o:
                print('Current AI :', ai.weights)
                print('Top 5 AI so far :', best_ai)
            if event.key == pg.K_p:
                enable_kicks = not enable_kicks


    if loop_screen:
        # handle going off screen
        base.x = base.x % screen_width
        pencil.tip_position[0] = pencil.tip_position[0] % screen_width
        pencil.com_position[0] = pencil.com_position[0] % screen_width
    elif base.x < 0 or base.x > screen_width:
        pencil.fallen = True

    # render
    screen.fill(background_color)
    pencil.simulate_physics()
    pencil.draw(screen)
    pg.draw.rect(screen, (0,0,0), base)
    if show_controls:
        controls = ['SPACE : Restart game',
                    'R : Randomize AI + restart game',
                    'D : Toggle AI control',
                    'T : Print stopwatch time, tick speed to console',
                    'Y : Toggle slow motion',
                    'U : Toggle fast motion',
                    'TAB : Hide controls',
                    'Q : Nudge pencil by a random amount',
                    'I : Toggle auto-restart',
                    'O : Print best 5 AI so far',
                    'P : Toggle kicks']
        y_sep = 0
        for control in controls:
            base_font=pg.font.Font(None, 24)
            txt_surface = base_font.render(control, True, (255,255,255))
            txt_rect = txt_surface.get_rect()
            screen.blit(txt_surface, [50-txt_rect[1], 670-txt_rect[3]/2-y_sep])
            y_sep += 20
    else:
        render_text(screen, "TAB to show controls", 32, (255,255,255), [150, 670])

    # game end logic
    if pencil.fallen:
        render_text(screen, "Whoops", 64, (255,255,255), [screen_width/2, 280])
        render_text(screen, "You lasted {0:.2f} s".format(stopwatch.time()),
                    32, (255,255,255), [screen_width/2, 320])
        stopwatch.paused = True
        if len(best_ai) < 5:
            best_ai.append({'fitness':ticks,
                            'weights':ai.weights,
                            'biases':ai.biases})
        else:
            if ticks > best_ai[0]['fitness']:
                print(f'fitness = {ticks}, removing {best_ai[0]}')
                best_ai.append({'fitness':ticks,
                                'weights':ai.weights,
                                'biases':ai.biases})
                # remove first element
                best_ai.pop(0)
        if stopwatch.time() > best_time:
            best_time = stopwatch.time()
            print("New best time! {0:.2f} s".format(best_time))
            print(f"Weights: {ai.weights}")
        if auto_restart: # loop until we find steady state
            if restarts >= 15:
                NEAT(best_ai, ai)
            else:
                ai.randomize()
            if zero_biases:
                ai.set_biases(np.zeros( (np.asarray(ai.biases).shape) ))
            del base
            del pencil
            del stopwatch
            restarts += 1
            ticks = 0
            base, pencil, stopwatch = restart_game()
    else:
        render_text(screen, "{0:.2f} s".format(stopwatch.time()),
                    32, (255,255,255), [screen_width/2, 100])
    if enable_AI:
        render_text(screen, "AI Enabled", 32, (255,255,255), [100, 50])
        render_neural_network(screen, ai, [520,120], x_stretch=2,
                              input_labels=['angle',
                                            'tip speed',
                                            'tip position'],
                              output_labels=['movement'])
        render_text(screen, f"Restarts = {restarts}", 32, (255,255,255), [620, 670])
        if restarts >= 15:
            render_text(screen, "NEAT Enabled", 32, (255,255,255), [620, 620])

    pg.display.flip() # updates everything
    if tick_speed != 'unlimited':
        clock.tick(tick_speed) # limits loop to 60 FPS
