import pygame as pg
import numpy as np
import os
dirname = os.path.dirname(__file__)

pg.init()

class Pencil:
    def __init__(self, tip_position=[360, 480], g=0.3, damping=0.005):
        self.height = 360
        self.tip_position = tip_position
        self.com_position = [self.tip_position[0],
                             self.tip_position[1] - self.height/2]

        self.image = pg.image.load(os.path.join(dirname, 'pencil.png'))
        self.rect = self.image.get_rect()
        self.rect.center = self.com_position

        self.angular_velocity = 0 # clockwise
        self.tip_speed = 0
        self.tip_accel = 0
        self.angle = 0 # radians
        self.g = g
        self.vertical_speed = 0
        self.damping = damping
        self.fallen = False

    def move_tip(self, dx):
        self.tip_position[0] += dx
        self.tip_accel = dx - self.tip_speed
        self.tip_speed = dx

    def rotate_image(self, pivot, offset):
        angle_degrees = self.angle*180/np.pi

        rotated_image = pg.transform.rotate(self.image, -angle_degrees)
        rotated_offset = offset.rotate(angle_degrees)
        rotate_rect = rotated_image.get_rect(center=pivot+rotated_offset)
        return rotated_image, rotate_rect

    def draw(self, screen):
        self.rect.center = self.com_position

        if not self.fallen:
            pivot = self.tip_position
            offset = pg.math.Vector2(0, -self.height/2)
        else:
            pivot = self.com_position
            offset = pg.math.Vector2(0, 0)

        rotated_image, rotate_rect = self.rotate_image(pivot, offset)

        screen.blit(rotated_image, rotate_rect)
        return

    def simulate_physics(self):
        if np.abs(self.angle*180/np.pi % 360 - 180) < 90 or self.fallen:
            # pencil has fallen
            self.fallen = True
            self.vertical_speed += self.g
            self.tip_position[1] += self.vertical_speed
            self.com_position[1] += self.vertical_speed
            self.angle += self.angular_velocity
            return

        # damping
        horizontal_drag = self.damping*self.tip_speed*np.cos(self.angle)/(self.height/2)
        rotational_drag = self.damping*self.angular_velocity
        self.angular_velocity -= horizontal_drag + rotational_drag
        #print("horizontal_drag*1e5 = {0:.5f}".format(horizontal_drag*1e5))
        #print("rotational_drag*1e5 = {0:.5f}".format(rotational_drag*1e5))
        #print("angle*180/pi = {0:.1f}".format(self.angle*180/np.pi % 360))

        # rotation due to tip movement
        self.angular_velocity -= (self.tip_accel / (self.height/2)) * np.cos(self.angle)

        # rotation due to gravity
        self.angular_velocity += self.g*np.sin(self.angle)/(self.height/2)

        self.angle += self.angular_velocity

        # update CoM position
        self.com_position[0] = self.tip_position[0] + (self.height/2) * np.sin(self.angle)
        self.com_position[1] = self.tip_position[1] - (self.height/2) * np.cos(self.angle)
        return
