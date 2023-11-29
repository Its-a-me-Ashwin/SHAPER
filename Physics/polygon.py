import pymunk
import pygame
from Physics.utils import *
from math import atan2

class Polygon():
    def __init__(self, space, originalAngle, goalAngle, points, collision_type = 40):
        self.orginal = originalAngle
        self.goal = goalAngle
        self.currentAngle = originalAngle

        self.points = points
        
        positionX = 0
        positionY = 0
        for idx in range(len(points)):
            positionX += points[idx][0]
            positionY += points[idx][1]

        positionX = positionX/len(points)
        positionY = positionY/len(points)

        self.body = pymunk.Body()
        self.body.position = positionX, positionY
        self.body.angle = atan2(originalAngle[1],originalAngle[0])
        self.shape = pymunk.Poly(self.body, self.points)
        self.shape.density = 1
        self.shape.friction = 1
        self.shape.collision_type = collision_type
        space.add(self.body, self.shape)


    def draw(self, display, color=(255,255,255)):
        pygame.draw.polygon(display, color, list(map(convertCoordinartes, self.points)))


    ## We need more fucntions like so to extract the required inputs to the agents.
    ## Think of all the details the agent might need reaggarding the object and implement them.
    def getCurrentPosition(self):
        return self.body.position
    
    def getCurrentVelocity(self):
        return self.body.velocity_at_world_point