import pymunk
import pygame
from Physics.utils import *
from math import atan2
import math

## The object that needs to handled by the robot arms.
class Polygon():
    def __init__(self, space, originalAngle, points, collision_type = 40):
        self.orginal = originalAngle
        self.currentAngle = originalAngle

        self.points = points
        
        positionX = 0
        positionY = 0
        for idx in range(len(points)):
            positionX += points[idx][0]
            positionY += points[idx][1]

        positionX = positionX/len(points)
        positionY = positionY/len(points)

        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.angle = originalAngle
        self.shape = pymunk.Poly(self.body, self.points)
        self.shape.density = 1
        self.shape.friction = 1
        self.shape.collision_type=collision_type

        space.add(self.body, self.shape)

    def setPosition(self, pos):
        self.body.position = pos

    def draw(self, display, color=(255,255,255)):
        pygame.draw.polygon(display, color, list(map(convertCoordinartes, self.points)))


    ## We need more fucntions like so to extract the required inputs to the agents.
    ## Think of all the details the agent might need reaggarding the object and implement them.
    def getCurrentPosition(self):
        return self.body.position
    
    def getCurrentVelocity(self):
        return self.body.velocity_at_world_point
    
    def on_collision_arbiter_begin(self,arbiter, space, data):
    
        polygon = data["polygon"]
        min_distance = float('inf')
        armObject = None
        armObjKey= ""
        for key in data.get("arms_data"):

            last_joint=data.get("arms_data")[key][0].Objects[-1]

            contact_point = arbiter.contact_point_set.points[0].point_a
            arm=data.get("arms_data")[key][0]

            last_joint_object=last_joint["Object"]
            last_joint_length=last_joint["Length"]
            end_of_last_joint = last_joint_object.position + (0, last_joint_length / 2)

            distance = math.sqrt((end_of_last_joint.x - contact_point.x) ** 2 + (end_of_last_joint.y - contact_point.y) ** 2)

            if distance < min_distance:
                min_distance = distance
                armObject = arm
                armObjKey = key

        try:
            print("Handeling collision for", armObjKey)
            if data.get("arms_data")[armObjKey][1]:
                data.get("arms_data")[armObjKey][0].grab(contact_point,polygon)
        except KeyError:
            return True
        except Exception as e:
            print(e)
            return True
        return True