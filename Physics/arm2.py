import pymunk
from pymunk import SimpleMotor
#from Physics.utils import *
from math import atan2, sin, cos, sqrt
#from Physics.gripper import *
#from Physics.armsection import *
#from Physics.ball import *


MASS_PER_LENGTH = 10
ARM_WIDTH = 10
PI = 355/113 ## Fancy approximation for Pi


## PID settings
P_ = 1.9
D_ = -0.5
I_ = 0.0005

class Arm1():
    def __init__(self, space, anchorPoistion, group=1):
        self.space = space
        if len(anchorPoistion) != 2:
            print("Error size incorrect")
            return
        self.anchor = anchorPoistion
        self.complete = False

        self.CurrentAngles = []
        self.ExpectedAngles = []

        self.Objects = []
        self.collType = group
        self.shapeFilter = pymunk.ShapeFilter(group=1)


        ## Number of frames since the expected angles were set.
        ## This variable will be rest every time the SetAngles function is called.
        ## It will be incremented every time the arbiter is executed successfully. 
        self.diffCounter = []

        # One purpose of the arm is to track the polygon so that it can catch it.
        # This parameters govern it
        self.pinJoint = None

    def addJoint(self, length, rotation = 0, collision_type = 20, end=False):
        if self.complete:
            # if this is the end of the arm, set a collision type to it so that the gripper can use it.
            self.Objects[-1]["Shape"].collision_type=collision_type
            return
        if len(self.Objects) == 0: 
            ## Need to add an anchor.

            ## Generate the anchor.
            newAnchor = pymunk.Body(body_type=pymunk.Body.STATIC)
            newAnchor.position = self.anchor

            ## Generate the mass of the body.
            armMass = MASS_PER_LENGTH*length
            ## Create the body.
            newArmObject = pymunk.Body(armMass, pymunk.moment_for_box(armMass, (ARM_WIDTH, length)))
            newArmObject.position = self.anchor[0], self.anchor[1]+length/2

            ## Create the shape for the object.
            newArmShape = pymunk.Poly.create_box(newArmObject, (ARM_WIDTH, length))
            newArmShape.color = 0, 0, 0, 100

            ## Add constrains and motors to the bodies.
            newJoint = pymunk.PinJoint(newArmObject, newAnchor, (0, -length/2), (0, 0))
            newMotor = pymunk.SimpleMotor(newArmObject, newAnchor, rotation)

            ## Disable colisions between the arms. 
            ## Might remove this based on hhow the model performs.
            newArmShape.filter = self.shapeFilter

            ## Also add some meta data here. Makes it easier for rendering.
            self.Objects.append({
                                "Object":newArmObject,
                                "Motor": newMotor,
                                "Middle" : (self.anchor[0], self.anchor[1]+length/2),
                                "Length": length,
                                "Shape": newArmShape
                                })

            ## Add all the objects and shapes to the space.
            self.space.add(newArmObject)
            self.space.add(newArmShape)
            self.space.add(newJoint)
            self.space.add(newMotor)

        else:
            ## Need to add a new arm.

            prevBody = self.Objects[-1].get("Object")
            prevPosition = prevBody.position
            prevLength = self.Objects[-1].get("Length")

            ## Generate the mass of the body.
            armMass = MASS_PER_LENGTH*length
            ## Create the body.
            newArmObject = pymunk.Body(armMass, pymunk.moment_for_box(armMass, (ARM_WIDTH, length)))
            newArmObject.position = prevPosition[0], prevPosition[1]+prevLength/2+length/2

            ## Create the shape for the object.
            newArmShape = pymunk.Poly.create_box(newArmObject, (ARM_WIDTH, length))
            newArmShape.color = 0, 0, 0, 100

            ## Add constrains and motors to the bodies.
            newJoint = pymunk.PinJoint(newArmObject, prevBody, (0, -length/2), (0, prevLength/2))
            newMotor = pymunk.SimpleMotor(newArmObject, prevBody, rotation)

            ## Disable colision between the arms.
            newArmShape.filter = self.shapeFilter

            ## Also add some meta data here. Makes it easier for rendering.
            self.Objects.append({
                                "Object":newArmObject,
                                "Motor": newMotor,
                                "Middle" : (prevPosition[0], prevPosition[1]+prevLength/2+length/2),
                                "Length": length,
                                "Shape": newArmShape
                            })
            self.space.add(newArmObject)
            self.space.add(newArmShape)
            self.space.add(newJoint)
            self.space.add(newMotor)
        if end:
            self.complete = True
            self.CurrentAngles = [0]*len(self.Objects)
            self.diffCounter = [0]*len(self.Objects)
            #self.space.add_collision_handler()


    ## Might need this function for compatibility with other objects.
    def draw(self, display, color = [0, 0, 0]):
        return
    
    def preHit(self, arbiter, space, data):
        print("Pre hit called for arm")
        pass


    ## Inputs are between -1 and 1, convert it to 0 to 2PI
    ## Converts the agent's output to a format that the physics engine can work with.
    def agentToPhysics(self, agentData):
        inputs = list(map(lambda x: ((x+1)/2.0)*2*PI, inputs))
        return inputs

    ## Converts the data from the physcis engine to a format that can be processed by the agent
    ## Normalize the angles and the position of the bodies. Convert to radians if necessary.
    def physicsToAgent(self):
        agentInputs = dict()
        agentInputs["Angles"] = []
        agentInputs["Rates"] = []
        agentInputs["Positions"] = []
        for objIdx in range(len(self.Objects)):
            obj = self.Objects[objIdx]["Object"]
            mot = self.Objects[objIdx]["Motor"]
            l = self.Objects[objIdx]["Length"]/2
            ## The current angle of the arms wrt to the global XY plane
            agentInputs["Angles"].append(obj.angle)
            ## The rate at which the arms are moving
            agentInputs["Rates"].append(mot.rate)
            ## Gets the position of the endpoints of the arm.
            agentInputs["Positions"].extend(centerToEndPoints(obj.position,l,obj.angle))
        return agentInputs

    ## Set the expected angles.
    def setAngles(self, inputs):
        if not self.complete:
            return
        if len(inputs) != len(self.Objects):
            return
        ## Clip it between 0 and 2PI
        inputs = list(map(lambda x: x%(2*PI), inputs))
        self.ExpectedAngles = inputs
        self.diffCounter = [0]*len(self.Objects)

    ## Get the current angles
    def getAngles(self, update=True):
        for objectIdx in range(len(self.Objects)):
            self.CurrentAngles[objectIdx] = self.Objects[objectIdx]["Object"].angle
        if update:
            self.arbiterAgent()
        #print("Current angle:", self.CurrentAngles, "Expected angle:", self.ExpectedAngles)
        return self.CurrentAngles
    
    # Once we get the data from the agent we need the physics engine to execute it.
    # This function uses a simple PID system to achieve that. 
    # DIFF = (EXPECTED - CURRENT)
    def arbiterAgent(self):
        if len(self.ExpectedAngles) == 0:
            return
        diff = list(map(lambda x: x[1]-x[0], zip(self.CurrentAngles, self.ExpectedAngles)))
        for objIdx in range(len(self.Objects)):
            if diff[objIdx] < 10**-6:
                continue
            self.diffCounter[objIdx] += diff[objIdx]
            self.Objects[objIdx]["Motor"].rate = P_*diff[objIdx] + D_*self.Objects[objIdx]["Motor"].rate + I_*self.diffCounter[objIdx]

    # Collision Handler
    def on_collision_arbiter_begin(self, arbiter, space, data):
        '''
        The crux of forming the joint.
        This function does 2 things - 
            1. Check how close the polygon is. and 
            2. if the polygon is very close, form a pymunk pinJoint.
        '''
        min_distance = 10
        closest_arm = self.Objects[-1].get("Object", None)

        if arbiter.shapes[0].collision_type == 40:
            polygon= arbiter.shapes[0]
            # arm=arbiter.shapes[1]
        elif arbiter.shapes[1].collision_type == 40:
            polygon=arbiter.shapes[1]
            # arm=arbiter.shapes[0]

        contact_point = arbiter.contact_point_set.points[0].point_a
        current_arm = closest_arm
        current_arm_length = self.Objects[-1]["Length"]
        end_of_current_arm = current_arm.position + (0, current_arm_length / 2)

        distance = sqrt(
            (end_of_current_arm.x - contact_point.x) ** 2 + (end_of_current_arm.y - contact_point.y) ** 2
        )

        if distance < min_distance and current_arm is not None and not self.pinJoint:
            self.pinJoint = pymunk.PinJoint(current_arm, polygon.body, (0, current_arm_length / 2),
                                    polygon.body.world_to_local(contact_point))
            space.add(self.pin_joint)
        
        return True

    def gripPolygon(self, polygon):
        '''
        When called, this method checks the end of the arm is making contact with the polygon provided
        as argument and builds a pinJoint b/w the end and the polygon.
        '''
        arm_body = 20
        polygon_body = 40
        arms_data = {"Arm_1": self.Objects[-1]}
        collision = self.space.add_collision_handler(polygon_body, arm_body)
        collision.begin = Arm1.on_collision_arbiter_begin
        collision.data["arms_data"] = arms_data


    def dropPolygon(self):
        '''
        When called, this method removes the pinJoin of the arm if it has any.
        '''

        if self.pinJoint is not None:
            space.remove(self.pinJoint)
            self.pinJoint = None

def centerToEndPoints(centerPos, length, angle):
    return [centerPos[0]+length*cos(angle), centerPos[1]+length*sin(angle),
            centerPos[0]-length*cos(angle), centerPos[1]-length*sin(angle)
            ]


if __name__ == "__main__":
    space = pymunk.Space()
    arm = Arm1(space, (250, 250))
    arm.addJoint(100)
    arm.addJoint(100, True)

    curAngles = arm.getAngles()
    print("Current Angles:", curAngles)

    arm.setAngles([3.14, 0])

    while True:
        curAngles = arm.getAngles()
        print("Current Angles:", curAngles)