import torch
import gym
import numpy as np
from gym.envs.classic_control import rendering
from gym import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print "Using", device
UsingGPU = torch.cuda.is_available()

"""
CTO variant with only 1 observer
"""

class CtoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    Summary of the the environment variables
            
        runTime
            The total time simulation runs for

        updateRate
            Time interval between each decision making/action

        episodes
            Equal to runTime / updaRate
            Simulation terminates after these many episodes

        gridWidth, gridHeight
            Dimensions of the 2D arena

        sensorRange
            The maximum distance between agent and target for the agent to notice
            the target
    
    """

    def __init__(self):
        self.viewer = None


    def initialize(self, targets=10, sensorRange=15, updateRate=10, targetMaxStep=100,
                    targetSpeed=1.0,
                    totalSimTime=1500, gridWidth=150, gridHeight=150, compact=False):
        #System variables
        self.curr_episode = torch.tensor(0).to(device)
        self.curr_step = torch.tensor(0).to(device)
        
        # general variables in the environment
        self.runTime = torch.tensor(totalSimTime).to(device)        

        #total number of targets in the simulation
        self.numTargets = torch.tensor(targets).to(device)

        #maximum time for which one target can stay oncourse for its destination
        self.targetMaxStep = torch.tensor(targetMaxStep).to(device)

        #speed of target
        self.targetSpeed = torch.tensor(targetSpeed).to(device)
        self.agentSpeed = torch.tensor(1.0).to(device)

        #sensor range of the observer
        self.sensorRange = torch.tensor(sensorRange).type(torch.FloatTensor).to(device)

        #time after which observer takes the decision
        self.updateRate = torch.tensor(updateRate).to(device)

        #2D field dimensions
        self.gridHeight = torch.tensor(gridHeight).type(torch.FloatTensor).to(device)
        self.gridWidth = torch.tensor(gridWidth).type(torch.FloatTensor).to(device)
        self.gridDimensions = torch.tensor([self.gridWidth, self.gridHeight]).to(device)

        self.compactRepresentation = torch.tensor(compact).to(device)

        #Initialize target locations and their destinations
        self.targetLocations = torch.zeros(self.numTargets, 2).to(device)
        self.targetDestinations = torch.zeros(self.numTargets, 2).to(device)
        self.targetSteps = torch.empty(self.numTargets).fill_(self.targetMaxStep).to(device)
        self.targetPosIncrements = torch.empty(self.numTargets, 2).fill_(-1000.0).to(device).to(device)

        if not UsingGPU:
            self.targetDestinations = self.gridDimensions * torch.rand(self.numTargets, 2).to(device)
            self.targetLocations = self.gridDimensions * torch.rand(self.numTargets, 2).to(device)
        else:
            self.targetDestinations = self.gridDimensions * torch.cuda.FloatTensor(self.numTargets, 2).uniform_(0, 1)
            self.targetLocations = self.gridDimensions * torch.cuda.FloatTensor(self.numTargets, 2).uniform_(0, 1)

        #Initialize the agent and ensure it is not on top of other target
        self.agentPosition = torch.zeros(2).to(device)

        if not UsingGPU:
            self.agentPosition = self.gridDimensions * torch.rand(2).to(device)
        else:
            self.agentPosition = self.gridDimensions * torch.cuda.FloatTensor(2).uniform_(0, 1)

        self.agentPosIncrements = torch.empty(2).fill_(-1000.0).to(device)

        self.episodes = self.runTime / self.updateRate

        #Defaulted variables
        self.ze1 = torch.zeros(1).to(device)
        self.ze2 = torch.zeros(2).to(device)
        self.agentReachedDest = torch.tensor(False).to(device)
        self.reward = torch.zeros(1).to(device)

    # Checks whether the two points are at least one unit apart
    # def acceptable(self, index, agent=False):
    #     if not agent:
    #         if index == 0:
    #             return True            
    #         else:
    #             for i in xrange(index):
    #                 if self.distance(self.targetLocations[index], self.targetLocations[i]) <= 1:
    #                     return False
    #             return True        
    #     else:
    #         for i, pos in enumerate(self.targetLocations):
    #             if self.distance(self.agentPosition, pos) <= 1:
    #                     return False
    #         return True


    # Calculates euclidean distance between two points
    def distance(self, pos1, pos2):
        euclideanDistance = torch.sum( torch.pow(pos1 - pos2, 2) )
        return torch.sqrt(euclideanDistance)


    def reset(self):
        if self.compactRepresentation:
            self.state = None
            for i, t in enumerate(self.targetLocations):
                if self.distance(self.agentPosition, t) <= self.sensorRange:
                    if self.state is None:
                        self.state = t
                    elif self.state.shape == t.shape:
                        self.state = torch.cat([self.state.unsqueeze(0), t.unsqueeze(0)], dim=0)
                    else:
                        self.state = torch.cat([self.state, t.unsqueeze(0)], dim=0)

        else:
            self.state = torch.zeros(self.numTargets, 2).to(device)

            for i, t in enumerate(self.targetLocations):
                if self.distance(self.agentPosition, t) <= self.sensorRange:
                    self.state[i] = t

        # if self.state is None:
        #     self.state = self.agentPosition.unsqueeze(0)
        # elif self.state.shape == self.agentPosition.shape:
        #     self.state = torch.cat([self.state.unsqueeze(0), self.agentPosition.unsqueeze(0)], dim=0)
        # else:
        #     self.state = torch.cat([self.state, self.agentPosition.unsqueeze(0)], dim=0)
        return self.state


    def step(self, action):
        if self.curr_episode > self.episodes:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'initialize()' and 'reset()' once you receive 'done = True'")
            return

        self.curr_episode += 1

        self.reward.fill_(0.0)
        self.agentReachedDest.fill_(False)
        self.agentPosIncrements.fill_(-1000.0)

        for _ in xrange(self.updateRate):
            self.curr_step += 1

            #Move targets
            for i in xrange(self.numTargets):
                self.moveTarget(i)

            #Move agent
            if not self.agentReachedDest:
                self.agentReachedDest.fill_(self.moveAgent(action))
            else:
                self.agentPosition = action

            #Calculate reward at this step
            for i, t in enumerate(self.targetLocations):
                if self.distance(self.agentPosition, t) <= self.sensorRange:
                    self.reward += 1

            if self.viewer is not None:
                self.render()
        
        return self.reset(), self.reward, self.curr_episode >= self.episodes, {}
            

    def moveTarget(self, idx):
        # Check if this target has been oncourse for max allowed time or it reached its destination
        if self.targetSteps[idx] == 0 or (abs(self.targetDestinations[idx][0] - self.targetLocations[idx][0]) < 1 and 
            abs(self.targetDestinations[idx][1] - self.targetLocations[idx][1]) < 1):
            self.targetDestinations[idx] = self.gridDimensions * torch.rand(2).to(device)     
            #Create new destination and reset step counter to max allowed time and position increments to default   
            self.targetSteps[idx] = self.targetMaxStep
            self.targetPosIncrements[idx].fill_(-1000.0)

        if self.targetPosIncrements[idx][0] == -1000.0 or self.targetPosIncrements[idx][1] == -1000.0:
            self.targetPosIncrements[idx] = self.calculateIncrements(self.targetLocations[idx], 
                                                                        self.targetDestinations[idx], self.targetSpeed)       

        self.targetLocations[idx] += self.targetPosIncrements[idx]

        self.targetLocations[idx] = torch.max( torch.min(self.targetLocations[idx], self.gridDimensions), self.ze2 )

        self.targetSteps[idx] -= 1


    def moveAgent(self, dest):
        if self.agentPosIncrements[0] == -1000.0 or self.agentPosIncrements[1] == -1000.0:
            self.agentPosIncrements = self.calculateIncrements(self.agentPosition, dest, self.agentSpeed)
        
        self.agentPosition += self.agentPosIncrements

        self.agentPosition = torch.max( torch.min(self.agentPosition, self.gridDimensions), self.ze2 )

        if abs(dest[0] - self.agentPosition[0]) < 1 and abs(dest[1] - self.agentPosition[1]) < 1:
            return True
        else:
            return False

    
    def calculateIncrements(self, loc, dest, speed):
        delta = dest - loc

        theta = torch.zeros(1).to(device)
        if abs(delta[0]) > abs(delta[1]):
            theta = torch.abs(delta[0])
        else:
            theta = torch.abs(delta[1])
        
        if theta == 0.0:
            return self.ze2

        inc = delta / theta
        normalizer = torch.sqrt( torch.sum( torch.pow(inc, 2) ) )

        inc = speed * (inc / normalizer)

        return inc

    def getAgentPosition(self):
        return self.agentPosition


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        borderOffset = 50.0 #Reduces 50px along 4 sides

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.scale = ( (screen_width - 2*borderOffset)*1.0/self.gridWidth, 
                    (screen_height - 2*borderOffset)*1.0/self.gridHeight)
            #Borders for neat view
            border1 = rendering.Line((borderOffset, borderOffset), 
                                    (screen_width - borderOffset, borderOffset))
            self.viewer.add_geom(border1)
            border2 = rendering.Line((borderOffset, borderOffset), 
                                    (borderOffset, screen_height - borderOffset))
            self.viewer.add_geom(border2)
            border3 = rendering.Line((screen_width - borderOffset, screen_height - borderOffset), 
                                    (screen_width - borderOffset, borderOffset))
            self.viewer.add_geom(border3)
            border4 = rendering.Line((screen_width - borderOffset, screen_height - borderOffset), 
                                    (borderOffset, screen_height - borderOffset))
            self.viewer.add_geom(border4)

            #Memorize
            self.targets_geom = []
            for i in self.targetLocations:
                point = (self.scale[0]*i[0] + borderOffset, self.scale[1]*i[1] + borderOffset)

                location = rendering.Transform(translation=point)
                axle = rendering.make_circle(4.0)
                axle.add_attr(location)
                axle.set_color(1.0, 0.0, 0.0)
                self.targets_geom.append(location)
                self.viewer.add_geom(axle)

            self.agent_geom = (self.scale[0]*self.agentPosition[0] + borderOffset, 
                                self.scale[1]*self.agentPosition[1] + borderOffset)
            location = rendering.Transform(translation=self.agent_geom)
            self.agent_geom = rendering.make_circle(4.0)
            self.agent_geom.add_attr(location)
            self.agent_geom.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(self.agent_geom)
            self.agent_geom = location

            coverage = self.viewer.draw_circle(radius=self.scale[0]*self.sensorRange, res=30, filled=False)
            coverage.add_attr(location)
            coverage.set_color(0.5, 0.5, 0.8)

            self.viewer.add_geom(coverage)

        else:
            for i, t in enumerate(self.targets_geom):
                point = (self.scale[0]*self.targetLocations[i][0] + borderOffset, self.scale[1]*self.targetLocations[i][1] + borderOffset)

                t.set_translation(point[0], point[1])

            point = (self.scale[0]*self.agentPosition[0] + borderOffset, self.scale[1]*self.agentPosition[1] + borderOffset)
            self.agent_geom.set_translation(point[0], point[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def stopRender(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None