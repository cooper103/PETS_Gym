#!/usr/bin/env python3

#version two has been changed to reflect how learning is done in turtlebot docs
#environment has been changed, and rewards will also be much better

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math
import random

import rospy
import time
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import LinkStates
from std_srvs.srv import Empty
#if doesn't work make sure to inherit from gym Env class
class turtlebot3Env():
    def __init__(self):
        self.action_space = Discrete(5)
        scan_array_low = np.full((24), .11)
        scan_array_high = np.full((24), 3.5)
        low_arr = np.append(scan_array_low, [-math.pi,0])
        high_arr = np.append(scan_array_high, [math.pi,4])
        self.observation_space = Box(low=low_arr, high=high_arr) #can find this from openai_ros tb2 examples
        #print('observation_space is: ', self.observation_space)
        self.tb3_position = Pose()
        self.tb3_vel = Twist()
        self.goal_pos = Pose()
        self.scan = LaserScan()
        self.goal_pos.position.x = 1
        self.goal_pos.position.y = 1.5
        self.goal_pos.position.z = 0
        self.collision = False
        self.success = False
        self.collision_value = 0.2 #distance under which a scan value is considered a collision
        self.forward_linear_speed = 0.15 #constant forward linear speed
        self.steps = 0
        self.max_steps = 500
        self.reset_simulation_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.init_node('DQN_simulation', anonymous=True)
        rospy.Subscriber("/gazebo/link_states", LinkStates, self.get_pose)
        rospy.Subscriber("/scan", LaserScan, self.get_scan)

    def get_pose(self, data):
        ind = data.name.index('turtlebot3_burger::base_footprint')
        self.tb3_position = data.pose[ind]
    
        orientation_list = [self.tb3_position.orientation.x, self.tb3_position.orientation.y, self.tb3_position.orientation.z, self.tb3_position.orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_pos.position.y - self.tb3_position.position.y, self.goal_pos.position.x - self.tb3_position.position.x)

        diff = goal_angle - yaw
        if diff > math.pi:
            diff -= 2 * math.pi

        elif diff < -math.pi:
            diff += 2 * math.pi

        self.angle_from_goal = round(diff, 2)
        self.distance_from_goal = round(self.calc_distance_from_goal(),2)

    def get_scan(self, scan):
        #for this implemtation, we will assume that the laserscan has been modified such that
        #only 24 scan values are considered
        self.scan = scan
    
    def get_state(self, scan):
        scan_range = []
        ang = self.angle_from_goal
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(round(scan.ranges[i],2))

        if self.collision_value > min(scan_range) > 0:
            done = True

        dist = self.distance_from_goal
        if dist < 0.5:
            self.success = True
            done = True

        return scan_range + [ang, dist], done

    def get_reward(self, state, done, action):
        yaw_reward = []
        current_distance = state[-1]
        heading = state[-2]

        for i in range(5):
            angle = -math.pi / 4 + heading + (math.pi / 8 * i) + math.pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.starting_goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if self.success:
            print('success')
            reward = 100
        elif done:
            print('collided')
            reward = -100

        return reward

    def step(self, action):
        done = False
        reward = 0
        self.tb3_vel.linear.x = self.forward_linear_speed
        if action == 0:
            ang_vel = -1.5
        if action == 1:
            ang_vel = -.75
        if action == 2:
            ang_vel = 0
        if action == 3:
            ang_vel = 0.75
        if action == 4:
            ang_vel = 1.5
        self.tb3_vel.angular.z = ang_vel
        self.pub_vel.publish(self.tb3_vel)

        scans = None
        while scans is None:
            try:
                scans = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.get_state(scans)
        reward = self.get_reward(state, done, action)

        info = {"steps": self.steps}
        if self.steps == self.max_steps -1:
            done = True
        self.steps += 1
        return state, reward, done, info
    
    def calc_distance_from_goal(self):
        a = np.array((self.tb3_position.position.x, self.tb3_position.position.y, self.tb3_position.position.z))
        b = np.array((self.goal_pos.position.x, self.goal_pos.position.y, self.goal_pos.position.z))
        return np.linalg.norm(a-b)

    def action_space_sample(self):
        return random.randint(0,4)

    def render(self):
        pass

    def evaluate_actions(self, states ,actions):
        #to do: determine how TSinf will know if it has predicted a temrinal
        #state and also account for this in the reward evaluation of action 
        #sequences

        #actions is a 20 x 30 numpy array of action sequences
        #states is a 20 x 30 array
        #add terminnal state prediction in this function 
        #if any state in the sequence is the terminal state, then just end the
        #reward summation there after subtracting a penalty
        reward = np.empty((20))
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int)

        for action_sequence in range(20):
            for t in range(20):
                reward[action_sequence] += self.get_reward(states[action_sequence][t], False, actions[action_sequence][t])
        reward /= np.size(actions, 1) #average reward of each sequence
        return reward

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        self.reset_simulation_client()
        self.success = False
        self.collision = False
        self.steps = 0
        scan = None
        while scan is None:
            try:
                scan = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
            
        self.starting_goal_distance = self.calc_distance_from_goal()
        state, _ = self.get_state(scan)
        return state,{}
"""
if __name__ == '__main__':
    env = turtlebot3Env()
    
    for i in range(10):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = random.randint(0,4)
            state, reward, done, info = env.step(action)
            score += reward
            #print('angle from goal: ', state[-2], 'scan samples:', state[0])
        print('Episode: {0} Reward: {1} Steps: {2}'.format(i, score, info))

"""