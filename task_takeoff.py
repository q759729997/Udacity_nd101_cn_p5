import numpy as np
from physics_sim import PhysicsSim


class Takeoff(object):
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.初始化指定本任务所需的几个变量
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3  # 重复调用动作次数

        self.state_size = self.action_repeat * 6  # 状态向量中每个分量的数值
        self.action_low = 0  # 动作最小值
        self.action_high = 900  # 动作最大值
        self.action_size = 4  # 四维动作空间

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """
        Uses current pose of sim to return reward. 计算奖励值。
        """
        distance_vec = self.sim.pose[:3] - self.target_pos
        if self.sim.pose[2] < self.target_pos[2]:
            reward = -min(abs(np.linalg.norm(distance_vec)), 20.0)/20.0
        else:
            reward = min(abs(np.linalg.norm(distance_vec)), 20.0)/20.0
        # reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            # 判断是否到达目标位置
            if self.sim.pose[2] >= self.target_pos[2]:
                done = True

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
