import numpy as np
import traci
from sumolib import checkBinary
from datetime import datetime
import time
import os
from SimpleSAC.ego_policy.fvdm import fvdm_model
from SimpleSAC.utils.car_dis_comput import dist_between_cars
import traci
import ipdb
import torch

class Env:
    def __init__(self, realdata_path, num_agents = 1, dt = 0.04, sim_horizon = 200,
                 r_ego = 'r1', r_adv = 'r1', ego_policy = 'sumo', adv_policy = 'sumo', sim_seed = 42, gui = False):
        self.dt = dt
        self.state_space = [(num_agents + 1) * 4] # agents + ego
        self.action_space_ego = [2]
        self.action_space_adv = [num_agents * 2]
        self.num_agents = num_agents
        self.sim_horizon = sim_horizon
        self.max_speed = 40
        self.r_ego = r_ego
        self.r_adv = r_adv

        self.car_info = []

        
        for i in range(num_agents + 1):
            self.car_info.append([i, "lane", [5, 1.8]])
        
        self.road_info = {'lane': [0, 12]}

        realdata_filelist = []
        for f in os.listdir(realdata_path):
            for ff in os.listdir(os.path.join(realdata_path, f)):
                realdata_filelist.append(os.path.join(realdata_path, f, ff))
        self.realdata_filelist = realdata_filelist

        self.ego_policy = ego_policy
        self.adv_policy = adv_policy
        
        self.up_cut = [0.6 * 9.8 * self.dt, np.pi / 3 * self.dt]
        self.low_cut = [-0.8 * 9.8 * self.dt, -np.pi / 3 * self.dt]

        self.gui = gui
        if self.gui:
            self.app = 'sumo-gui'
        else:
            self.app = 'sumo'
        self.sim_seed = sim_seed
        self.cfg_sumo = 'config/lane_sim.sumocfg'

        self.command = [checkBinary(self.app), '-c', self.cfg_sumo]
        self.command += ['--routing-algorithm', 'dijkstra']
        self.command += ['--seed', str(self.sim_seed)]
        self.command += ['--no-step-log', 'True']
        self.command += ['--time-to-teleport', '300']
        self.command += ['--no-warnings', 'True']
        self.command += ['--duration-log.disable', 'True']
        self.command += ['--waiting-time-memory', '1000']
        self.command += ['--eager-insert', 'True']
        self.command += ['--lanechange.duration', '1.5']
        self.command += ['--lateral-resolution', '0.0']

    def reset(self, ego_policy, adv_policy, idx = None):
        self.ArrivedVehs = []
        self.CollidingVehs = []
        self.timestep = 0
        self.ego_col_cost_record = []

        for i in range(self.num_agents):
            self.ego_col_cost_record.append(0)

        self.adv_col_cost_record = float('inf')
        self.adv_dev_cost_record = 0
        self.states = [] # recording veh state, 0 means ego_agent, 1:-1 means adv_agent
        self.actions = [] # recording veh action, 0 means ego_agent, 1:-1 means adv_agent

        for t in range(self.sim_horizon + 1):
            state_t = []
            action_t = []
            for id in range(self.num_agents + 1):
                state_t.extend([0, 0, 0, 0])
                action_t.extend([0, 0])
            self.states.append(state_t)
            self.actions.append(action_t)
        if idx is not None:
            filepath = self.realdata_filelist[idx]
        else:
            filepath = self.realdata_filelist[np.random.randint(len(self.realdata_filelist))]
        # ipdb.set_trace()
        dataset = np.load(filepath, allow_pickle = True).item()
        self.states[0] = dataset['observations'][0]

        self.ego_policy = ego_policy
        self.adv_policy = adv_policy

        for i in range(self.num_agents + 1):
            self.states[0][i * 4 + 2] = min(self.states[0][i * 4 + 2], self.max_speed) # limit the speed
        
        
        # ipdb.set_trace()
        for vehicle in traci.vehicle.getIDList():
            traci.vehicle.remove(vehicle)

        cur_time = float(traci.simulation.getTime())

        '''ego vehicle'''
        # For the case of uniform speed, set the default initial angle to 0
        # ipdb.set_trace()
        if self.ego_policy == 'uniform':
            self.states[0][3] = 0
            traci.vehicle.add(vehID="car0", 
                              routeID="straight", 
                              typeID="AV",
                              depart=cur_time, 
                              departLane=0, 
                              departPos=10.0,
                              departSpeed=self.states[0][2])
        
            
        else:
            traci.vehicle.add(vehID="car0", 
                              routeID="straight", 
                              typeID="AV",
                              depart=cur_time, 
                              departLane=0, 
                              departPos=10.0,
                              departSpeed=self.states[0][2])
        
        traci.vehicle.moveToXY(vehID = 'car0',
                               x = self.states[0][0],
                               y = self.states[0][1],
                               angle = -self.states[0][3] * 180 / np.pi + 90,
                               lane = 0,
                               edgeID = 0)
        

        if self.ego_policy == 'fvdm':
            self.ego_policy_model = fvdm_model('car0')

        '''adversary vehicles'''
        if self.adv_policy == 'uniform':
            for i in range(self.num_agents):
                self.states[0][4 * (i + 1) + 3] = 0
            for i in range(self.num_agents):
                traci.vehicle.add(vehID = "car" + str(i + 1), 
                                  routeID = "straight", 
                                  typeID = "BV",
                                  depart = cur_time, 
                                  departLane = 1,
                                  departPos = 40.0, 
                                  arrivalPos = np.inf,
                                  departSpeed = self.states[0][4 * (i + 1) + 2])
        elif self.adv_policy == 'sumo':
            for i in range(self.num_agents):
                traci.vehicle.add(vehID = "car" + str(i + 1), 
                                  routeID = "straight", 
                                  typeID = "BV",
                                  depart = cur_time, 
                                  departLane = 1, 
                                  departPos = 40.0,
                                  arrivalLane = np.random.randint(0, 3), arrivalPos=np.inf,
                                  departSpeed = self.states[0][4 * (i + 1) + 2])
        else:
            for i in range(self.num_agents):
                traci.vehicle.add(vehID = "car" + str(i + 1), 
                                  routeID = "straight", 
                                  typeID = "BV",
                                  depart = cur_time, 
                                  departLane = 1,
                                  departPos = 40.0, 
                                  arrivalPos = np.inf,
                                  departSpeed = self.states[0][4 * (i + 1) + 2])
        
        for i in range(self.num_agents):
            traci.vehicle.moveToXY(vehID = "car" + str(i + 1),
                                x = self.states[0][4 * (i + 1)],
                                y = self.states[0][4 * (i + 1) + 1],
                                angle = -self.states[0][4 * (i + 1) + 3] * 180 / np.pi + 90,
                                lane = 0,
                                edgeID = 0)
            
        
        if self.adv_policy == 'fvdm':
            self.adv_policy_model = []
            for i in range(self.num_agents):
                self.adv_policy_model.append(fvdm_model("car" + str(i + 1)))
        
        traci.simulationStep()

        return self.states[0]
    
    def traci_start(self):
        traci.start(self.command)

    def traci_close(self):
        traci.close()

    def get_state(self, car_id):
        return self.states[self.timestep][car_id * 4: (car_id + 1) * 4]
    
    def set_state(self, state, car_id):
        self.states[self.timestep + 1][car_id * 4: (car_id + 1) * 4] = state

    def step(self, action_ego, action_adv):
        '''ego vehicle'''
        if self.ego_policy == "uniform":
            av_action = [0, 0]
            new_state = self.motion_model(self.get_state(0), av_action)
            traci.vehicle.moveToXY(vehID = "car0",
                                   x = new_state[0],
                                   y = new_state[1],
                                   angle = -new_state[3] * 180 / np.pi + 90,
                                   lane = 0, 
                                   edgeID = 0)
            traci.vehicle.setSpeed(vehID = 'car0', speed = new_state[2])
        elif self.ego_policy == "fvdm":
            self.ego_policy_model.run()
        elif self.ego_policy == 'RL':
            new_state = self.motion_model(self.get_state(0), action_ego)
            traci.vehicle.moveToXY(vehID = "car0",
                                   x = new_state[0],
                                   y = new_state[1],
                                   angle = -new_state[3] * 180 / np.pi + 90,
                                   lane = 0, 
                                   edgeID = 0)
        elif self.ego_policy == "sumo":
            ...
        

        '''adversary vehicles'''
        if self.adv_policy == "uniform":
            for i in range(self.num_agents):
                new_state = self.motion_model(self.get_state(i + 1), [0, 0])
                traci.vehicle.moveToXY(vehID = "car" + str(i + 1),
                                       x = new_state[0],
                                       y = new_state[1],
                                       angle = -new_state[3] * 180 / np.pi + 90,
                                       lane = 0,
                                       edgeID = 0)
                traci.vehicle.setSpeed(vehID = "car" + str(i + 1), speed = new_state[2])

        elif self.adv_policy == "fvdm":
            for i in range(self.num_agents):
                self.adv_policy_model[i].run()
        elif self.adv_policy == 'RL':
            for i in range(self.num_agents):
                new_state = self.motion_model(self.get_state(i + 1), action_adv[2 * i: 2 * (i + 1)])
                traci.vehicle.moveToXY(vehID = "car" + str(i + 1),
                                       x = new_state[0],
                                       y = new_state[1],
                                       angle = -new_state[3] * 180 / np.pi + 90,
                                       lane = 0,
                                       edgeID = 0)
                traci.vehicle.setSpeed(vehID = "car" + str(i + 1), speed = new_state[2])
        elif self.ego_policy == "sumo":
            ...

        traci.simulationStep()
        # time.sleep(0.04)
        # ipdb.set_trace()
        next_state = []
        for i in range(self.num_agents + 1):
            (x, y) = traci.vehicle.getPosition('car' + str(i))
            speed = traci.vehicle.getSpeed('car' + str(i))
            yaw = (90 - traci.vehicle.getAngle('car' + str(i))) * np.pi / 180
            new_state = [x, y, speed, yaw]
            next_state.extend(new_state)
            self.set_state(new_state, i)
            delta_speed = self.states[self.timestep + 1][i * 4 + 2] - self.states[self.timestep][i * 4 + 2]
            delta_yaw = self.states[self.timestep + 1][i * 4 + 3] - self.states[self.timestep][i * 4 + 3]
            action_speed = (delta_speed - self.low_cut[0]) * 2 / (self.up_cut[0] - self.low_cut[0]) - 1
            action_yaw = (delta_yaw - self.low_cut[1]) * 2 / (self.up_cut[1] - self.low_cut[1]) - 1
            
            car_action = [action_speed, action_yaw]
            
            if new_state[0] >= 240:  # 240 is the path length
                self.ArrivedVehs.append(0)
            self.actions[self.timestep][i * 2: (i + 1) * 2] = car_action
        # ipdb.set_trace()
        
        if self.gui:
            time.sleep(0.01)

        self.timestep += 1
        self.collision_test()
        

        done = True
        info = [[], []]
        dis_av_crash = traci.vehicle.getDistance('car0')
        distance_BV = []
        for i in range(self.num_agents):
            distance_BV.append(traci.vehicle.getDistance('car' + str(i + 1)))
        dis_bv_crash = min(distance_BV)

        if 0 in self.ArrivedVehs:
            info[0] = "AV arrived!"
        elif 0 in self.CollidingVehs:
            info[0] = "AV crashed!"
            dis_bv_crash = 240
        elif len(self.CollidingVehs) != 0:
            info[0] = "BV crashed!"
            dis_av_crash = 240
        else:
            done = False

        reward = self.compute_cost(done)
        # info[1] = [ego_col_cost, adv_col_cost, adv_road_cost, dis_av_crash, dis_bv_crash, col_cost, speed_cost]
        info[1] = []
        return next_state, reward, done, info

    def motion_model(self, state, action):
        # load position
        curr_pos = [state[0], state[1]]
        curr_speed = state[2]
        curr_yaw = state[3]

        
        self.up_cut = [0.6 * 9.8 * self.dt, np.pi / 3 * self.dt]
        self.low_cut = [-0.8 * 9.8 * self.dt, -np.pi / 3 * self.dt]
        delta_speed = (action[0] + 1) * (self.up_cut[0] - self.low_cut[0]) / 2 + self.low_cut[0]
        delta_yaw = (action[1] + 1) * (self.up_cut[1] - self.low_cut[1]) / 2 + self.low_cut[1]

        
        speed = curr_speed + delta_speed
        yaw = curr_yaw + delta_yaw

        
        # if speed > self.max_speed or speed < 0:
        #     print(speed)
        if speed < 0:
            speed = speed - speed
        if speed > self.max_speed:
            speed = self.max_speed - speed + speed
        if yaw > np.pi / 3:
            yaw = np.pi / 3 + yaw - yaw
        if yaw < -np.pi / 3:
            yaw = -np.pi / 3 + yaw - yaw
        # ipdb.set_trace()
        if type(speed) == torch.Tensor:
            speed = speed.detach().cpu()
        if type(yaw) == torch.Tensor:
            yaw = yaw.detach().cpu()
        state = [state[0] + speed * np.cos(yaw) * self.dt,
                 state[1] + speed * np.sin(yaw) * self.dt,
                 speed, yaw]
        return state
    
    def compute_cost(self, done : bool):
        ego_state = self.get_state(0)
        adv_state = [self.get_state(i) for i in range(1, self.num_agents + 1)]

        '''ego vehicle cost'''
        if self.r_ego == 'r1': 
            col_cost_ego = -20 if 0 in self.CollidingVehs else 0
            speed_cost_ego = ego_state[2] / self.max_speed - 1 / 2
            yaw_cost_ego = - abs(ego_state[3]) / (np.pi / 3) * 5 * 0
            cost_ego = col_cost_ego + speed_cost_ego + yaw_cost_ego
        elif self.r_ego == 'stackelberg':
            cost_ego = 0

            if done:
                
                # col_cost_ego = -10 if 0 in self.CollidingVehs else 10
                if 0 in self.CollidingVehs:
                    cost_ego += -10
                else:
                    pass
            else:

                speed_cost_ego = ego_state[2] / self.max_speed
                # yaw_cost_ego = 1 - abs(ego_state[3]) / (np.pi / 3) * (ego_state[2] / self.max_speed)
                yaw_cost_ego = - abs(ego_state[3]) / (np.pi / 3)
                cost_ego += speed_cost_ego + yaw_cost_ego
                
        else:
            cost_ego = np.nan

        '''adversarial vehicle cost'''
        cost_ego_adv, cost_adv_adv, cost_adv_road = 0, 0, 0
        if self.r_adv == "r1":
            if 0 in self.CollidingVehs:
                cost_adv = 100
            elif len(self.CollidingVehs) != 0:
                cost_adv = -100
            else:
                cost_adv = 0

        elif self.r_adv == 'stackelberg':
            cost_adv = 0
            if done:
                if 0 in self.CollidingVehs:
                    cost_adv += 10
                elif len(self.CollidingVehs) != 0:
                    cost_adv += - 20 * len(self.CollidingVehs)
                else:
                    cost_adv += 0
            else:
                pass
        

        elif self.r_adv == 'stackelberg2':
            cost_adv = 0

            if done:
                
                if 0 in self.CollidingVehs:
                    cost_adv += 10
                else:
                    pass
            else:

                speed_cost_ego = ego_state[2] / self.max_speed
                # yaw_cost_ego = 1 - abs(ego_state[3]) / (np.pi / 3) * (ego_state[2] / self.max_speed)
                yaw_cost_ego = - abs(ego_state[3]) / (np.pi / 3) * 0
                cost_adv -= speed_cost_ego + yaw_cost_ego

        elif self.r_adv == 'stackelberg3':
            cost_adv = 0

            if done:
                
                if 0 in self.CollidingVehs:
                    cost_adv += 10
                elif len(self.CollidingVehs) != 0:
                    cost_adv += -20
                else:
                    pass
            else:

                speed_cost_ego = ego_state[2] / self.max_speed
                # yaw_cost_ego = 1 - abs(ego_state[3]) / (np.pi / 3) * (ego_state[2] / self.max_speed)
                yaw_cost_ego = - abs(ego_state[3]) / (np.pi / 3)
                cost_adv -= speed_cost_ego + yaw_cost_ego

                
        elif self.r_adv[0:2] == "r2":
            bv_bv_thresh = 1.5
            bv_road_thresh = float("inf")
            Rb = [100, -100]
            a, b, c = list(map(float, self.r_adv[3:].split('-')))
            # a, b, c = 1, 1, 0

            ego_col_cost_record, adv_col_cost_record, adv_road_cost_record = float('inf'), float('inf'), float('inf')
            for i in range(self.num_agents):
                car_ego = [ego_state[0], ego_state[1],
                           self.car_info[0][2][0], self.car_info[0][2][1], ego_state[3]]
                car_adv = [adv_state[i][0], adv_state[i][1],
                           self.car_info[i + 1][2][0], self.car_info[i + 1][2][1], adv_state[i][3]]
                dis_ego_adv = dist_between_cars(car_ego, car_adv)
                # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i][0]) ** 2 +
                #                         (ego_state[1] - adv_state[i][1]) ** 2)
                if dis_ego_adv < ego_col_cost_record:
                    ego_col_cost_record = dis_ego_adv
            cost_ego_adv = ego_col_cost_record

            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    car_adv_j = [adv_state[j][0], adv_state[j][1],
                                 self.car_info[j + 1][2][0], self.car_info[j + 1][2][1], adv_state[j][3]]
                    car_adv_i = [adv_state[i][0], adv_state[i][1],
                                 self.car_info[i + 1][2][0], self.car_info[i + 1][2][1], adv_state[i][3]]
                    dis_adv_adv = dist_between_cars(car_adv_i, car_adv_j)
                    # dis_adv_adv = np.sqrt((adv_state[i][0] - adv_state[j][0]) ** 2 +
                    #                       (adv_state[i][1] - adv_state[j][1]) ** 2)
                    if dis_adv_adv < adv_col_cost_record:
                        adv_col_cost_record = dis_adv_adv
            cost_adv_adv = min(adv_col_cost_record, bv_bv_thresh)

            road_up, road_low = 12, 0
            car_width = 1.8
            for i in range(self.num_agents):
                y = adv_state[i][1]
                dis_adv_road = min(road_up - (y + car_width / 2), (y - car_width / 2) - road_low)
                if dis_adv_road < adv_road_cost_record:
                    adv_road_cost_record = dis_adv_road
            cost_adv_road = min(adv_road_cost_record, bv_road_thresh)

            cost_adv = - a * cost_ego_adv + b * cost_adv_adv + c * cost_adv_road
            if 0 in self.CollidingVehs:
                cost_adv += Rb[0]
            elif len(self.CollidingVehs) != 0:
                cost_adv += Rb[1] 

        
        elif self.r_adv == "r3":
            ego_col_cost_record = float('inf')
            for i in range(self.num_agents):
                car_ego = [ego_state[0], ego_state[1],
                           self.car_info[0][2][0], self.car_info[0][2][1], ego_state[3]]
                car_adv = [adv_state[i][0], adv_state[i][1],
                           self.car_info[i + 1][2][0], self.car_info[i + 1][2][1], adv_state[i][3]]
                dis_ego_adv = dist_between_cars(car_ego, car_adv)
                # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i][0]) ** 2 +
                #                         (ego_state[1] - adv_state[i][1]) ** 2)
                if dis_ego_adv < ego_col_cost_record:
                    ego_col_cost_record = dis_ego_adv
            cost_ego_adv = ego_col_cost_record # min(distance between AV and BVs)
            cost_adv = -cost_ego_adv
            if done:
                if 0 in self.CollidingVehs:
                    cost_adv += 100
                elif len(self.CollidingVehs) != 0:
                    cost_adv += -100 * len(self.CollidingVehs)

        else:
            cost_adv = np.nan
        # ipdb.set_trace()
        return [cost_ego, cost_adv]
    
    def collision_test(self):
        '''
        car_info (list): all vehicles infos, [[ID(int), lane(str), [length(float), width(float)]], ...]
        car_state (list): all vehicles pos, [[x(float), y(float), speed(float), yaw(float)], ...]
        road_info (dict): road info, {lane(str): [minLaneMarking(float), maxLaneMarking(float)], ...}
        '''
        car_bound_list = [] # 4 bound points of each car
        car_info_list = [] # record info of car existed
        col_list = []

        # out-of-lane detection
        for i in range(len(self.car_info)):
            [ID, lane, [length, width]] = self.car_info[i]
            if ID in self.ArrivedVehs:
                continue
            [x, y, speed, yaw] = self.states[self.timestep][i * 4: (i + 1) * 4]
            [minLaneMarking, maxLaneMarking] = self.road_info[lane]
            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)
            matrix_car = [[x - width / 2 * sin_yaw - length * cos_yaw, y + width / 2 * cos_yaw - length * sin_yaw],
                          [x - width / 2 * sin_yaw, y + width / 2 * cos_yaw],
                          [x + width / 2 * sin_yaw, y - width / 2 * cos_yaw],
                          [x + width / 2 * sin_yaw - length * cos_yaw, y - width / 2 * cos_yaw - length * sin_yaw]]
            matrix_car = np.array(matrix_car)
            car_bound_list.append(matrix_car)
            car_info_list.append(self.car_info[i])
            y_max = np.max(matrix_car[:, 1])
            y_min = np.min(matrix_car[:, 1])
            if y_min < minLaneMarking < y_max or y_min < maxLaneMarking < y_max:
                col_list.append(ID)

        # collision detection
        # https://blog.csdn.net/m0_37660632/article/details/123925503
        def xmult(a, b, c, d):
            vectorAx = b[0] - a[0]
            vectorAy = b[1] - a[1]
            vectorBx = d[0] - c[0]
            vectorBy = d[1] - c[1]
            return (vectorAx * vectorBy - vectorAy * vectorBx)

        while len(car_info_list) != 0:
            ID_i = car_info_list.pop(0)[0]
            matrix_car_i = car_bound_list.pop(0)
            j = 0
            while j < len(car_info_list):
                ID_j = car_info_list[j][0]
                matrix_car_j = car_bound_list[j]
                collision = False
                for p in range(-1, 3):
                    c, d = matrix_car_i[p], matrix_car_i[p + 1]
                    for q in range(-1, 3):
                        a, b = matrix_car_j[q], matrix_car_j[q + 1]
                        xmult1 = xmult(c, d, c, a)
                        xmult2 = xmult(c, d, c, b)
                        xmult3 = xmult(a, b, a, c)
                        xmult4 = xmult(a, b, a, d)
                        if xmult1 * xmult2 < 0 and xmult3 * xmult4 < 0:
                            collision = True
                            break
                    if collision:
                        break
                if collision:
                    if ID_i not in col_list:
                        col_list.append(ID_i)
                    if ID_j not in col_list and ID_j not in self.CollidingVehs:
                        col_list.append(ID_j)
                j += 1
        self.CollidingVehs += col_list

    def record(self, filepath = None):
        if filepath is None:
            filepath = 'output/newoutput_test/record-' + datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv'
        record = []
        for t in range(self.timestep):
            state_t = self.states[t]
            action_t = self.actions[t]
            for id in range(self.num_agents + 1):
                
                if t > 0 and state_t[id * 4: (id + 1) * 4] == self.states[t - 1][id * 4: (id + 1) * 4]:
                    record_t_id = [t, id, -1]
                elif id == 0:
                    record_t_id = [t, id, 0]
                else:
                    record_t_id = [t, id, 1]
                record_t_id.extend(state_t[id * 4: (id + 1) * 4])
                record_t_id.extend(action_t[id * 2: (id + 1) * 2])
                record.append(record_t_id)
        np.savetxt(filepath, record)
        return filepath