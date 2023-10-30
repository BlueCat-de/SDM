import time
import traci
import numpy as np
import random
from sumolib import checkBinary
import optparse
import sys

class fvdm_model(object):
    def __init__(self, carID, dt=0.04):
        self.carID = carID
        self.buslen = 10
        self.carlen = 5
        self.minGap = 1.5  # db-carlen
        self.follow_distance = 50
        self.lanechange_time = 5
        self.p = 0  # driver reaction time
        self.dt = dt
        self.type = traci.vehicle.getTypeID(self.carID)  # vehicle type
        self.vmax = traci.vehicle.getMaxSpeed(self.carID)  # max speed
        self.maxacc = traci.vehicle.getAccel(self.carID)  # max acceleration
        self.maxdec = traci.vehicle.getDecel(self.carID)  # max deceleration
        self.length = traci.vehicle.getLength(self.carID)  # car length

        self.speed = traci.vehicle.getSpeed(self.carID)
        self.lane = traci.vehicle.getLaneID(self.carID)  # car lane
        self.lanePosition = traci.vehicle.getLanePosition(self.carID)  # car lane position


    # front car id
    def frontCar(self):
        m = float('inf')
        vehicle_frontCarID = ""
        for carID in traci.vehicle.getIDList():  # find front car
            lanePosition = traci.vehicle.getLanePosition(carID)
            if traci.vehicle.getLaneID(carID) == self.lane \
                    and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_frontCarID = carID
        return vehicle_frontCarID

    # id of the car in front of the adjacent lane
    def nearFrontCar(self):
        m = float('inf')
        vehicle_nearFrontCarID_0 = ""  # adjacent front car in lane 0
        vehicle_nearFrontCarID_1 = ""  # adjacent front car in lane 1
        vehicle_nearFrontCarID_2 = ""  # adjacent front car in lane 2
        for carID in traci.vehicle.getIDList():
            lanePosition = traci.vehicle.getLanePosition(carID)
            if (self.lane == "lane0" or self.lane == "lane2") and traci.vehicle.getLaneID(carID) == "lane1" \
                    and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:  # Find the adjacent lane vehicle id of the far right or far left lane
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_nearFrontCarID_1 = carID
            elif self.lane == "lane1" and self.lanePosition < lanePosition \
                    and lanePosition - self.lanePosition < self.follow_distance:  # Find the adjacent lane vehicle id of the middle lane
                if traci.vehicle.getLaneID(carID) == "lane0":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearFrontCarID_0 = carID
                if traci.vehicle.getLaneID(carID) == "lane2":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearFrontCarID_2 = carID
        return vehicle_nearFrontCarID_0, vehicle_nearFrontCarID_1, vehicle_nearFrontCarID_2

    # id of the car behind the adjacent lane
    def nearRearCar(self):
        m = float('inf')
        vehicle_nearRearCarID_0 = ""  # Adjacent rear car in lane 0
        vehicle_nearRearCarID_1 = ""  # Adjacent rear car in lane 1
        vehicle_nearRearCarID_2 = ""  # Adjacent rear car in lane 2
        for carID in traci.vehicle.getIDList():
            lanePosition = traci.vehicle.getLanePosition(carID)
            if (self.lane == "lane0" or self.lane == "lane2") and traci.vehicle.getLaneID(carID) == "lane1" \
                    and lanePosition < self.lanePosition:  # Find the right-most lane or left-most adjacent lane vehicle id
                if lanePosition - self.lanePosition < m:
                    m = lanePosition - self.lanePosition
                    vehicle_nearRearCarID_1 = carID
            elif self.lane == "lane1" and lanePosition < self.lanePosition:  # Find the adjacent lane vehicle id of the far right lane
                if traci.vehicle.getLaneID(carID) == "lane0":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearRearCarID_0 = carID
                if traci.vehicle.getLaneID(carID) == "lane2":
                    if lanePosition - self.lanePosition < m:
                        m = lanePosition - self.lanePosition
                        vehicle_nearRearCarID_2 = carID
        # print(vehicle_nearRearCarID_0, vehicle_nearRearCarID_1, vehicle_nearRearCarID_2)
        return vehicle_nearRearCarID_0, vehicle_nearRearCarID_1, vehicle_nearRearCarID_2

    # Generates the next time velocity
    def speed_generate(self):
        v_next = self.speed
        Pslow, Ps = 0, 0  # Slow start and random slowing probability
        # Pslow, Ps = 0.5, 0.3  # Slow start and random slowing probability
        frontCar = self.frontCar()
        if frontCar != "":
            frontCarSpeed = traci.vehicle.getSpeed(frontCar)  # Ahead speed
            frontCarDistance = traci.vehicle.getLanePosition(frontCar)  # The front car travels through the distance
            minAccSpeed = min(self.speed + self.maxacc, self.vmax)
            if self.speed == 0 and random.uniform(0, 1) < Pslow:  # Slow start phenomenon
                v_next = 0  # Next time the velocity is 0
            elif frontCarSpeed + frontCarDistance - (
                    minAccSpeed + self.speed) / 2 - self.lanePosition > self.minGap + self.length:  # Acceleration condition
                v_next = minAccSpeed
                if random.uniform(0, 1) < Ps:  # Random slowing down phenomenon
                    v_next = max(v_next - self.maxdec, 0)
            elif frontCarSpeed + frontCarDistance - (
                    minAccSpeed + self.speed) / 2 - self.lanePosition == self.minGap + self.length:  # Uniform condition
                if random.uniform(0, 1) < Ps:  # Random slowing down phenomenon
                    v_next = max(v_next - self.maxdec, 0)
            else:  # Deceleration condition
                v_next = max(self.speed - self.maxdec, 0)
        # TODO: It does not take into account the situation that the rear car and the side rear car accelerate to follow the car, and the side front car slows down
        else:
            v_next = min(self.speed + self.maxacc, self.vmax)
        return v_next

    # Determine whether to change lanes
    def changeLane(self):
        ifChangeLane = False  # Change lanes or not
        leftChangeLane = False  # Left lane change
        rightChangeLane = False  # Change lanes to the right
        Prc, Plc = 0.6, 0.9  # TODO: The probability of changing lanes to the right and left
        nearFrontCar_0 = self.nearFrontCar()[0]
        nearFrontCar_1 = self.nearFrontCar()[1]
        nearFrontCar_2 = self.nearFrontCar()[2]
        nearRearCar_0 = self.nearRearCar()[0]
        nearRearCar_1 = self.nearRearCar()[1]
        nearRearCar_2 = self.nearRearCar()[2]
        frontCar = self.frontCar()
        minAccSpeed = min(self.speed + self.maxacc, self.vmax)
        # 0. No car in front, or too close to the front car, give up changing lanes
        if frontCar == "" or traci.vehicle.getLanePosition(frontCar) - self.lanePosition < self.minGap + self.length:
            ...
        # 1. Left lane change to center
        elif self.lane == "lane2":
            # If there is a left rear vehicle and the distance does not meet the lane change requirements, the lane change is abandoned
            if nearRearCar_1 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) < self.minGap + self.length:
                ...
            # If there is a front-left vehicle, and the distance does not meet the lane change requirements, or the lane change is not a better choice, then the lane change is abandoned
            elif nearFrontCar_1 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # Meet the requirement of lane change, calculate the probability of lane change
            elif random.uniform(0, 1) <= Prc:
                ifChangeLane = True
        # 2. Right lane change to center
        elif self.lane == "lane0":
            # If there is a right rear vehicle and the distance does not meet the lane change requirements, the lane change is abandoned
            if nearRearCar_1 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_1) < self.minGap + self.length:
                ...
            # If there is a front right vehicle, and the distance does not meet the lane change requirements, or the lane change is not a better choice, then the lane change is abandoned
            elif nearFrontCar_1 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_1) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # Meet the requirement of lane change, calculate the probability of lane change
            elif random.uniform(0, 1) <= Plc:
                ifChangeLane = True
        # 3. Center lane change
        elif self.lane == "lane1":
            # 3.1. The center lane changes to the left
            # If there is a left rear vehicle and the distance does not meet the lane change requirements, the lane change is abandoned
            if nearRearCar_2 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(nearRearCar_2) < self.minGap + self.length:
                ...
            # If there is a front-left vehicle, and the distance does not meet the lane change requirements, or the lane change is not a better choice, then the lane change is abandoned
            elif nearFrontCar_2 != "" \
                    and (traci.vehicle.getLanePosition(nearFrontCar_2) - self.lanePosition < self.minGap + self.length
                         or traci.vehicle.getLanePosition(nearFrontCar_2) - self.lanePosition <
                         traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # Meet the requirement of lane change, calculate the probability of lane change
            elif random.uniform(0, 1) <= Plc:
                ifChangeLane = True
                leftChangeLane = True
            # 3.2. Change the center lane to the right
            if ifChangeLane:
                ...
            # If there is a right rear vehicle and the distance does not meet the lane change requirements, the lane change is abandoned
            elif nearRearCar_0 != "" \
                    and self.lanePosition - traci.vehicle.getLanePosition(
                nearRearCar_0) < self.minGap + self.length:
                ...
            # If there is a front right vehicle, and the distance does not meet the lane change requirements, or the lane change is not a better choice, then the lane change is abandoned
            elif nearFrontCar_0 != "" \
                    and (
                    traci.vehicle.getLanePosition(nearFrontCar_0) - self.lanePosition < self.minGap + self.length
                    or traci.vehicle.getLanePosition(nearFrontCar_0) - self.lanePosition <
                    traci.vehicle.getLanePosition(frontCar) - self.lanePosition):
                ...
            # Meet the requirement of lane change, calculate the probability of lane change
            elif random.uniform(0, 1) <= Prc:
                ifChangeLane = True
                rightChangeLane = True

        return [ifChangeLane, leftChangeLane, rightChangeLane]

    def run(self):
        self.speed = traci.vehicle.getSpeed(self.carID)
        self.lane = traci.vehicle.getLaneID(self.carID)  # Return driveway
        self.lanePosition = traci.vehicle.getLanePosition(self.carID)  # Return distance traveled by vehicle
        # if traci.vehicle.getLanePosition(carID):  # The vehicle is controlled from the lane
        changeLane = self.changeLane()
        if self.lane == "lane0" or self.lane == "lane2":  # If you're in the far right lane or the far left lane
            if changeLane[0]:
                traci.vehicle.changeLane(self.carID, 1, self.lanechange_time)
            else:  # If no lane change is performed, then update speed and position, the same below
                speed_next = self.speed_generate()
                traci.vehicle.setSpeed(self.carID, speed_next)
                traci.vehicle.changeLane(self.carID, traci.vehicle.getLaneIndex(self.carID), 0)
                # traci.vehicle.moveTo(self.carID, self.lane, speed_next * (1 + self.p) * self.dt + self.lanePosition)

        elif self.lane == "lane1":  # If it's in the middle lane
            if changeLane[0]:
                if changeLane[1]:
                    traci.vehicle.changeLane(self.carID, 2, self.lanechange_time)
                elif changeLane[2]:
                    traci.vehicle.changeLane(self.carID, 0, self.lanechange_time)
            else:
                speed_next = self.speed_generate()
                traci.vehicle.setSpeed(self.carID, speed_next)
                traci.vehicle.changeLane(self.carID, traci.vehicle.getLaneIndex(self.carID), 0)
                # traci.vehicle.moveTo(self.carID, self.lane, speed_next * (1 + self.p) * self.dt + self.lanePosition)


if __name__ == "__main__":
    cfg_sumo = '/home/qh802/cqm/Cross-Learning/Scripts/config/lane_sim.sumocfg'
    sim_seed = 42
    app = "sumo-gui"
    command = [checkBinary(app), '-c', cfg_sumo]
    command += ['--routing-algorithm', 'dijkstra']
    # command += ['--collision.action', 'remove']
    command += ['--seed', str(sim_seed)]
    command += ['--no-step-log', 'True']
    command += ['--time-to-teleport', '300']
    command += ['--no-warnings', 'True']
    command += ['--duration-log.disable', 'True']
    command += ['--waiting-time-memory', '1000']
    command += ['--eager-insert', 'True']
    command += ['--lanechange.duration', '2']
    command += ['--lateral-resolution', '0.0']
    traci.start(command)

    cur_time = float(traci.simulation.getTime())
    traci.vehicle.add(vehID="veh0", routeID="straight", typeID="AV",
                      depart=cur_time, departLane=1, arrivalLane=0, departPos=0.0, arrivalPos=float('inf'),
                      departSpeed=5)
    traci.vehicle.add(vehID="veh1", routeID="straight", typeID="BV", arrivalLane=2,
                      depart=cur_time, departLane=1, departPos=40.0,
                      departSpeed=5)
    
    traci.simulationStep()
    car0 = fvdm_model("veh0")
    car1 = fvdm_model("veh1")
    for t in range(500):
        time.sleep(0.04)
        car0.run()
        car1.run()
        traci.simulationStep()
    traci.close()