import numpy as np
import traci
from sumolib import checkBinary
import time
import os
import matplotlib.pyplot as plt
from PIL import Image


def sim_by_record(filepath):
    states = np.loadtxt(filepath)
    t = 0
    cfg_sumo = '../../config/lane_sim.sumocfg'
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
    command += ['--lanechange.duration', '5']
    command += ['--lateral-resolution', '0.0']
    traci.start(command)
    traci.simulationStep()
    for state in states:
        [timestep, carid, typeid, x, y, speed, yaw, delta_speed, delta_yaw] = state
        type = "AV" if typeid == 0 else "BV" if typeid == 1 else None
        if timestep != t and traci.vehicle.getIDList() != ():
            traci.simulationStep()
            time.sleep(0.01)
            t += 1
        if t == 0:
            traci.vehicle.add(
                vehID="car" + str(int(carid)),
                routeID="straight",
                typeID=type,
                depart=10,
                departLane="best",
                departPos=5.0,
                departSpeed=speed,
            )
            traci.vehicle.moveToXY(vehID="car" + str(int(carid)),
                                   x=x, y=y, angle=-yaw * 180 / np.pi + 90,
                                   edgeID=0, lane=0)
        elif "car" + str(int(carid)) not in traci.vehicle.getIDList():
            continue
        elif type is None:
            traci.vehicle.remove(vehID="car" + str(int(carid)))
        else:
            traci.vehicle.moveToXY(vehID="car" + str(int(carid)),
                                   x=x, y=y, angle=-yaw * 180 / np.pi + 90,
                                   edgeID=0, lane=0)
    traci.close()


def sim_by_record_and_draw(filepath):
    states = np.loadtxt(filepath)
    t = 0
    cfg_sumo = '../../config/lane_sim.sumocfg'
    sim_seed = 42
    app = "sumo"
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
    command += ['--lanechange.duration', '5']
    command += ['--lateral-resolution', '0.0']
    traci.start(command)
    traci.simulationStep()
    for state in states:
        [timestep, carid, typeid, x, y, speed, yaw, delta_speed, delta_yaw] = state
        type = "AV" if typeid == 0 else "BV" if typeid == 1 else None
        if timestep != t and traci.vehicle.getIDList() != ():
            traci.simulationStep()
            time.sleep(0.01)
            t += 1
        if t == 0:
            traci.vehicle.add(
                vehID="car" + str(int(carid)),
                routeID="straight",
                typeID=type,
                depart=10,
                departLane="best",
                departPos=5.0,
                departSpeed=speed,
            )
            traci.vehicle.moveToXY(vehID="car" + str(int(carid)),
                                   x=x, y=y, angle=-yaw * 180 / np.pi + 90,
                                   edgeID=0, lane=0)
        elif "car" + str(int(carid)) not in traci.vehicle.getIDList():
            continue
        elif type is None:
            traci.vehicle.remove(vehID="car" + str(int(carid)))
        else:
            traci.vehicle.moveToXY(vehID="car" + str(int(carid)),
                                   x=x, y=y, angle=-yaw * 180 / np.pi + 90,
                                   edgeID=0, lane=0)
    traci.close()


def video_to_image(filepath, savepath, timeF=10, mintime=0, maxtime=float('inf')):
    import cv2
    vc = cv2.VideoCapture(filepath)  # 读入视频文件
    c = 0
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    m = 1
    imgs = []
    last_frame = None
    while rval and c < maxtime / 0.04:  # 循环读取视频帧
        last_frame = frame
        if c < mintime / 0.04:
            continue
        if c % timeF == 0:  # 每隔timeF帧进行存储操作
            # print(frame)
            (h, w) = frame.shape[:2]
            center = (w / 2, h / 2)
            # print(center)
            M = cv2.getRotationMatrix2D(center, 0, 1.0)
            rotated = cv2.warpAffine(frame, M, (w, h))
            imgs.append(rotated)
            cv2.imshow('b', rotated)
            cv2.waitKey(0)
            m = m + 1
        c = c + 1
        cv2.waitKey(1)
        rval, frame = vc.read()
    (h, w) = last_frame.shape[:2]
    center = (w / 2, h / 2)
    # print(center)
    M = cv2.getRotationMatrix2D(center, 0, 1.0)
    rotated = cv2.warpAffine(last_frame, M, (w, h))
    # imgs.append(rotated)
    imgs[-1] = rotated
    vc.release()

    result = imgs[-1]
    for i in range(len(imgs) - 1):
        result = cv2.addWeighted(result, 1.0, imgs[i], 0.5, 0)
        # cv2.imshow('b', result)
        # cv2.waitKey(0)
    cv2.imwrite(savepath, result)
    # cv2.imshow('b', result)
    cv2.waitKey(0)

# -15.15, -7.34

def video_clip_and_rotate(filepath, mintime=0.0, maxtime=float('inf'), outputfile="test.mp4"):
    # 逆时针九十度并按时间裁剪
    import cv2
    vc = cv2.VideoCapture(filepath)  # 读入视频文件
    c = 0

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
        video_writer = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(*'mp4v'),
                                       60, [frame.shape[0], frame.shape[1]])
    else:
        rval = False
    while rval and c < maxtime * 60:  # 循环读取视频帧
        if c < mintime * 60:
            c += 1
            rval, frame = vc.read()
            continue
        frame = cv2.transpose(cv2.flip(frame, -1))  # 0是顺时针90度，-1是逆时针90度
        video_writer.write(frame)
        c += 1
        rval, frame = vc.read()
    vc.release()
    video_writer.release()
    cv2.destroyAllWindows()

def draw(filepath):
    plt.figure(facecolor='black')
    states = np.loadtxt(filepath)
    num_agents = np.max(states[:, 1]) + 1

    # scale = 0.003
    scale_1 = 380

    for i in range(int(num_agents)):
        states_i = states[np.where(states[:, 1] == i)]
        typeid = states_i[0][2]
        type = "AV" if typeid == 0 else "BV"
        c = "r" if type == "AV" else "y"
        trace = []
        t = 0
        for state in states_i:
            [timestep, carid, typeid, x, y, speed, yaw, delta_speed, delta_yaw] = state
            type = "AV" if typeid == 0 else "BV" if typeid == 1 else None
            if type is None:
                break
            x = x * scale_1
            y = y * scale_1
            trace.append([x, y])
            # if t % 10 == 0:
            #     plt.plot(x, y, 's', color=c, markersize=3)
            t += 1

        trace = np.array(trace).T
        plt.plot(trace[0], trace[1], c, alpha=0.5)

        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        x_rect = x + 1.8 * scale_1 / 2 * sin_yaw - 5 * scale_1 * cos_yaw
        y_rect = y - 1.8 * scale_1 / 2 * cos_yaw - 5 * scale_1 * sin_yaw
        rectangle = plt.Rectangle((x_rect, y_rect), 5 * scale_1, 1.8 * scale_1, color=c)
        rectangle.set_angle(yaw * 180 / np.pi)
        plt.gca().add_patch(rectangle)

        # image = Image.open('av.png') if type == "AV" else Image.open('bv.png')
        # image = image.rotate(45)

        # image = plt.imread('av.png') if type == "AV" else plt.imread('bv.png')
        # plt.imshow(image, extent=[x - 5 * scale_1, x + 2560 - 5 * scale_1,
        #                           y - 1.8 * scale_1, y + 926 - 1.8 * scale_1])

    x_max = (np.max(states[:, 3]) + 15) * scale_1
    x_min = -10 * scale_1
    x_line = np.arange(x_min, x_max, 10)
    y_line = np.ones_like(x_line) * 0 * scale_1
    plt.plot(x_line, y_line, "white")
    y_line = np.ones_like(x_line) * 4 * scale_1
    plt.plot(x_line, y_line, "white", linestyle='--', linewidth=2)
    y_line = np.ones_like(x_line) * 8 * scale_1
    plt.plot(x_line, y_line, "white", linestyle='--', linewidth=2)
    y_line = np.ones_like(x_line) * 12 * scale_1
    plt.plot(x_line, y_line, "white")
    plt.ylim(0, 12 * scale_1)
    plt.xlim(x_min, x_max)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.savefig('1.pdf', bbox_inches='tight', dpi=300)
    plt.show()



if __name__ == "__main__":
    # filepath = "../../output/simdata-ratio=0.5_av=sumo_bv=5-RL_r-ego=r1_r-adv=r3_seed=42_time=23-03-09-17-36-23/avarrive/"
    # filepath = "E:\\Scenario_Generation\\output\\(Re)2_H2O\\=2"
    for i in range(1):
        bv = 3
        filepath = "/home/qh802/cqm/Cross-Learning/Scripts/eval_output/EVAL_bv=5-RL_r-ego=r1_r-adv=r1_seed=42_time=2023-09-18-15-24-43reg_reward=True_pretrain_adv=RL/pretrained-AVvsRL-BV/avcrash_0"
        pathlist = os.listdir(filepath)
        num = 0
        for path in pathlist:
            print(num, path)
            num += 1
            # sim_by_record(filepath + path)
            draw(os.path.join(filepath, path))

        # filepath = "E:\\save_finetuned\\bv=" + str(bv) + "\\" + str(i) + "\\avarrive\\"
        # pathlist = os.listdir(filepath)
        # num = 0
        # for path in pathlist:
        #     print(num, path)
        #     num += 1
        #     sim_by_record(filepath + path)

    # # # # # # # # # # # # # # # # # # # # #
    # video_path = "D:\\RenKun\\Documents\\oCam\\bv=2\\"
    # image_path = "D:\\RenKun\\Documents\\oCam\\bv=2\\"
    # for p in os.listdir(video_path):
    #     if "clip" not in p:
    #         continue
    #     print(p)
    #     video_to_image(video_path + p, image_path + p.split('.')[0] + '.png', timeF=20)
    # p = "4bv-2.mp4"
    # video_to_image(video_path + p, image_path + p.split('.')[0] + '.png', timeF=10)
    # import cv2
    # for p in os.listdir(image_path):
    #     x = cv2.imread(image_path + p)
    #     y = x[:, 44: 164, :]
    #     cv2.imwrite(image_path + p, y)
    #

    # # # # # # # # # # # # # # # # # # #
    # rootpath = "D:\\RenKun\\Documents\\oCam\\bv=2\\"
    # mintime = 0
    # maxtime = 2.25
    # f1 = os.listdir(rootpath)[1]
    # f = '录制_2023_05_15_11_08_21_665.mp4'
    # output = rootpath + "clip_" + f
    # video_clip_and_rotate(rootpath + f, mintime=mintime, maxtime=maxtime, outputfile=output)