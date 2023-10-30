# https://zhuanlan.zhihu.com/p/569701615
# -*- coding:utf-8 -*-

"""计算平面2个矩形最小距离的模块"""
import math
import time


class Point:
    """平面点"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def rotate(self, theta):
        """旋转算子（坐标系逆时针旋转theta弧度）"""
        x = self.x * math.cos(theta) + self.y * math.sin(theta)
        y = -self.x * math.sin(theta) + self.y * math.cos(theta)
        return Point(x, y)

    def translate(self, x, y):
        """平移算子"""
        self.x += x
        self.y += y
        return Point(self.x, self.y)


def dist_between_points(p1: Point, p2: Point) -> float:
    """平面2点距离"""
    return math.sqrt(abs(p1.x - p2.x) ** 2 + abs(p1.y - p2.y) ** 2)


class Line:
    """平面线段"""

    def __init__(self, p1: Point, p2: Point):
        self.P1 = p1
        self.P2 = p2

        self.A, self.B, self.C = self._solve()

    def _solve(self):
        """求直线A、B、C参数"""
        if math.isclose(self.P1.y, self.P2.y):
            A = 0
            B = 1
            C = -self.P1.y
        elif math.isclose(self.P1.x, self.P2.x):
            A = 1
            B = 0
            C = -self.P1.x
        else:
            A = (self.P2.y - self.P1.y) / (self.P1.x - self.P2.x)
            B = 1
            C = -A * self.P1.x - self.P1.y
        return A, B, C

    def dist_to_point(self, p: Point) -> float:
        """点到线段距离"""
        # 点到直线距离
        dist = abs(self.A * p.x + self.B * p.y + self.C) / math.sqrt(self.A ** 2 + self.B ** 2)

        # 垂足坐标
        X = (self.B ** 2 * p.x - self.A * self.B * p.y - self.A * self.C) / (self.A ** 2 + self.B ** 2)
        Y = (self.A ** 2 * p.y - self.A * self.B * p.x - self.B * self.C) / (self.A ** 2 + self.B ** 2)
        P = Point(X, Y)

        if P in self:
            # 垂足在线段内
            return dist
        else:
            # 垂足不在线段内时，直接返回inf
            return float('inf')

    def __contains__(self, item: Point) -> bool:
        """判断直线上的点是否在线段内"""
        return min(self.P1.x, self.P2.x) <= item.x <= max(self.P1.x, self.P2.x) and min(
            self.P1.y, self.P2.y) <= item.y <= max(self.P1.y, self.P2.y)


def cross(l1: Line, l2: Line) -> bool:
    """平面2线段有无交点"""
    delta = l1.A * l2.B - l2.A * l1.B
    if math.isclose(delta, 0):                  # 平行线
        if math.isclose(l1.C, l2.C):            # 直线重合
            if l1.P1 in l2 or l1.P2 in l2:      # 线段重合
                return True
    else:
        # 直线交点坐标
        X = (l2.C * l1.B - l1.C * l2.B) / delta
        Y = (l1.C * l2.A - l2.C * l1.A) / delta
        P = Point(X, Y)
        # 判断交点是否在两条线段上
        if P in l1 and P in l2:
            return True
    return False


class Rect:
    """平面矩形"""

    def __init__(self, center_x: float, head_y: float, length: float, width: float, yaw: float):
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        # matrix_car = [[center_x - width / 2 * sin_yaw - length * cos_yaw, head_y + width / 2 * cos_yaw - length * sin_yaw],
        #               [center_x - width / 2 * sin_yaw, head_y + width / 2 * cos_yaw],
        #               [center_x + width / 2 * sin_yaw, head_y - width / 2 * cos_yaw],
        #               [center_x + width / 2 * sin_yaw - length * cos_yaw, head_y - width / 2 * cos_yaw - length * sin_yaw]]

        # 4个顶点(以矩形自身坐标系的象限排序)
        self.A = Point(center_x - width / 2 * sin_yaw - length * cos_yaw, head_y + width / 2 * cos_yaw - length * sin_yaw)
        self.B = Point(center_x - width / 2 * sin_yaw, head_y + width / 2 * cos_yaw)
        self.C = Point(center_x + width / 2 * sin_yaw, head_y - width / 2 * cos_yaw)
        self.D = Point(center_x + width / 2 * sin_yaw - length * cos_yaw, head_y - width / 2 * cos_yaw - length * sin_yaw)
        self.points = [self.A, self.B, self.C, self.D]
        self.lines = [Line(self.A, self.B), Line(self.B, self.C), Line(self.C, self.D), Line(self.D, self.A)]

    # def area(self) -> float:
    #     """面积"""
    #     return self.length * self.width


def dist_between_rectangles(ego: Rect, obj: Rect) -> float:
    """平面2矩形距离"""
    # 相交判断
    for L1 in ego.lines:
        for L2 in obj.lines:
            if cross(L1, L2):
                return 0
    # 考虑仿真场景下，额外判断大矩形是否完全包围小矩形
    # if ego.area() >= obj.area():
    #     flag = contains(ego, obj)
    # else:
    #     flag = contains(obj, ego)
    # if flag:
    #     return -1
    # 不考虑相交的情况，矩形最小距离只会出现在以下情况中
    lst = []
    for P1 in ego.points:       # 顶点距离
        for P2 in obj.points:
            lst.append(dist_between_points(P1, P2))
    for L in ego.lines:         # 顶点到边的距离
        for P in obj.points:
            lst.append(L.dist_to_point(P))
    for L in obj.lines:
        for P in ego.points:
            lst.append(L.dist_to_point(P))
    return min(lst)


def dist_between_cars(car1, car2):
    # car1, car2: list [x, y, length, width, yaw]
    R1 = Rect(car1[0], car1[1], car1[2], car1[3], car1[4])
    R2 = Rect(car2[0], car2[1], car2[2], car2[3], car2[4])
    return dist_between_rectangles(R1, R2)

# def contains(big: Rect, small: Rect) -> bool:
#     """判断大矩形是否完全包围小矩形"""
#     A_adj = big.A.rotate(-big.heading)
#     B_adj = big.B.rotate(-big.heading)
#     C_adj = big.C.rotate(-big.heading)
#     D_adj = big.D.rotate(-big.heading)
#
#     E_adj = small.A.rotate(-big.heading)
#     F_adj = small.B.rotate(-big.heading)
#     G_adj = small.C.rotate(-big.heading)
#     H_adj = small.D.rotate(-big.heading)
#
#     x_max = max(A_adj.x, B_adj.x, C_adj.x, D_adj.x)
#     x_min = min(A_adj.x, B_adj.x, C_adj.x, D_adj.x)
#     y_max = max(A_adj.y, B_adj.y, C_adj.y, D_adj.y)
#     y_min = min(A_adj.y, B_adj.y, C_adj.y, D_adj.y)
#
#     if (x_min <= E_adj.x <= x_max and y_min <= E_adj.y <= y_max) \
#             and (x_min <= F_adj.x <= x_max and y_min <= F_adj.y <= y_max) \
#             and (x_min <= G_adj.x <= x_max and y_min <= G_adj.y <= y_max) \
#             and (x_min <= H_adj.x <= x_max and y_min <= H_adj.y <= y_max):
#         return True
#     else:
#         return False


if __name__ == "__main__":
    car1 = [2, 1, 2, 2, 0]
    car2 = [4, 3, 2, 2, 0]
    t = time.time()
    for i in range(1000):
        dist_between_cars(car1, car2)
    print(time.time() - t)
