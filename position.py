import math
import pyproj


class position:
    def __init__(self, width, height, pos):
        self.pos = pos
        self.img_width = width
        self.img_height = height
        self.longitude = 130
        self.latitude = 40
        self.person_height = 1.8
        self.person_width = 0.6

    def calculate_center(self):
        left = self.pos[0]
        top = self.pos[1]
        right = self.pos[2]
        bottom = self.pos[3]

        center = [(left + right) / 2, (top + bottom) / 2]
        return center

    def calculate_size(self):
        left = self.pos[0]
        top = self.pos[1]
        right = self.pos[2]
        bottom = self.pos[3]

        size = (right - left) + (bottom - top)
        return size

    def calculate_distance(self, size, center_pos):
        xx = (center_pos[0] - self.img_width / 2) ** 2
        yy = (center_pos[1] - self.img_height / 2) ** 2
        ratio = size / (self.person_width + self.person_height)

        distance = ratio * math.sqrt(xx + yy)
        return distance

    def calculate_angle(self):
        x = self.pos[0] - self.img_width / 2
        y = -self.pos[1] + self.img_height / 2

        angle = -(180 / math.pi) * math.atan(y / x)
        if x > 0:
            angle = angle + 90
        else:
            angle = angle - 90 + 360

        return angle * math.pi / 180

    def calculate_position(self, _distance, _angle):
        grs80 = pyproj.Geod(ellps='GRS80')
        obj_longitude, obj_latitude, back_angle = grs80.fwd(self.longitude, self.latitude, _angle, _distance)
        posi = [obj_longitude, obj_latitude]
        return posi



