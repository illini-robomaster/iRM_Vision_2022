from matplotlib.pyplot import close
import numpy as np

ARMOR_HEIGHT = 135
ARMOR_WIDTH_S = 135
ARMOR_WIDTH_B = 235

class Aim:
    def __init__(self, focal_len, sensor_w, sensor_h):
        self.focal_len = focal_len
        self.sensor_w = sensor_w
        self.sensor_h = sensor_h

    def get_closest(self, pred_n4):
        '''
            Args:
                -pred_n4: nd_array, prediction result from yolo in format [(x, y, w, h)]
            
            Return:
                - [x,y,w,h] using rough estimation that assumes no camera distortion and armor
                plate is facing the camera
        '''

        # rough estimation of distance
        d1_n = (416 / pred_n4[:, 2]) * ARMOR_WIDTH_S * self.focal_len / self.sensor_w
        d2_n = (416 / pred_n4[:, 2]) * ARMOR_WIDTH_B * self.focal_len / self.sensor_w
        d3_n = (416 / pred_n4[:, 3]) * ARMOR_HEIGHT * self.focal_len / self.sensor_h
        print(d1_n, d2_n, d3_n)
        d_n = (d1_n + d2_n + d3_n) / 3

        closest_d = d_n[np.argmin(d_n)]

        return np.concatenate((pred_n4[np.argmin(d_n)], [closest_d]))

    def get_rotation(self, pred_n4):
        '''
            Args:
                - pred_n4 : nd_array, Yolo prediction result in format [(x,y,w,h)]
            
            Return:
                - hori_offset, vert_offset: 2 floats of horizontal and vertical offset of 
                desire target from center of screen in milimiters
        '''
        closet_5 = self.get_closest(pred_n4)
        print(closet_5)
        x, y, w, h, d = closet_5
        x_center = x + w / 2
        y_center = y + h / 2

        perception_height = self.sensor_h * d / self.focal_len
        perception_width = self.sensor_w * d / self.focal_len

        return ((208.0 - x_center) / 416) * perception_width, ((208 - y_center) / 416) * perception_height

if __name__ == "__main__":
    test = Aim(16, 4.8, 3.2)
    pred_4n = np.array([[52.5, 66.6, 50, 40]])
    print(test.get_rotation(pred_4n))