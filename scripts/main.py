#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch

from utils.yolox import YOLOX
from detected_object_msgs.msg import BoundingBoxes
from detected_object_msgs.msg import BoundingBox
from sensor_msgs.msg import Image

class HumanDetector:
    def __init__(self):
        rospy.init_node('human_detector_node', anonymous=True)
        camera_topic = rospy.get_param('~camera_topic', '/usb_cam/image_raw')
        model_name = rospy.get_param('~model_name', 'yolov5s')

        self.model = YOLOX()
        self.bridge = CvBridge()

        self.detected_human_pub = rospy.Publisher('/human_detector/human_info', BoundingBoxes, queue_size=1)
        self.image_sub = rospy.Subscriber(camera_topic, Image, self.callback, queue_size=1)

    def calc_dist(self):
        return 0

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        object_info = self.model.predict(cv_image)

        human_bbox_positions, human_bbox_probability = [], []
        for key, values in object_info.items():
            if key != 'person':
                continue
            
            for box, prob in zip(values['bbox'], values['probability']):
                human_bbox_positions.append(box)
                human_bbox_probability.append(prob)

        detected_human_msg = BoundingBoxes()
        detected_human_msg.image = msg

        for box, prob in zip(human_bbox_positions, human_bbox_probability):
            bounding_box = BoundingBox()
            xmin, ymin, xmax, ymax = box

            bounding_box.probability = prob
            bounding_box.Class = 'person'
            bounding_box.xmin = xmin
            bounding_box.ymin = ymin
            bounding_box.xmax = xmax
            bounding_box.ymax = ymax
            bounding_box.dist = self.calc_dist()
            detected_human_msg.bounding_boxes.append(bounding_box)

        self.detected_human_pub.publish(detected_human_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        yolo_ros_node = HumanDetector()
        yolo_ros_node.run()
    except rospy.ROSInterruptException:
        pass
