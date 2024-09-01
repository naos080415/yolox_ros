#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from detected_object_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image

class Visualizer:
    def __init__(self):
        rospy.init_node('human_detector_debug_node', anonymous=True)
        self.bridge = CvBridge()

        self.debug_pub = rospy.Publisher('/human_detector/debug/image', Image, queue_size=1)
        self.detected_human_sub = rospy.Subscriber('/human_detector/human_info', BoundingBoxes, self.callback, queue_size=1)

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
        bboxes = msg.bounding_boxes
        
        color = (255, 0, 0)
        for bbox in bboxes:
            probability = bbox.probability
            xmin = bbox.xmin
            ymin = bbox.ymin
            xmax = bbox.xmax
            ymax = bbox.ymax
            dist = bbox.dist
            cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), color, 2)

        debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        self.debug_pub.publish(debug_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        vis = Visualizer()
        vis.run()
    except rospy.ROSInterruptException:
        pass
