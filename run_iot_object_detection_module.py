# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

import time
import threading
import matplotlib.pyplot as plt
import numpy as np
from IoTObjectDetection.IoTObjectDetectionModule import *


start_time = time.time()

print("Simulation started at the real-time: ", getRealTimeInPrintFormat())

iot_object_detection_module = iot_object_detection_module(run_yolo_flag = True)

iot_object_detection_module.run()
    
iot_object_detection_module.print_history()
iot_object_detection_module.visualize_history()

end_time = time.time()

response_times = [task.response_time for task in iot_object_detection_module.history if task.response_time is not None]
overall_avg_response_time = sum(response_times) / len(response_times) if response_times else 0

print("Simulation ended at the real-time: ", getRealTimeInPrintFormat())

# Problem 1
print("Average response time for processing the entire image dataset: {:.2f} seconds".format(overall_avg_response_time))

# Problem 2
data = get_group_avg_response_time(iot_object_detection_module.history)
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('whatever')
plt.ylabel('Avg. Response Time')
plt.title('Avg. Reponse time of bbox')
plt.show()
print("Total Elapsed time: %fs" % (end_time - start_time))
