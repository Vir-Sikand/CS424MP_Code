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

print("Total Elapsed time: %fs" % (end_time - start_time))

# Problem 2

# convert the list of TaskEntity objects into a dictionary format
history_dict = {i: task.__dict__ for i, task in enumerate(iot_object_detection_module.history)}

# now pass this dictionary to the function
depth_group_avg_response_times = get_group_avg_response_time(history_dict)

plt.figure(figsize=(10, 6))
depth_ranges = [f"{i*10}-{(i+1)*10-1}m" for i in range(10)]  # ["0-10m", "10-20m", ..., "90-100m"]
plt.bar(depth_ranges, depth_group_avg_response_times, color="skyblue")
plt.xlabel("")
plt.ylabel("Average Response Time (s)")
plt.title("Average Response Time by Depth Group")
plt.show()
