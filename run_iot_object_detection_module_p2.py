# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

import time
import threading
import heapq
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

def calculate_inversions(history):

    inversions = []
    total_inversions = 0
    largest_inversion = 0

    history_sorted = sorted(history, key=lambda task: task.enqueue_time)

    for i, task_x in enumerate(history_sorted):
        inversion_time = 0
        
        for j in range(i + 1, len(history_sorted)):
            task_y = history_sorted[j]
            
            if task_y.priority > task_x.priority:
                overlap_start = max(task_x.enqueue_time, task_y.enqueue_time)
                overlap_end = min(task_x.enqueue_time + task_x.exec_time, task_y.enqueue_time + task_y.exec_time)
                
                if overlap_start < overlap_end:
                    inversion_time += (overlap_end - overlap_start)

        inversions.append(inversion_time)
        total_inversions += 1
        largest_inversion = max(largest_inversion, inversion_time)

    total_inversion_time = sum(inversions)
    avg_inversion = total_inversion_time / total_inversions if total_inversions > 0 else 0
    
    return inversions, total_inversion_time, avg_inversion, largest_inversion

inversions, total_inversion_time, avg_inversion, largest_inversion = calculate_inversions(iot_object_detection_module.history)
print("Inversions per task:", inversions)
print("Total Inversion Time:", total_inversion_time)
print("Average Inversion:", avg_inversion)
print("Largest Inversion:", largest_inversion)

# Problem 2

history_dict = {i: task.__dict__ for i, task in enumerate(iot_object_detection_module.history)}

# now pass this dictionary to the function
depth_group_avg_response_times = get_group_avg_response_time(history_dict)

# plt.figure(figsize=(10, 6))
# depth_ranges = [f"{i*10}-{(i+1)*10-1}m" for i in range(10)]  # ["0-10m", "10-20m", ..., "90-100m"]
# plt.bar(depth_ranges, depth_group_avg_response_times, color="skyblue")
# plt.xlabel("")
# plt.ylabel("Average Response Time (s)")
# plt.title("Average Response Time by Depth Group")
# plt.show()
