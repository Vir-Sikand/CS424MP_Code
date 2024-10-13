# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

from IoTObjectDetection.IoTObjectDetectionModuleHelperFunctions import *
from IoTObjectDetection.TaskEntity import *


# read the input bounding box data from file
box_info = read_json_file('../../dataset/waymo_ground_truth_flat.json')


def process_frame(frame):
    """Process frame for scheduling.

    Process a image frame to obtain cluster boxes and corresponding scheduling parameters
    for scheduling. 

    Student's code here.

    Args:
        param1: The image frame to be processed. 

    Returns:
        A list of tasks with each task containing image_path and other necessary information. 
    """

    cluster_boxes_data = get_bbox_info(frame, box_info)
    tasks = []

    for bbox in cluster_boxes_data:
        x1, y1, x2, y2, depth= bbox[:5]

        area = (x2 - x1) * (y2 - y1) #added priority

        prio = 0.1 * (1/area) + 0.9 * (depth) 

        task = TaskEntity(image_path= frame.path,
                          coord=[x1, y1, x2, y2],
                          priority=prio,
                          depth=depth
        )
        tasks.append((prio, task))
    
    tasks.sort()
    sorted_tasks = []
    for duo in tasks :
        sorted_tasks.append(duo[1])

    return sorted_tasks
    # student's code here


    


#### NEW PRIORITY FORMULA:
## 1/AREA * 0.1 + Depth*0.9
