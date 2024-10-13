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

    for i, box in enumerate(cluster_boxes_data):
        coord = [box[0], box[1], box[2], box[3]]
        depth = box[4]
        bbox_id = i

        task = TaskEntity(image_path= frame.path,
                          coord=coord,
                          depth=depth,
                          bbox_id=bbox_id
        )
        tasks.append(task)
    
    return tasks
    # student's code here


    