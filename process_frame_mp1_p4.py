# UIUC CS 424 Machine Problems 
# Designed and developed by Md Iftekharul Islam Sakib, Ph.D. Student, CS, UIUC (miisakib@gmail.com) under the supervision of Prof. Tarek Abdelzaher, CS, UIUC (zaher@illinois.edu)

from IoTObjectDetection.IoTObjectDetectionModuleHelperFunctions import *
from IoTObjectDetection.TaskEntity import *


# read the input bounding box data from file
box_info = read_json_file('../../dataset/waymo_ground_truth_flat.json')


iteration = 0
depduplication = {}

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
    global iteration
    iteration += 1
    for bbox in cluster_boxes_data:
        x1, y1, x2, y2, depth= bbox[:5]

        change, found = check_change(bbox)
        if not found :
            depduplication[(x1, y1, x2, y2)] = iteration

        elif not change :
            continue
    
        
        area = (x2 - x1) * (y2 - y1) #added priority

        prio = int((0.1*(area/(1920 * 1280)) + 0.9*(depth/100))*100)

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

def check_change(bbox) :
    x1, y1, x2, y2, depth= bbox[:5]
    for coordinates in depduplication :
        X1, Y1, X2, Y2 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        left_x = abs(X1 - x1)
        bottom_y = abs(Y1 - y1)
        right_x = abs(X2 - x2)
        top_y = abs(Y2 - y2)
        #check distane between this box and previous box
        if left_x + bottom_y + right_x + top_y < 25 :
            #box hasn't been updated in 10 frames case:
            if iteration - depduplication[coordinates] >= 10 :
                del depduplication[coordinates]
                depduplication[(x1, y1, x2, y2)] = iteration
                return True, True
            #it's okay we can update later case
            else :
                return False, True
            
    #no similar box found case        
    return False, False



    


#### NEW PRIORITY FORMULA:
## int((0.1*(area/(1920 * 1280)) + 0.9*(depth/100))*100)

#part 4, don't add every bounding box, if its moved enough add it, or if its been 10 frames add it

# Without Duplicaton Removal:
#396    frame_camera_197.png      (664,645), (845,771)  37.814        34   19700.000        3.000       28.000      19728.000    19700.100         1
# History visualization started at the real-time:  2024-10-13 18:22:34
# History visualization completed at the real-time:  2024-10-13 18:23:36
# History visualization took a total time of  62.12863802909851 s
# Simulation ended at the real-time:  2024-10-13 18:23:36
# Average response time for processing the entire image dataset: 29.95 seconds
# Total Elapsed time: 167.041254s

# With Duplication Removal
# 114    frame_camera_197.png      (664,645), (845,771)  37.814        34   19700.000        3.000        3.000      19703.000    19700.100         1
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# History visualization started at the real-time:  2024-10-13 18:25:33
# History visualization completed at the real-time:  2024-10-13 18:25:51
# History visualization took a total time of  17.781059980392456 s
# Simulation ended at the real-time:  2024-10-13 18:25:51
# Average response time for processing the entire image dataset: 20.99 seconds
# Total Elapsed time: 51.282227s

