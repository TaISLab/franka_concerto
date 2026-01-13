#!/bin/bash
BAG_PATH=$(rosparam get /exp_optitrack_25/rosbag_path)
if [ "$BAG_PATH" != "" ]; then
    rosbag play -r 1 "$BAG_PATH" --clock --pause
    # rqt_bag "$BAG_PATH"
else
    echo "No se encontró el parámetro /exp_optitrack_25/rosbag_path"
fi