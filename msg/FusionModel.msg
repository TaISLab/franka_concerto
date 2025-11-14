# FusionModel.msg

geometry_msgs/Point kp_R_shoulder # vision
geometry_msgs/Point kp_R_elbow # vision
geometry_msgs/Point kp_R_wrist_gripped # gripper
geometry_msgs/Point kp_R_wrist_non_gripped # non gripper

geometry_msgs/Vector3 normal_vector # Vector normal al triángulo del brazo, positivo hacia fuera del cuerpo
geometry_msgs/Vector3 upperarm_vector
geometry_msgs/Vector3 forearm_vector

float64 upperarm_length
float64 forearm_length

geometry_msgs/Vector3 vector_q1

# geometry_msg/Vector3 vector_q2 # añadir cuando haga falta
# geometry_msg/Vector3 vector_q3 # añadir cuando haga falta