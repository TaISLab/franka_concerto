# franka_concerto

## fr3_move_server
Se encarga de cambiar entre tipos de controladores. Se le manda la pose objetivo como .action.
Para mover el franka, se dispone del nodo franka_action_server.py encargado de cambiar el controlador y mandar la pose/velocidad deseada: 
0 Homing. Controlador de velocidad cartesiana TODO: cambiar nombre
1 Impedancia cartesiana
2 Joint pose control effort transmision

Algunos comandos importantes:


1. Ir a homing con controlador de posición articular: La position, orientation no sirve pq la coge del código. Lo importante es el task_type:

```bash
$ rostopic pub /fr3_motion_server/goal franka_concerto/MoveFR3ActionGoal "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
goal_id:
  stamp: {secs: 0, nsecs: 0}
  id: ''
goal:
  task_type: 2
  target_pose:
    header:
      seq: 0
      stamp: {secs: 0, nsecs: 0}
      frame_id: ''
    pose:
      position: {x: 0.0, y: 0.0, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  max_velocity: 0.0"

```
2. Ir a una posición con el controlador de velocidad cartesiana. Dispone de un P para convertir el error de posición en velocidad. Kp=0.1. Además dispone de un filtro exponencial para evitar discontinuidades en la velocidad. Aún asi aparecen en la aceleración. considerar rate_limiter de libfranka.
```bash
rostopic pub /fr3_motion_server/goal franka_concerto/MoveFR3ActionGoal "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
goal_id:
  stamp: {secs: 0, nsecs: 0}
  id: ''
goal:
  task_type: 0
  target_pose:
    header:
      seq: 0
      stamp: {secs: 0, nsecs: 0}
      frame_id: ''
    pose:
      position: {x: 0.5, y: 0.0, z: 0.3}
      orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}
  max_velocity: 0.1"
```

3. Comandos de impedancia cartesiana: 

```bash
rostopic pub /fr3_motion_server/goal franka_concerto/MoveFR3ActionGoal "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
goal_id:
  stamp: {secs: 0, nsecs: 0}
  id: ''
goal:
  task_type: 1
  target_pose:
    header:
      seq: 0
      stamp: {secs: 0, nsecs: 0}
      frame_id: ''
    pose:
      position: {x: 0.5, y: 0.0, z: 0.3}
      orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}
  max_velocity: 0.1"
```


# Behaviour tree
La nueva implementación consiste en un BT.
Importante: franka_control, gripper_4f, conv deben estar ejecutandose

```bash
$ roslaunch skeleton_3d process_skeletons.launch use_rviz:=True
$ roslaunch gripper_4f gripper_controller_franka_buttons.launch
$ roslaunch franka_control franka_control.launch
$ roslaunch franka_concerto bt.launch
$ roslaunch gripper_4f conv_lstm_inferir.launch
```

Para depurar el arbol: 
```bash
py-trees-blackboard-watcher --snapshot

```

## Funding

![MICIU, cofinanciado por la Unión Europea, Agencia Estatal de Investigación](docs/funding-logo.jpg)

MICIU / AEI / FEDER

This work is part of project PID2021-127221OB-I00 (CONCERTO — Control Colaborativo para Interacción física Empática entre RoboT y humanO), funded by MICIU/AEI/10.13039/501100011033/FEDER, UE.