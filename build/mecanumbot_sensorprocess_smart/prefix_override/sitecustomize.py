import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ubuntu/mecanumbot_ws/src/mecanumbot_sensorprocess_smart/install/mecanumbot_sensorprocess_smart'
