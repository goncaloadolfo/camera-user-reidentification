import os
import time

if os.path.exists("stop"):
  os.remove("stop")
  
os.system('cd /home/pi/Desktop/Projeto_Versions/Projeto_v18/PythonAlgs/src/tests/movement_test')
os.system('python3.5 movement_detector.py')


