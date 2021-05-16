import os
if os.path.exists("stop"):
  os.remove("stop")
  
os.system('cd /home/pi/Desktop/Projeto_Versions/Projeto_v19_Sensor_Extract/PythonAlgs/src/main_app')
os.system('python3.5 process_starter.py')