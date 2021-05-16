import RPi.GPIO as GPIO
import time

class EntryExitSensor:

    def __init__(self, entrance_pin, exit_pin):
        self.__entrance_pin = entrance_pin
        self.__exit_pin = exit_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.__entrance_pin, GPIO.IN)
        # GPIO.setup(self.__entrance_pin, GPIO.IN)
        GPIO.setup(self.__exit_pin, GPIO.IN)
#        GPIO.setup(20, GPIO.OUT)

    def check_sensor(self):
        '''Check sensor to determine
         if person is entering or leaving'''
        if GPIO.input(self.__entrance_pin):
            while GPIO.input(self.__entrance_pin):
                continue
##            GPIO.output(20,GPIO.HIGH)
##            time.sleep(0.2)
##            GPIO.output(20,GPIO.LOW)
            print("Entrance!")
            return True
        if GPIO.input(self.__exit_pin):
            while GPIO.input(self.__exit_pin):
                continue
#            GPIO.output(20,GPIO.HIGH)
#            time.sleep(0.2)
#            GPIO.output(20,GPIO.LOW)
            print("Exit!")
            return False

        return None

    def cleanGPIO(self):
        GPIO.cleanup()


if __name__ == '__main__':
    door_sensor = EntryExitSensor(21, 20)
    try:
        while True:
            door_sensor.check_sensor()
    except KeyboardInterrupt:
        print("Shutdown entrance/exit detection")
        door_sensor.cleanGPIO()

