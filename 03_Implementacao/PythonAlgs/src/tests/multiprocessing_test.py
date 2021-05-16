import time
from threading import Thread, Event


class MainAppSimulator(Thread):

    def __init__(self):
        super(MainAppSimulator, self).__init__()
        self.event_flag = Event()

    def run(self):
        while not self.event_flag.is_set():
            print("tracking...")

        self.__apply_algorithms()

    def __apply_algorithms(self):
        print("running algorithm")


p = MainAppSimulator()
p.start()

time.sleep(10)
p.event_flag.set()

