# built-in
import os
import sys
sys.path.append("..")

# libs
import tornado.ioloop
import tornado.web

# own libs
from services.obfuscation_handler import ObfuscationHandler
from services.counting_handler import CountingHandler
from services.global_lib import GlobalLib


def create_server_dirs():
    create_dir(GlobalLib.UPLOADS_PATH)
    create_dir(GlobalLib.RESULTS_PATH)
    create_dir(GlobalLib.RESULTS_OBF_PATH)
    create_dir(GlobalLib.RESULTS_COUNTING_PATH)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def run_server():
    # file management system
    create_server_dirs()

    # create an app
    handlers = [("/obfuscation", ObfuscationHandler),
                ("/counting", CountingHandler),

                ("/images/(.*)", tornado.web.StaticFileHandler, {"path": "./images"},),
                ("/css/(.*)", tornado.web.StaticFileHandler, {"path": "./css"},),
                ("/js/(.*)", tornado.web.StaticFileHandler, {"path": "./js"},),

                (GlobalLib.RESULTS_PATH[1:] + "/(.*)", tornado.web.StaticFileHandler,
                 {"path": GlobalLib.RESULTS_PATH},)]

    app = tornado.web.Application(handlers)

    # start listening
    app.listen(GlobalLib.PORT)
    print("listening on port " + str(GlobalLib.PORT))
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    run_server()
