# built-in
import os

# libs
from typing import Optional, Awaitable
import tornado.web
import cv2

# own libs
from services.global_lib import GlobalLib
from obfuscation_system.face_blur_dnn_v2 import FaceAnonymizer


class ObfuscationHandler(tornado.web.RequestHandler):

    FACE_NET = cv2.dnn.readNetFromCaffe("../../models/deploy.prototxt",
                                        "../../models/res10_300x300_ssd_iter_140000.caffemodel")

    PERSON_NET = cv2.dnn.readNetFromCaffe("../../models/MobileNetSSD_deploy.prototxt",
                                          "../../models/MobileNetSSD_deploy.caffemodel")

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        self.render("pages/ObfuscationService.html", content_type=None, result_path=None, mime_type=None)

    def post(self):
        print("obfuscation request")
        # try to read file
        try:
            self.__read_params()

        except KeyError as e:
            print(str(e))
            raise tornado.web.HTTPError(reason=GlobalLib.UPLOAD_FILE_ERROR, status_code=400)

        # get filename and extension
        filename = self.__file['filename']
        extension = os.path.splitext(filename)[1]

        # validate extension
        content_type = GlobalLib.validate_extension(extension)
        if content_type is None:
            raise tornado.web.HTTPError(reason=GlobalLib.EXTENSION_ERROR, status_code=400)

        # upload file
        new_filename = GlobalLib.upload_file(extension, self.__file['body'])

        # apply obfuscation algorithm
        if content_type == "video":
            face_anon = FaceAnonymizer(ObfuscationHandler.FACE_NET, ObfuscationHandler.PERSON_NET)
            result_path = face_anon.face_anonymizer(GlobalLib.UPLOADS_PATH + "/" + new_filename,
                                                    GlobalLib.UPLOADS_PATH + "/" + new_filename + extension)

        else:
            # todo
            print("image obfuscation ... to do")
            result_path = None

        # move file to results dir
        result_extension = "mp4" if content_type == "video" else extension[1:]
        os.rename(result_path, GlobalLib.RESULTS_OBF_PATH + "/" + new_filename + "." + result_extension)
        result_path = GlobalLib.RESULTS_OBF_PATH + "/" + new_filename + "." + result_extension

        # render result
        self.render("pages/" + "ObfuscationService.html", content_type=content_type,
                    result_path=result_path, mime_type=content_type + "/" + result_extension)

    def __read_params(self):
        self.__file = self.request.files["upload_file"][0]

        output_format = self.get_argument("outputFormat", None)
        self.__output_format = output_format if output_format is not None else GlobalLib.HTML_RESPONSE

        # check output format
        if self.__output_format != GlobalLib.HTML_RESPONSE and self.__output_format != GlobalLib.XML_RESPONSE:
            raise tornado.web.HTTPError(reason=GlobalLib.OUTPUT_FORMAT_ERROR, status_code=400)
