# built-in
from datetime import datetime


# libs


# own libs


class GlobalLib:

    # config
    PORT = 8081
    UPLOADS_PATH = "./uploads"
    RESULTS_PATH = "./results"
    RESULTS_OBF_PATH = "./results/obf"
    RESULTS_COUNTING_PATH = "./results/counting"

    # response types
    HTML_RESPONSE = "html"
    XML_RESPONSE = "xml"
    OUTPUT_FORMAT_ERROR = "Unrecognized output format"

    # file constants
    FILENAME_FORMAT = "%Y%m%d-%H%M%S%f"
    UPLOAD_FILE_ERROR = "Problems uploading file"

    # extensions
    EXTENSION_ERROR = "File extension not supported"
    ALLOWED_IMG_EXT = [".jpg", ".jpeg", ".png"]
    ALLOWED_VIDEO_EXT = [".mp4", ".wav", ".ogg", ".avi", ".mov"]

    def __init__(self):
        if type(self) == GlobalLib:
            raise ValueError("Cannot create CONSTANTS instance. Abstract Class")

    @staticmethod
    def validate_extension(extension):
        if extension in GlobalLib.ALLOWED_IMG_EXT:
            return "image"

        elif extension in GlobalLib.ALLOWED_VIDEO_EXT:
            return "video"

    @staticmethod
    def generate_file_name():
        return datetime.utcnow().strftime(GlobalLib.FILENAME_FORMAT)

    @staticmethod
    def upload_file(extension, file_body):
        new_filename = GlobalLib.generate_file_name()
        output = open(GlobalLib.UPLOADS_PATH + "/" + new_filename + extension, "wb")
        output.write(file_body)
        return new_filename
