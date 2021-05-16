# built-in
import os

# libs
from typing import Optional, Awaitable
import tornado.web
import cv2
import matplotlib.pyplot as plt

# own libs
from services.global_lib import GlobalLib
from tracking_system.tracking_system import TrackingSystem
from tracking_system.dnn_method import DNNMethod


class CountingHandler(tornado.web.RequestHandler):

    DNN_PROTOTXT_DIR = "../../models/MobileNetSSD_deploy.prototxt"
    DNN_MODEL_DIR = "../../models/MobileNetSSD_deploy.caffemodel"

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        self.render("pages/CountingService.html")

    def post(self):
        print("counting request")
        # read params
        self.__read_params()

        # upload file
        new_filename = GlobalLib.upload_file(self.__extension, self.__file['body'])

        # create results folder
        folder_name = GlobalLib.generate_file_name()
        folder_dir = GlobalLib.RESULTS_COUNTING_PATH + "/" + folder_name
        os.makedirs(folder_dir)

        # apply counting system
        person_storage = self.__apply_counting(new_filename, folder_dir)

        # save person frames
        frame_paths = CountingHandler.write_person_frames(folder_dir, person_storage)

        # save some features plots
        entries_plots_path = CountingHandler.save_info_plots(folder_dir, person_storage.entries, "entries")
        exits_plots_path = CountingHandler.save_info_plots(folder_dir, person_storage.exits, "exits")

        self.render("pages/CountingResults.html", video_path=folder_dir + "/debugVideo.mp4", frames_path=frame_paths,
                    entries_plots=entries_plots_path, exits_plots=exits_plots_path, entries=person_storage.entries,
                    exits=person_storage.exits)

    def __read_params(self):
        # counting line
        self.__x_initial = self.get_argument("xInitial", None)
        self.__y_initial = self.get_argument("yInitial", None)
        self.__x_end = self.get_argument("xEnd", None)
        self.__y_end = self.get_argument("yEnd", None)

        # entry direction
        self.__x_entry_dir = self.get_argument("xEntryDir", None)
        self.__y_entry_dir = self.get_argument("yEntryDir", None)

        # tracking alg
        self.__tracking_alg = self.get_argument("trackingAlg", None)
        self.__tracking_thr = self.get_argument("trackingThr", None)

        # file
        self.__file = self.request.files["upload_file"][0]
        self.__filename = self.__file['filename']
        self.__extension = os.path.splitext(self.__filename)[1]

        # output format
        output_format = self.get_argument("outputFormat", None)
        self.__output_format = output_format if output_format is not None else GlobalLib.HTML_RESPONSE

    def __apply_counting(self, filename, folder_dir):
        # create counting system
        net = cv2.dnn.readNetFromCaffe(CountingHandler.DNN_PROTOTXT_DIR,
                                       CountingHandler.DNN_MODEL_DIR)
        threshold = 0.3
        pd_obj = DNNMethod(net, threshold)

        init_point = (int(self.__x_initial), int(self.__y_initial))
        end_point = (int(self.__x_end), int(self.__y_end))
        entry_vector = (int(self.__x_entry_dir), int(self.__y_entry_dir))
        tracking_thr = int(self.__tracking_thr)

        counting_system = TrackingSystem(init_point, end_point, entry_vector, pd_obj,
                                         tracking_thr=tracking_thr, tracking_type=self.__tracking_alg)

        # run counting system
        video_capture = cv2.VideoCapture(GlobalLib.UPLOADS_PATH + "/" + filename + self.__extension)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(folder_dir + "/debugVideo.mp4", fourcc, 10.0, (640, 480))

        while True:
            ret, frame = video_capture.read()

            if frame is None:
                break

            counting_system.process_frame(None, frame)
            CountingHandler.write_debug_frame(counting_system, out, init_point, end_point)

        video_capture.release()
        out.release()

        # get counting results
        person_storage = counting_system.person_storage

        return person_storage

    @staticmethod
    def write_debug_frame(counting_system, video_writer, vline_init, vline_end):
        # get debug frame and last detected persons
        debug_frame = counting_system.debug_frame
        persons = counting_system.last_frame_persons

        # draw centroids
        for person in persons:
            centroid = person.centroid
            cv2.putText(debug_frame, "ID " + str(person.id), centroid, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)

        # draw counting line
        cv2.line(debug_frame, vline_init, vline_end, (255, 0, 0), thickness=2)

        # draw text for number of entries and exits
        entries = counting_system.nr_entries
        exits = counting_system.nr_exists
        cv2.putText(debug_frame, "Entries: " + str(entries), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(debug_frame, "Exits: " + str(exits), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

        # write frame
        video_writer.write(debug_frame)

    @staticmethod
    def write_person_frames(folder_dir, person_storage):
        # list to store frames path
        frames_path = []

        # get person lists
        entries = person_storage.entries
        exits = person_storage.exits
        all_persons = entries + exits

        # write frames
        for i in range(len(all_persons)):
            path = folder_dir + "/frame" + str(i) + ".jpg"
            frames_path.append(path)
            cv2.imwrite(path, all_persons[i].person_frame)

        return frames_path

    @staticmethod
    def save_info_plots(folder_dir, person_list, direction):
        plots_path = []

        for person in person_list:
            # hue histogram
            plt.figure(direction + " Hue Histograms")
            plt.xlabel("hue")
            plt.ylabel("probability")
            h_hist = person.hue_hist
            plt.plot(h_hist)

            # saturation histogram
            plt.figure(direction + " Saturation Histograms")
            plt.xlabel("saturation")
            plt.ylabel("probability")
            s_hist = person.saturation_hist
            plt.plot(s_hist)

            # edge descriptor
            plt.figure(direction + " Edge Descriptors")
            plt.xlabel("edge")
            plt.ylabel("decision index")
            edge_descr = person.edge_hist
            plt.plot(edge_descr)

            # color descriptor
            plt.figure(direction + " Color Descriptors")
            plt.xlabel("color")
            plt.ylabel("decision index")
            color_descr = person.cs_hist
            plt.plot(color_descr)

        # area
        plt.figure(direction + " Areas")
        plt.xlabel("person")
        plt.ylabel("area")
        areas = [person.area for person in person_list]
        plt.plot(areas)

        # save figs
        CountingHandler.save_fig(direction + " Areas", folder_dir + "/" + direction + "Areas.png", plots_path)
        CountingHandler.save_fig(direction + " Hue Histograms", folder_dir + "/" + direction + "Hue.png",
                                 plots_path)
        CountingHandler.save_fig(direction + " Saturation Histograms", folder_dir + "/" + direction + "Saturation.png",
                                 plots_path)
        CountingHandler.save_fig(direction + " Edge Descriptors", folder_dir + "/" + direction + "Edge.png",
                                 plots_path)
        CountingHandler.save_fig(direction + " Color Descriptors", folder_dir + "/" + direction + "Color.png",
                                 plots_path)

        return plots_path

    @staticmethod
    def save_fig(fig_name, filename, path_list):
        plt.figure(fig_name)
        plt.savefig(filename)
        path_list.append(filename)
