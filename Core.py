from skimage import data, io
from matplotlib import pyplot as plt
import pyrealsense2 as rs
import numpy as np
import cv2

import time

class Core:
    def __init__(self, model):
        self.model = model

    def runImage(self, path_to_image):
        image = io.imread(path_to_image)
        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)
        self.model.detect(image, ax)
        plt.show()

    def runVideo(self, path_to_video):
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, path_to_video)
        colorizer = rs.colorizer()
        pipeline.start(config)
        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)
        #_, ax2 = plt.subplots(2, figsize=figsize)
        plt.ion()
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            depth_color_frame = colorizer.colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            self.model.detect(color_image, ax)
            plt.pause(0.1)


    def runLive(self):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

        # Start streaming

        ctx = rs.context()
        profile = ctx.devices[0]
        sensor_dep = profile.first_depth_sensor()
        sensor_color = profile.first_color_sensor()

        print("Exposure settings -> need to adjust to environment")
        sensor_dep.set_option(rs.option.enable_auto_exposure, 0)
        sensor_color.set_option(rs.option.enable_auto_exposure, 0)
        sensor_dep.set_option(rs.option.exposure, 800)
        sensor_color.set_option(rs.option.exposure, 70)

        print("exposure depth: %d" % sensor_dep.get_option(rs.option.exposure))
        print("exposure color: %d" % sensor_color.get_option(rs.option.exposure))
        depth_scale = sensor_dep.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        sensor_dep.set_option(rs.option.gain, 16)
        sensor_color.set_option(rs.option.brightness, 0)
        sensor_color.set_option(rs.option.contrast, 50)
        sensor_color.set_option(rs.option.hue, 0)
        sensor_color.set_option(rs.option.brightness, 0)
        sensor_color.set_option(rs.option.saturation, 64)
        sensor_color.set_option(rs.option.sharpness, 50)

        profile = pipeline.start(config)
        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)

        while True:
            frames = pipeline.wait_for_frames()

            raw_depth_frame = frames.get_depth_frame()
            raw_color_frame = frames.get_color_frame()

            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            depth_frame = spatial.process(raw_depth_frame)
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal.set_option(rs.option.filter_smooth_delta, 20)
            depth_frame = temporal.process(depth_frame)
            hole_filling = rs.hole_filling_filter()
            depth_frame = hole_filling.process(depth_frame)

            colorizer = rs.colorizer()
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_color_frame.get_data())
            color_image = np.asanyarray(raw_color_frame.get_data())

            self.model.detect(color_image, ax)
            plt.pause(0.1)