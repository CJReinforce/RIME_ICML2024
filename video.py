import imageio
import os
import numpy as np
import sys

import utils

class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.duration = int(1000 * 1 / self.fps)
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array',
                               height=self.height,
                               width=self.width,
                               camera_id=self.camera_id)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, duration=self.duration, loop=0)

class MetaWorldVideoRecorder(object):
    def __init__(self, root_dir, height=128, width=128, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            """frame = env.env.sim.render(
                *(env.env._rgb_array_res),
                mode='offscreen',
                camera_name='corner'
            )[:, :, ::-1]"""
            frame = env.render(mode='rgb_array')
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            
class ShortVideoRecorder(object):
    def __init__(self, height=256, width=256, camera_id=0, fps=30):
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []

    def record(self, env):
        frame = env.render(mode='rgb_array',
                           height=self.height,
                           width=self.width,
                           camera_id=self.camera_id)
        self.frames.append(frame)
    
    def get_record(self):
        out = self.frames
        self.frames = []
        return out

    def save(self, path):
        imageio.mimsave(path, self.frames, fps=self.fps)
