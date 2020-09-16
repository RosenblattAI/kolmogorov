#! /usr/bin/env python3

import cv2
import logging
import os
import pathlib
import random
try:
    import cupy as np
    from cupy.fft import ifft2, fft2
    using_cupy = True
except Exception as e:
    print(f'Unable to import cupy, defaulting to numpy. {str(e)}')
    import numpy as np
    from numpy.fft import ifft2, fft2
    using_cupy = False
import matplotlib.pyplot as plt
from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov
from scipy.stats import norm
from typing import List


class PhaseScreen(PhaseScreenKolmogorov):
    """Wraps Kolmogorov phase screen and exposes values of interest
    """
    def __init__(self, height: int, aperture_size: float, fried_param: float, outer_scale: int, random_seed: int, stencil_length_factor: int):
        self.height = height
        self.aperture_size = aperture_size
        self.pxl_scale = aperture_size / height
        self.fried_param = fried_param
        self.outer_scale = outer_scale
        self.random_seed = random_seed
        self.stencil_length_factor = stencil_length_factor

        super().__init__(
            self.height,
            self.pxl_scale,
            self.fried_param,
            self.outer_scale,
            self.random_seed,
            self.stencil_length_factor
        )

    def __update_screen(self):
        for i in range(random.randint(1, self.height)):
            self.add_row()

    def __mul__(self, other):
        self.__update_screen()
        return self.scrn * other

    def __rmul__(self, other):
        self.__update_screen()
        return other * self.scrn

    def __getattr__(self, name):
        return getattr(self.scrn, name)


class PhaseScreenContainer:
    """Contains the set of phasescreens queried by DistortionController
    """
    def __init__(self,
                 size: int,
                 aperture_size: float,
                 fried_param: float=None,
                 outer_scale: int=None,
                 stencil_length_factor: int=None,
                 random_seed: float=None,
                 interval: List[int]=None,
                 mean: float=None,
                 std: float=None):
        assert size is not None, 'pixel length of image side must be provided'
        assert aperture_size is not None, 'aperture size must be provided'

        if interval is not None:
            assert len(interval) == 2 and \
                    isinstance(interval[0], int) and \
                    isinstance(interval[1], int), \
                    'Interval argument must be a list of 2 integers'
            assert fried_param is None, 'Interval argument is provided a value, fried-param must be None'
        else:
            assert fried_param is not None, 'Interval argument is None, both aperture-size and fried-param must have a value'

        outer_scale = outer_scale or 100
        stencil_length_factor = stencil_length_factor or 4
        std = std or 1.0

        self.phase_screens = self.__generate_phase_screens(
            size=size,
            aperture_size=aperture_size,
            fried_param=fried_param,
            outer_scale=outer_scale,
            stencil_length_factor=stencil_length_factor,
            random_seed=random_seed,
            interval=interval
        )

        self.weights = None
        if len(self.phase_screens) > 1 and mean is not None:
            self.weights = self.__calculate_weights(interval, mean, std)

    def __generate_phase_screens(
            self,
            size: int,
            aperture_size: float,
            fried_param: float,
            outer_scale: int,
            stencil_length_factor: int,
            random_seed: float,
            interval: List[int]) -> List[PhaseScreen]:
        if not interval:
            return [
                PhaseScreen(
                    height=size,
                    aperture_size=aperture_size,
                    fried_param=fried_param,
                    outer_scale=outer_scale,
                    random_seed=random_seed,
                    stencil_length_factor=stencil_length_factor
                )
            ]

        phase_screens = []
        for dr0 in range(interval[0], interval[1] + 1):
            phase_screens.append(PhaseScreen(
                height=size,
                aperture_size=aperture_size,
                fried_param=aperture_size / dr0,
                outer_scale=outer_scale,
                random_seed=random_seed,
                stencil_length_factor=stencil_length_factor))

        return phase_screens

    def __calculate_weights(self, interval: List[int], mean: float, std: float=1.0) -> List[int]:
        normal_dist = norm(mean, std)
        return [normal_dist.pdf(dr0) for dr0 in range(interval[0], interval[1] + 1)]

    @property
    def phase_screen(self) -> PhaseScreen:
        phase_screen = random.choices(self.phase_screens, self.weights)[0]
        return phase_screen


class DistortionController:
    """Exposes API to control application of distortion to image files
    API is:
    * __init__: stores phase screen container
    * apply_distortion: initiates distortion process
    """
    def __init__(self, phase_screen_container: PhaseScreenContainer):
        self.phase_screen_container = phase_screen_container

    def __create_circular_mask(self, h: int, w: int, center: int=None, radius: int=None) -> np.ndarray:
        # https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

        if center is None: # use the middle of the image
            center = (w//2, h//2)
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

        mask = dist_from_center <= radius
        return mask.astype(int)

    def __circular_aperture(self, img: np.ndarray) -> np.ndarray:
        orig_dim = img.shape

        pri_h = orig_dim[0] // 2 # 224 -> 112
        pri_w = orig_dim[1] // 2 # 224 -> 112

        sec_h = int(orig_dim[0] * 0.09375) # 224 -> 21
        sec_w = int(orig_dim[1] * 0.09375) # 224 -> 21
        sup_start = (pri_h-sec_h) // 2 # start boundary to super impose pupil2 mask into mask of size pupil1
        sup_end = sup_start + sec_h # end boundary to super impose pupil2 on pupil1

        pupil1 = self.__create_circular_mask(pri_h, pri_w, radius=pri_h//2)
        pupil2_mask = self.__create_circular_mask(sec_h, sec_w, radius=sec_h//2)
        pupil2 = np.zeros_like(pupil1)
        pupil2[sup_start:sup_end, sup_start:sup_end] = pupil2_mask

        pupil_mask = pupil1 - pupil2
        pupil_mask = np.pad(pupil_mask, ((orig_dim[0] - pri_h)// 2, ), 'constant')
        pupil_mask = pupil_mask * orig_dim[0] / np.sqrt(np.sum(pupil_mask ** 2))

        return pupil_mask

    def __atmospheric_distort(self, img: np.ndarray) -> np.ndarray:
        pupil_mask = self.__circular_aperture(img)
        phase_screen = self.phase_screen_container.phase_screen

        aperture_size = phase_screen.aperture_size
        fried_param = phase_screen.fried_param
        outer_scale = phase_screen.outer_scale
        stencil_length_factor = phase_screen.stencil_length_factor

        if using_cupy:
            phase_screen = np.asarray(phase_screen, dtype=np.float32)

        # this sets each satellite pixel to max brightness. Remove this line if it interferes with dynamic brightness range
        img[img > 0] = 255
        a = ifft2(pupil_mask * np.exp(1j * phase_screen))
        h = abs(abs(a) ** 2)
        img = ifft2(fft2(h) * fft2(img[:,:,1])).real

        img /= np.max(img)
        img *= 255

        return np.repeat(img[:,:,np.newaxis], 3, axis=2), aperture_size, fried_param, outer_scale, stencil_length_factor

    def __atmospheric_distort_image_file(self, filepath: pathlib.Path, output_directory: str) -> pathlib.Path:
        def make_distorted_image_filename(filename, aperture_size, fried_param, outer_scale, stencil_length_factor):
            return '-'.join([
                filename,
                f'aperture_size={aperture_size}',
                f'fried_param={fried_param}',
                f'outer_scale={outer_scale}',
                f'stencil_length_factor={stencil_length_factor}'
            ])

        img = cv2.imread(str(filepath))
        if using_cupy:
            img = np.asarray(img)
        img, aperture_size, fried_param, outer_scale, stencil_length_factor = self.__atmospheric_distort(img)
        if using_cupy:
            img = np.asnumpy(img)

        filename, ext = os.path.splitext(filepath.name)
        new_filename = make_distorted_image_filename(filename, aperture_size, fried_param, outer_scale, stencil_length_factor)
        new_filepath = output_directory/str(filepath.name).replace(str(filename), new_filename)
        # print(new_filepath)

        new_filepath.parents[0].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(new_filepath), img)
        return new_filepath

    def apply_distortion(self, source_directory: str, output_directory: str):
        assert isinstance(source_directory, str), 'source-directory must be a valid path'
        assert isinstance(output_directory, str), 'output-directory must be a valid path'

        source_directory = pathlib.Path(source_directory)
        output_directory = pathlib.Path(output_directory)

        for root, _, files in os.walk(source_directory):
            for path_str in files:
                try:
                    path = pathlib.Path(os.path.join(root, path_str))
                    trimmed_path = pathlib.Path(*path.parts[len(source_directory.parts):]).parent
                    self.__atmospheric_distort_image_file(
                        path,
                        output_directory/trimmed_path
                    )
                except Exception as e:
                    if any(path_str.endswith(ext) for ext in ['.png', '.jpg']):
                        raise e
                    else:
                        print(f'File at {path_str} is not an image -- {e}')


def apply_atmospheric_distortion(
        source_directory: str,
        output_directory: str,
        img_side_length: int,
        aperture_size: float,
        fried_param: float=None,
        outer_scale: int=None,
        stencil_length_factor: int=None,
        random_seed: float=None,
        interval: List[int]=None,
        mean: float=None,
        std: float=None):
    PSC = PhaseScreenContainer(img_side_length, aperture_size, fried_param, outer_scale, stencil_length_factor, random_seed, interval, mean, std)
    controller = DistortionController(PSC)
    controller.apply_distortion(source_directory, output_directory)
