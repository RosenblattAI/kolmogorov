# Generator Scripts

Scripts and Python tools used for satellite situational research.

This package installs the following CLI commands:

* <pre>$ check_duplicates TARGET_DIR</pre>
*  <pre>$ distort_dataset [-h] --side-length SIDE_LENGTH \
                       --aperture-size APERTURE_SIZE \
                       [--fried-param FRIED_PARAM] \
                       [--outer-scale OUTER_SCALE] \
                       [--stencil-length-factor STENCIL_LENGTH_FACTOR] \
                       [--random-seed RANDOM_SEED] \
                       [--interval INTERVAL INTERVAL] [--mean MEAN] \
                       [--std STD] \
                       source_directory output_directory
   </pre>
## Installation

1. Clone the repository to your computer:

    ```$ git clone git@github.com:Rosenblatt-LLC/kolmogorov.git```

1. Install using pip and providing path to the repo directory

    ```$ pip install PATH_TO_PACKAGE```

You will then have the previously listed scripts available to you, and, in a Python language environment, can import the distortion module:
```
from generator_scripts.distortion import apply_atmospheric_distortion

apply_atmospheric_distortion(
    source_directory: str,
    output_directory: str,
    img_side_length: int,
    aperture_size: float,
    fried_param: float,
    outer_scale: int,
    stencil_length_factor: int,
    random_seed: float,
    interval: List[int, int],
    mean: float,
    std: float
)
```

## Script Details

### check_duplicates

<pre>$ check_duplicates TARGET_DIR</pre>

This is a minimal script that checks if there are any images that have the same orientation values between the validation and training image sets for a satellite class. It requires a path to the directory containing both a training and validation folder, and outputs the number of unique samples for each image set.

### distort_dataset

<pre>$ distort_dataset [-h] --side-length SIDE_LENGTH \
                       --aperture-size APERTURE_SIZE \
                       [--fried-param FRIED_PARAM] \
                       [--outer-scale OUTER_SCALE] \
                       [--stencil-length-factor STENCIL_LENGTH_FACTOR] \
                       [--random-seed RANDOM_SEED] \
                       [--interval INTERVAL INTERVAL] [--mean MEAN] \
                       [--std STD] \
                       source_directory output_directory
</pre>

Applies an approximate filter that replicates atmospheric distortion to images in a directory whose filenames match the provided glob pattern, and then stores the results in the target directory. The filepaths excluding the sample directory name are appended to the target directory.

### Arguments:

  `-h`, `--help`            show this help message and exit  

positional arguments:  
  `source_directory`      absolute or relative path to directory containing
                        images  
  `output_directory`      absolute or relative path to where distorted images
                        will be saved

The distortion characteristics can be controlled using the cli args (passed after the sample directory, glob pattern, and target directory):

optional arguments:  
  `--side-length SIDE_LENGTH`, `-l SIDE_LENGTH`
                        the number of pixels along one side of the (square)
                        images  
  `--aperture-size APERTURE_SIZE`, `-a APERTURE_SIZE`
                        size of the telescope aperture in meters (8 is a good
                        value)  
  `--fried-param FRIED_PARAM`, `-f FRIED_PARAM`
                        Fried param in meters used to generate Kolmogorov
                        phase screen (AOTools library)  
  `--outer-scale OUTER_SCALE`, `-x OUTER_SCALE`
                        outer scale in meters used to generate Kolmogorov
                        phase screen (AOTools library)  
  `--stencil-length-factor STENCIL_LENGTH_FACTOR`, `-s STENCIL_LENGTH_FACTOR`
                        stencil length factor used to generate Kolmogorov
                        phase screen (AOTools library), defaults to 4  
  `--random-seed RANDOM_SEED`, `-r RANDOM_SEED`
                        integer value to control randomness  
  `--interval INTERVAL INTERVAL`, `-i INTERVAL INTERVAL`
                        optional parameter to define a range of D/r0 values
                        for a set of Kolmogorov phase screens randomly applied
                        to each image. Expects two integer values, and is
                        inclusive. Do not provide a value for 'Fried param' if
                        'interval' is used.  
  `--mean MEAN`, `-u MEAN`  optional parameter used with 'interval' to control a
                        normal distribution pdf associated with the range of
                        D/r0 values in the defined interval. Should be a value
                        in the interval range. If not provided, pdf will be a
                        uniform distribution.  
  `--std STD`, `-o STD`     optional parameter used with 'interval' and 'mean' to
                        control a normal distribution pdf associated with the
                        range of D/r0 values in the defined interval.
