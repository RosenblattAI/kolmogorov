# Generator Scripts

Scripts and Python tools used for satellite situational research.

This package installs the following CLI commands:

* <pre>$ check_duplicates TARGET_DIR</pre>
*  <pre>$ distort_dataset DIRECTORY_PATH \
                   FILE_GLOB_STRING \
                   OUTPUT_DIRECTORY_PATH \
                   [APERTURE_SIZE:float \
                   FRIED_PARAM:float \
                   OUTER_SCALE:float \
                   RANDOM_SEED:{none|int} \
                   STENCIL_LENGTH_FACTOR:int]
   </pre>
* <pre>$ generic_dataset SAMPLE_DIRECTORY \
                  OUTPUT_DIRECTORY \
                  [SAMPLE_CLASS=RATIO:float ...]</pre>

## Installation

1. Clone the repository to your computer:

    ```$ git clone git@github.com:Rosenblatt-LLC/kolmogorov.git```

1. Install using pip and providing path to the repo directory

    ```$ pip install PATH_TO_PACKAGE```

You will then have the previously listed scripts available to you, and, in a Python language environment, can import the distortion module:
```
from generator_scripts.distortion import atmospheric_distort_directory

atmospheric_distort_directory(
    directory_path,
    file_glob_matcher,
    output_directory,
    aperture_size,
    fried_param,
    outer_scale,
    random_seed,
    stencil_length_factor
)
```

## Script Details

### check_duplicates

<pre>$ check_duplicates TARGET_DIR</pre>

This is a minimal script that checks if there are any images that have the same orientation values between the validation and training image sets for a satellite class. It requires a path to the directory containing both a training and validation folder, and outputs the number of unique samples for each image set.

### distort_dataset

<pre>$ distort_dataset DIRECTORY_PATH \
                   FILE_GLOB_STRING \
                   OUTPUT_DIRECTORY_PATH \
                   [APERTURE_SIZE:float \
                   FRIED_PARAM:float \
                   OUTER_SCALE:float \
                   RANDOM_SEED:{none|int} \
                   STENCIL_LENGTH_FACTOR:int]
</pre>

Applies an approximate filter that replicates atmospheric distortion to images in a directory whose filenames match the provided glob pattern, and then stores the results in the target directory. The filepaths excluding the sample directory name are appended to the target directory.

The distortion characteristics can be controlled using the cli args (passed after the sample directory, glob pattern, and target directory):

1. `aperture_size` (float): The size of the telescope aperture in meters.
2. `fried_param` (float): Size of atmospheric coherence length (Fried param) in meters
3. `outer_scale` (float): Described in the [AOTools Kolmogorov Phase Screen docs](https://aotools.readthedocs.io/en/v1.0.1/turbulence.html?highlight=L0#aotools.turbulence.infinitephasescreen.PhaseScreenKolmogorov)
4. `random_seed` (int): Control the randomness (pass "none" to ignore)
5. `stencil_length_factor` (int): Described in the [AOTools Kolmogorov Phase Screen docs](https://aotools.readthedocs.io/en/v1.0.1/turbulence.html?highlight=stencil_length_factor#aotools.turbulence.infinitephasescreen.PhaseScreenKolmogorov)

### generic_dataset

<pre>$ generic_dataset SAMPLE_DIRECTORY \
                  OUTPUT_DIRECTORY \
                  [SAMPLE_CLASS=RATIO:float ...]
</pre>

Given a sample directory that contains distinct datasets of satellite classes/images, create a new dataset that contains a specified portion of samples from each distinct dataset. Note that the ratio is relative to the number of samples belonging to a distinct dataset, and not relative to the number of samples in the final dataset. You can pass any number of distinct dataset dirs along with the required ratio using the `SAMPLE_CLASS` arg described above.

Defaults to:
* not_distorted: 0.15
* less_distorted: 0.75
* more_distorted: 0.10

The expected folder structure would be as follows:
```
sample_directory
|___ not_distorted
|    |___ training
|    |    |___ satellite_class
|    |    |___ another_satellite_class
|    |
|    |___ validation
|         |___ satellite_class
|         |___ another_satellite_class
|
|___ less_distorted
|    |___ ...
|
|___ more_distorted
     |___ ...
```

The output structure is as follows:
```
output_directory
|___ training
|    |___ satellite_class
|    |___ another_satellite_class
|
|___ validation
     |___ satellite_class
     |___ another_satellite_class
```
