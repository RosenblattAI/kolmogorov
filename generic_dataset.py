#! /usr/bin/env python3
'''

Split: 75 less, 15 not, 10 more

existing dirs
less_distorted
    training
        classes
    validation
        classes

more_distorted
    training
        classes
    validation
        classes

not_distorted
    training
        classes
    validation
        classes

result:
generic
    training
        classes
            75 less
            15 not
            10 more
    validation
        classes
            75 less
            15 not
            10 more
'''

import os
import re

if __name__ == '__main__':
    classes = {
        'not_distorted': 0.15,
        'less_distorted': 0.75,
        'more_distorted': 0.10
    }

    for distort_class, ratio in classes.items():
        # img_dir is where the image file dirs were located on my system
        img_dir = 'ssa-data/{}/'.format(distort_class)
        target_dir = 'generic-data/'

        TRAINING = lambda img_dir: img_dir + 'training'
        VALIDATION = lambda img_dir: img_dir + 'validation'

        for SUBDIR in [TRAINING, VALIDATION]:
            for sat in os.listdir(SUBDIR(img_dir)):
                sat_dir = SUBDIR(img_dir)+'/'+sat
                samples = [name for name in os.listdir(sat_dir)]

                for i in range(round(len(samples) * ratio)):
                    copy_name = '{}.{}.png'.format(samples[i][:-4],distort_class)
                    copy_target = '{}/{}/{}'.format(SUBDIR(target_dir), sat, copy_name)
                    original = sat_dir + '/' + samples[i]

                    # Because the imgage filenames contained parantheses at the time, they need to be escaped
                    copy_target = re.sub('\(', '\\\(', copy_target)
                    copy_target = re.sub('\)', '\\\)', copy_target)
                    original = re.sub('\(', '\\\(', original)
                    original = re.sub('\)', '\\\)', original)

                    os.makedirs(os.path.dirname(copy_target), exist_ok=True)
                    os.system('cp {} {}'.format(original, copy_target))
