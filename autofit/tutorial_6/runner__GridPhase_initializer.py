import os
import sys

import numpy as np


centre_0_min = -2.0
centre_0_max = 2.0
centre_1_min = -2.0
centre_1_max = 2.0
n = 5

if __name__ == "__main__":


    centre_0 = np.linspace(centre_0_min, centre_0_max, n)
    centre_1 = np.linspace(centre_1_min, centre_1_max, n)

    filename = "multiconf.conf"
    if os.path.isfile(filename):
        os.system(
            "rm {}".format(filename)
        )
    out = open(filename, "w")

    directory = os.path.dirname(
        os.path.realpath(__file__)
    )

    python_script_filename = "runner__GridPhase__lens__sie_and_subhalo__source__ellipticalsersic_and_kinematics__data__lens__sie_and_subhalo__source__ellipticalsersic_and_kinematics"

    i = 0
    for centre_0_lower_limit, centre_0_upper_limit in zip(centre_0[:-1], centre_0[1:]):
        for centre_1_lower_limit, centre_1_upper_limit in zip(centre_1[:-1], centre_1[1:]):
            out.write(
                "{} python {}/{}.py {} {} {} {}".format(
                    i,
                    directory,
                    python_script_filename,
                    centre_0_lower_limit,
                    centre_0_upper_limit,
                    centre_1_lower_limit,
                    centre_1_upper_limit
                )
            )
            out.write("\n")

            i += 1
