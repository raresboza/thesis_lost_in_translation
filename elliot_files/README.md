# ELLIOT Files

This directory contains the code that should be copied as is into a cloned repository of the open souce project [ELLIOT](https://github.com/sisinflab/elliot). The version of this tool that this study executed its experiment on was **v0.3.1**.

The addition of the changed files is fairly straight forward, as we have structured them in the same manner as in the original repository. The changes to the metrics are done to save their scores per user level to foster statistical significance testing. The `Slim` changes are a fix for the version of ELLIOT that we used, as it crashes in the aforementioned iteration of the project.

To run the desired experiments, we build the environment for this project as described in the source repository. The `configs` directory contains all our recommendation runs formulating the pipeline of the experiment. `experiment.py` contains a script that can be used to start these experiments as desired.

# Notes
Absolute paths were used for saving the results in some instances, as the code was run on different devices and different operating systems. Please change them according to your own file structure.