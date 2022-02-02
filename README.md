# DAC4RL
DAC4RL track of the DAC4AutoML competition at AutoML-Conf.

## Getting Started
```
# If using SSH keys:
git clone git@github.com:automl-private/dac4automlcomp.git
cd dac4automlcomp
pip install -e .
git clone git@github.com:automl-private/DAC4RL.git
cd DAC4RL
```

## Sample Submission
The directory `sample_submission` contains a sample submission with [a sample code file](sample_submission/submission.py) and an optional [`requirements.txt`](sample_submission/requirements.txt) which can contain additional Python packages to be installed *during* an evaluation. (#TODO installation time counts towards run time?)

The Bash script [`prepare_upload.sh`](https://www.github.com:automl-private/dac4automlcomp/) may be used to package a submission directory into a `.zip` file ready for submission to the competition servers.

```
bash prepare_upload.sh sample_submission
```

To create a submission:
  * Create a Python file that defines a class that implements the DAC policy. It should subclass [`AbstractPolicy`](#TODO link)
  * The same Python file should import `AbstractPolicy` and `run_experiment` as done in the sample submission.
  * The same Python file should be callable as a Python script from the command line. For this, please call `run_experiment` with an object of the policy that you defined above, as done in the sample submission.
  * Optionally, create a `requirements.txt` with additional Python packages that you would like to be installed
  * Place the above file(s) in the same directory and run `prepare_upload.sh` with the name of this directory as an argument.


## Evaluating Submissions Locally
[`evaluate_submission.sh`](evaluate_submission.sh) can be used to evaluate submissions in the same way that they would be evaluated on the competition serves. For example, you can execute the following command to evaluate the sample submission in [`sample_submission`](sample_submission):

```
bash evaluate_submission.sh -d sample_submission/ -f submission.py
```

## Singularity Container
To run your experiments in the same runtime environment as the competition servers they will be evaluated on, we provide a [Singularity container](https://sylabs.io/guides/3.5/user-guide/introduction.html). Please see [the Singularity container definition file](dac4rl.def) to see what additional packages will be available in the runtime environment.

The container may either be downloaded from the Container Library:
```
#TODO Add the correct link after uploading built container.
singularity pull IMG.sif library://raghuspacerajan/REPO/IMG:TAG
```

Or, it may be built using [our container definition file](dac4rl.def) as follows:
```
# CURRENTLY MIGHT NOT WORK
sudo singularity build dac4rl.sif dac4rl.def
```

Commands may be executed by entering a shell for the container as follows:
```
singularity shell dac4rl.sif
```

It is not a requirement to use the Singularity container (#TODO ?), but it will be easier for us to debug any issues that arise when you try running experiments for the competition.
