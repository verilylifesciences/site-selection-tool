# Development and deployment instructions for metis

* [Developer setup](#developer-setup)
* [How to diff notebooks](#how-to-diff-notebooks)
* [How to transfer notebooks to/from Terra](#how-to-transfer-notebooks-tofrom-terra)
* [Terra Docker image](#terra-docker-image)
  * [Development and testing](#development-and-testing)
  * [Deployment](#deployment)

# Developer setup
You only need to do these steps once.

1. Create a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) for Python 3.
```
# Tip: run this outside of your git clone, such as under ${HOME}/my-venvs
cd path/to/where/I/keep/my/venvs
python3 -m venv metis-env
```
2. Activate the [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).
```
source metis-env/bin/activate
```
3. Install [pre-commit](https://pre-commit.com/), a framework for managing and maintaining multi-language pre-commit hooks, [nbdime](https://nbdime.readthedocs.io/en/latest/) a tool for diffing and merging of Jupyter notebooks, and [nbstripout](https://github.com/kynan/nbstripout), a tool to strip output from Jupyter notebooks.
```
pip3 install pre-commit nbdime nbstripout
```
4. Enable nbdime, nbstripout, and install the git hook scripts.
```
cd path/to/my/clone/of/metis
pre-commit install
nbdime config-git --enable
nbstripout --install
```

See [.pre-commit-config.yaml](../.pre-commit-config.yaml) for the currently configured pre-commit hooks.

# How to diff notebooks
See [nbdime](https://nbdime.readthedocs.io/en/latest/) for more detail, including the merge tool.
```
# Terminal diff (git diff will also emit the same diff)
nbdiff notebooks/test.ipynb

# Use Chrome to view a diff of a notebook with uncommitted changes.
nbdiff-web notebooks/test.ipynb
```

# How to transfer notebooks to/from Terra

When you use [Terra](https://app.terra.bio/), notebooks are stored in the `notebooks` folder of your workspace bucket. Terra does not natively support source control integration, but we can do it manually.

By convention, for the Metis project the "source of truth" for the notebooks is source control, not Terra storage. Author new notebooks in your sandbox Terra workspace, but be sure to check them in as you reach good stopping points.

Option 1, **recommended**: The easy way to transfer notebooks is to use the Terra UI to manually upload and download notebooks to your git clone on your local machine.

Option 2: Clone this repo underneath the [detachable persistent disk](https://support.terra.bio/hc/en-us/articles/360047318551) attached to the Terra VM.

# Terra Docker image

## Development and testing

If you want to edit the code in the Docker container for testing purposes (so that its faster to test than rebuilding & redeploying a new image), do the following:

1. open the Jupyter console by right clicking on the Jupyter logo to open it in a new tab ![jupyter logo](https://jupyter.org/assets/nav_logo.svg)
2. in the terminal, move the code to your persistent disk and create a symbolic link
```
mv ${HOME}/metis_pkg ${HOME}/notebooks/
ln -s ${HOME}/notebooks/metis_pkg ${HOME}/
```
Now you can view and edit code files via the Jupyter console!

## Deployment

When you are ready to get the updated Docker image out to collaborators:

### Build and push

You can use a command similar to the following to build and deploy the [Terra](https://app.terra.bio/) Docker image to Google Container Registry. Be sure to replace the project id `verily-metis-data` in the command below with the id of your project.
```
# Do this in the source of truth (your Git clone), not Terra.
# Run this in the directory where the Dockerfile resides.
gcloud --billing-project verily-metis-data \
  --project verily-metis-data \
  builds submit \
  --timeout 20m \
  --tag gcr.io/verily-metis-data/metis_terra:`date +"%Y%m%d_%H%M%S"` .
```
The image is now ready for use on your Terra VM!

### Let your users know which image is the right one

Update the setup documentation embedded within your notebook(s) so that users run the correct image for the notebook.

  * Update the Docker image tag in the `Setup` cell at the top of the relevant Terra notebook(s) so that each notebook clearly states which Docker image it requires to run correctly.
  * We retain old Docker images, so if you have the old notebooks and its input files are still available and unchanged, you can reproduce your results!
  * Its more convenient for your collaborators if they do not have to switch between different containers to run different notebooks because it takes a couple minutes to redeploy the Terra VM with a different custom Docker image. So, when possible, ensure that all notebooks run on the most recent image.
  * Below is a convenient command to update the Docker tag in the documentation of your notebooks all at once. You don't need to do it this way. Just keep the setup instructions up to date!
```
# OPTIONAL: To update the Docker tag in the setup cell of all the notebooks in-place, run a command
# similar to the following and then check them in. Do this in the source of truth (your Git
# clone), not Terra.
cd path/to/my/clone/of/metis
find . -name "*.ipynb" -type f -print0 | \
  xargs -0 perl -i -pe \
  's/gcr.io\/verily-metis-data\/metis_terra:\d{8}_\d{6}/gcr.io\/verily-metis-data\/metis_terra:20200919_163335/g'
```
