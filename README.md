### 1. Build the Environment

Run the following command in your terminal where the `environment.yml` file is located:

```bash
conda env create -f environment.yml
```

Then, activate your new environment:

```bash
conda activate Touch3D
```

### 3. Install PyTorch and Related Packages

Now that your environment is set up with Python 3.9, run the following command to install PyTorch, torchvision, torchaudio, and the appropriate CUDA toolkit. Below is for CUDA 12.1:

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.8 -c pytorch -c nvidia
```

### 4. (Optional) Install Additional pip Packages

If you didn’t include the pip requirements in the YAML file, you can install them now:

```bash
pip install -r requirements.txt
```

This two-step process ensures you have a conda environment with Python 3.9 and then installs the CUDA-enabled PyTorch packages from the specified channels.