### 1. Build the Environment

- Run the following command in your terminal where the `environment.yml` file is located:

    ```bash
    conda env create -f environment.yml
    ```

- Then, activate your new environment:

    ```bash
    conda activate Touch3D
    ```

### 2. Install PyTorch and Related Packages

- Now that your environment is set up with Python 3.9, run the following command to install PyTorch, torchvision, torchaudio, and the appropriate CUDA toolkit. Below is for CUDA 12.1:

    ```bash
    conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.8 -c pytorch -c nvidia
    ```

### 3. (Optional) Install Additional pip Packages

- If you didn’t include the pip requirements in the YAML file, you can install them now:

    ```bash
    pip install -r requirements.txt
    ```

- 

### 4. (Optional) Training the Network

- Create the folder 'Training/Logs' (the checkpoints and tf events will be saved here)

- Run the following command in your terminal to train the network:

    ```bash
    python PPO.py
    ``` 
    **Note:** The network will train from scratch for 500k steps. You can change the parameters in `PPO.py` file or give a pretrained model path in `conf/RL.yaml` 

### 5. Test the Network on Unseen Objects

- Create the folder 'Outputs' (3D poit cloud will be saved here)

- Set configuration at `conf/test.yaml` for testing:

    - RL:
        - pretrain_model_path: `Training/Logs/PPO_Contact_AMB/your_latest_model.zip` 

    - Environment:
        - Object:
            - object_name: Name of the test object 
            - urdf path of the object: `objects/ycb/object/model.urdf`

- Run the following command in your terminal to test the model:

    ```bash
    python test.py
    ``` 

 The testing will begin and point cloud is generated after 5000 evaluation steps are over or the sensor goes out of bounds.

    **Note:** If due to some reason the point cloud cannot be visualized after testing is finished; follow the steps below to visualize the generated point cloud.

### 6:  (optional) Point Cloud Visualization

- Navigate to the `visualize_npy.py` file

- Change the default path of the point cloud to the `.npy` file generated during testing, Example: `outputs/DEMO_model_96.95.npy`

- Execute the file:

    ```bash
    python visualize_npy.py
    ```

    **NOTE:The live visualiztion is turned off by default to improve performance**
