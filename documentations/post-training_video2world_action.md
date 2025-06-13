# Video2World Post-training for Action-conditioning

We provide an example post-training instruction from a pre-trained video2world checkpoint.

## 1. Preparing Data
### 1.1 Download Bridge training dataset
We leverage the train/val splits of Bridge from IRASim as the dataset for action-conditional post-training.
Please use the following link to download the Bridge training dataset.
under `cosmos-predict2/` folder, run:
```
wget https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz
mv bridge_train_data.tar.gz datasets/
tar -xvzf your_file.tar.gz -C .
mv bridge/opensource_robotdata/bridge ./
```

Dataset folder format:
```
datasets/bridge/
├── annotations/
│   ├── *.json
├── videos/
    ├── *.mp4
```


## 2. Post-training

##### Cosmos-Predict2-2B-Video2World
Run the following command to execute an example post-training job with Bridge data.
```bash
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment="action_conditional_predict2_video2world_2b_training"
```

The model will be post-trained using the GR1 dataset. See the config predict2_video2world_training_2b_groot_gr1_480 defined in cosmos_predict2/configs/base/experiment/groot.py to understand how the dataloader is defined.


The checkpoints will be saved to checkpoints/PROJECT/GROUP/NAME. In the above example, PROJECT is posttraining, GROUP is video2world, NAME is 2b_groot_gr1_480.

See the job config to understand how they are determined.
```python
predict2_video2world_training_2b_groot_gr1_480 = dict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="video2world",
            name="2b_groot_gr1_480",
        ),
        ...
    )
)
```


## 3. Inference for DreamGen Benchmark

### 1.3 Inference
2B model
`s3://checkpoints-us-east-1/cosmos_diffusion_v2/action_conditional/i_frame_action_conditional-Cosmos-Predict2-2B-Res-720-Fps-16-debug-lr-2e-5-cf-1/checkpoints/iter_000060000`

