# Post-training for Action-Conditioning

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


### 1.2 Post-train the Model
##### Cosmos-Predict2-2B-Video2World
```bash
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/action_conditional/config.py -- experiment="action_conditional_predict2_video2world_2b_training"
```

##### Cosmos-Predict2-14B-Video2World
```bash
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/action_conditional/config.py -- experiment="action_conditional_predict2_video2world_14b_training"
```


### 1.3 Inference
2B model
`s3://checkpoints-us-east-1/cosmos_diffusion_v2/action_conditional/i_frame_action_conditional-Cosmos-Predict2-2B-Res-720-Fps-16-debug-lr-2e-5-cf-1/checkpoints/iter_000060000`

14B model
`s3://checkpoints-us-east-1/cosmos_diffusion_v2/action_conditional/i_frame_action_conditional-Cosmos-Predict2-14B-Res-720-Fps-16-debug-lr-1e-5-cf-1-state-t-2-tempwin-4/checkpoints/iter_000050000`
