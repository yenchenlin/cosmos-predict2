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
cd datasets
tar -xvzf bridge_train_data.tar.gz -C .
mv opensource_robotdata/bridge ./
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

The model will be post-trained using the Bridge dataset. See `cosmos_predict2/configs/action_conditional/defaults/data.py` to understand how the dataloader is defined.

To add action as additional condition, we create new `conditioner` to support action in `cosmos_predict2/configs/action_conditional/defaults/conditioner.py`.

The checkpoints will be saved to checkpoints/PROJECT/GROUP/NAME. In the above example, PROJECT is `posttraining`, GROUP is `video2world`, NAME is `action_conditional_predict2_video2world_2b_training_${now:%Y-%m-%d}_${now:%H-%M-%S}`.

See the job config to understand how they are determined.
```python
action_conditional_predict2_video2world_2b_training = dict(
    defaults=[
        {"override /model": "action_conditional_predict2_v2w_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_train": "bridge_train"},
        {"override /data_val": "bridge_val"},
        "_self_",
    ],
    model=dict(
        config=dict(
            num_video_frames=13,
            resolution="720",
            fsdp_shard_size=8,
        )
    ),
    job=dict(project="posttraining", group="video2world", name="action_conditional_predict2_video2world_2b_training_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    ...
)
```


## 3. Inference for Bridge
##### Cosmos-Predict2-2B-Video2World
For example, if a posttrained checkpoint with 1000 iterations is to be used, run the following command.
Use `--dit_path` argument to specify the path to the post-trained checkpoint.
```
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python examples/video2world_action.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/action_conditional_predict2_video2world_2b_training_${now:%Y-%m-%d}_${now:%H-%M-%S}/checkpoints/model/iter_000001000.pt" \
  --input_video datasets/bridge/opensource_robotdata/bridge/videos/test/13/rgb.mp4 \
  --input_annotation datasets/bridge/opensource_robotdata/bridge/annotation/test/13.json \
  --num_conditional_frames 1 \
  --save_path output/generated_video.mp4 \
  --guidance 0 \
  --seed 0 \
  --disable_guardrail \
  --disable_prompt_refiner 
```