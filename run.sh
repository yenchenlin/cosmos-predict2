clear
EXP=predict2_video2world_training_2b_groot_gr1_480_mock_data
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
rm -rf /project/cosmos/yenchenl/imaginaire4-output/posttraining/video2world/2b_groot_gr1_480/
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
rm -rf /project/cosmos/yenchenl/imaginaire4-output/posttraining/video2world/2b_groot_gr1_480/
