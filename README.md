## Usage

### News

- The ConvD complemention is based on the OpenKE framework.

### Requirements

- Python 3.6
- Pytorch 1.5.0 

### Training

 Run ''bash make.sh'' to compile the base package and then use the commands as:

	python train_ConvD.py --dataset FB15K237 --hidden_size 800 --num_of_filters 16 --input_dropout 0 --map_dropout 0.1 --dropout 0.5 --optim adam --neg_num 10 --valid_step 1000 --nbatches 100 --num_epochs 1000 --learning_rate 0.00008 --lmbda 0.2 --model_name ConvD_FB --mode train --kernel_size 1 --use_init 0 --negative_slope 1
	
	python train_ConvD.py --dataset WN18RR --hidden_size 350 --num_of_filters 48 --input_dropout 0 --dropout 0 --optim adam --map_dropout 0.2 --neg_num 10 --valid_step 1000 --nbatches 100 --num_epochs 1000 --learning_rate 0.00008 --lmbda 0.2 --model_name ConvD_WN --mode train --kernel_size 1 --use_init 0 --negative_slope 0