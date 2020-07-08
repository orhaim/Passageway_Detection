# doors_detection

Transfer learning based on  https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html



#### colab:
see colab.ipynb

#### Train:
$ python trainer.py --num_epochs 500  --train_dir data/train --test_dir data/test

 
The weights will be saved into saved_models folder, and model_state_dict.pt will be updated with last weight.

*While training, evaluation is based on test_dir 

#### Test your result:
create a new annotation file (using the trained weight)\
$ python runMe.py

test the results (based on the annotation file runMe.py create)\
*maybe wont work on colab\

$ python project_test.py


#### Data set:
test and train dirs structures should be:
<pre>
train_dir:
├── annotations.txt
└── images
    ├── image0001_0_aug.jpg
    ├── image0001_1_aug.jpg
    .
    .
    └── image0199_5_aug.jpg
    
test_dir:
├── annotations.txt
└── images
    ├── image0200.jpg
    ├── image0201.jpg
    .
    .
    └──  image0300.jpg
 </pre>





