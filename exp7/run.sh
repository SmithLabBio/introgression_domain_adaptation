sub -w "python conv2d_keras_train.py" -o out/ -n exp7-2d  
sub -w "python conv2d_ghost_train.py" -o out/ -n exp7-2d-ghost  
sub -w "python conv2d_dann_train.py" -o out/ -n exp7-2d-dann  
sub -w "python conv2d_cdan_train.py" -o out/ -n exp7-2d-cdan -m 32 
sub -w "python conv2d_fine_train.py" -o out/ -n exp7-2d-fine -m 64 