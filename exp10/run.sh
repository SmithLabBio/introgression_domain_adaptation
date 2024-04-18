sub -w "python conv1d_dann_train.py" -o out/ -n exp10-1d-dann -m 32 
sub -w "python conv1d_cdan_train.py" -o out/ -n exp10-1d-cdan -m 32 
sub -w "python conv1d_fine_train.py" -o out/ -n exp10-1d-fine -m 32 
sub -w "python conv1d_ghost_train.py" -o out/ -n exp10-1d-ghost -m 32 

sub -w "python conv2d_dann_train.py" -o out/ -n exp10-2d-dann -m 32 
sub -w "python conv2d_cdan_train.py" -o out/ -n exp10-2d-cdan -m 32 
sub -w "python conv2d_fine_train.py" -o out/ -n exp10-2d-fine -m 32 
sub -w "python conv2d_ghost_train.py" -o out/ -n exp10-2d-ghost -m 32 