sub -w "python conv1d_ghost_train.py" -o out/ -n exp11-1d-ghost
sub -w "python conv1d_dann_train.py" -o out/ -n exp11-1d-dann
sub -w "python conv1d_cdan_train.py" -o out/ -n exp11-1d-cdan
sub -w "python conv1d_fine_train.py" -o out/ -n exp11-1d-fine

sub -w "python conv2d_ghost_train.py" -o out/ -n exp11-2d-ghost
sub -w "python conv2d_dann_train.py" -o out/ -n exp11-2d-dann
sub -w "python conv2d_cdan_train.py" -o out/ -n exp11-2d-cdan
sub -w "python conv2d_fine_train.py" -o out/ -n exp11-2d-fine