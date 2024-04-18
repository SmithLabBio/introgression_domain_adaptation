sub -w "python conv1d_fine_train.py" -o out/ -n exp2-1d-fine
sub -w "python conv1d_ghost_train.py" -o out/ -n exp2-1d-ghost
sub -w "python conv1d_dann_train.py" -o out/ -n exp2-1d-dann
sub -w "python conv1d_cdan_train.py" -o out/ -n exp2-1d-cdan

sub -w "python conv2d_fine_train.py" -o out/ -n exp2-2d-fine
sub -w "python conv2d_ghost_train.py" -o out/ -n exp2-2d-ghost
sub -w "python conv2d_dann_train.py" -o out/ -n exp2-2d-dann
sub -w "python conv2d_cdan_train.py" -o out/ -n exp2-2d-cdan