sub -w "python conv1d_keras_train.py" -o out/ -n exp1-secondaryContact1-1d  
sub -w "python conv1d_dann_train.py" -o out/ -n exp1-secondaryContact1-1d-dann  
sub -w "python conv1d_cdan_train.py" -o out/ -n exp1-secondaryContact1-1d-cdan  

sub -w "python conv2d_keras_train.py" -o out/ -n exp1-secondaryContact1-2d  
sub -w "python conv2d_dann_train.py" -o out/ -n exp1-secondaryContact1-2d-dann  
sub -w "python conv2d_cdan_train.py" -o out/ -n exp1-secondaryContact1-2d-cdan  