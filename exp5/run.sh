sub -w "python conv1d_keras_train.py" -o out/ -n exp1-secondaryContact2-1d  
sub -w "python conv1d_dann_train.py" -o out/ -n exp1-secondaryContact2-1d-dann  
sub -w "python conv1d_cdan_train.py" -o out/ -n exp1-secondaryContact2-1d-cdan  
sub -w "python conv1d_fine_train.py" -o out/ -n exp1-secondaryContact2-1d-fine  

sub -w "python conv2d_keras_train.py" -o out/ -n exp1-secondaryContact2-2d  
sub -w "python conv2d_dann_train.py" -o out/ -n exp1-secondaryContact2-2d-dann  
sub -w "python conv2d_cdan_train.py" -o out/ -n exp1-secondaryContact2-2d-cdan  
sub -w "python conv2d_fine_train.py" -o out/ -n exp1-secondaryContact2-2d-fine  