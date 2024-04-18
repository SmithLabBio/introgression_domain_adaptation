# exp1
Single convolutional layer

# exp2 
Three convolutional layers, 50 epochs, 0.0003 learning rate, 1500 snps

# exp3
Three convolutional layers, 20 epochs, 0.001 learning rate, 500 snps

# exp4

# exp5 
Same network as exp1, addition of finetuning, trained on ghost2 and secondary2 which approximates bears

# exp6
Same network as exp1, trained on ghost1 and secondary1, with fine tuning and ghost only training/testing added

# exp7 
Same network as 6 with ghost3 and secondaryContact3

# exp8 
Same as 7 but without variable lambda

# exp9 
Same as 7 but with 3000 snps used

# exp10
Change to softmax, plotting of finetune 
1D looks good.

# exp11
Same as 10 but with learning rate= 0.00001 and max lambda = 10
1D Looks pretty good.

# exp12
Same as 11 but with lambda = 1
1D Looks pretty good but not as good as 11.

# exp13
Same as 11 but with learning rate = 0.001
Bad

# exp14
Same as 13 but with max lambda of 5
Bad