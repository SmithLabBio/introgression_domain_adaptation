//Number of population samples (demes)
2
//Population effective sizes (number of genes)
POPSIZE
POPSIZE
//Samples sizes and samples age 
20
20
//Growth rates: negative growth implies population expansion
0
0
//Number of migration matrices : 0 implies no migration between demes
2
//Migration matrix 0
0 RMIG
RMIG 0
//Migration matrix 1
0 0
0 0
//historical event: time, source, sink, migrants, new deme size, growth rate, migr mat index
2 historical event
TMIG 0 0 0 1 0 1
TDIV 0 1 1 1 0 1
//Number of independent loci [chromosome] 
1 0
//Per chromosome: Number of contiguous linkage Block: a block is a set of contiguous loci
1
//per Block:data type, number of loci, per gen recomb and mut rates
FREQ 0 0 1e-8