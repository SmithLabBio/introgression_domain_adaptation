library(ggplot2)

setwd("~/Documents/GitHub/popAI/megan_testing/")

# read encoded data
encoded_original_ghost <- read.delim('results/afs/original/ghost/ghost_encododed.txt', sep = " ", header=FALSE)
encoded_original <- encoded_original_ghost[1:20000,]
encoded_ghost <- encoded_original_ghost[20001:20100,]
encoded_original_bgs <- read.delim('results/afs/original/ghost/bgs_encododed.txt', sep = " ", header=FALSE)
encoded_bgs <- encoded_original_bgs[20001:20100,]
encoded_original_test <- read.delim('results/afs/original/ghost/test_encododed.txt', sep = " ", header=FALSE)
encoded_test <- encoded_original_test[20001:20100,]
# pca
pca_model <- princomp(encoded_original)
original_pca <- data.frame(predict(pca_model, newdata = encoded_original))
original_pca_subsample <- original_pca[sample(nrow(original_pca), 5000), ]
ghost_pca <- data.frame(predict(pca_model, newdata = encoded_ghost))
bgs_pca <- data.frame(predict(pca_model, newdata = encoded_bgs))
test_pca <- data.frame(predict(pca_model, newdata = encoded_test))
original_pca_subsample$model = "source"
ghost_pca$model = "ghost"
bgs_pca$model = "BGS"
results = rbind(original_pca_subsample, ghost_pca, bgs_pca)
results$model <- factor(results$model, levels = c("source", "ghost", "BGS"))
color_palette <- c("source" = "#DDCC77", "ghost" = "#332288", "BGS" = "#CC6677")
pca_plot <- ggplot(results, aes(x = Comp.1, y = Comp.2, color = model)) +
  geom_point() +
  labs(x = "PC1", y = "PC2", color = NULL) + # Remove legend title
  scale_color_manual(values = color_palette) +
  theme_bw() + 
  theme(legend.text = element_text(size = 14),
        axis.title = element_text(size=14),
        legend.position = c(0.15, 0.15),    # Place legend inside at specified coordinates
        #legend.justification = c(0, 1),
        legend.background = element_rect(color = "black", fill = "white"),
        axis.text = element_text(size = 12))
pdf("PCA_afs.pdf", height=4, width=4.5)
pca_plot
dev.off()

# test
conduct_test <- function(source, target, threshold){
  threshold = threshold / ncol(source)
  source_percentiles_01 <- apply(source, 2, quantile, probs=threshold)
  source_percentiles_99 <- apply(source, 2, quantile, probs=1-threshold)
  results <- c()
  for (i in 1:nrow(target)){
    row = target[i,]
    less_than = row < source_percentiles_01
    greater_than = row > source_percentiles_99
    extreme = less_than | greater_than
    violation = any(extreme)
    results <- c(results, violation)
  }
  return(results)
}

results_test_ghost_pca <- conduct_test(original_pca, ghost_pca, 0.01)
print(sum(results_test_ghost_pca))
results_test_bgs_pca <- conduct_test(original_pca, bgs_pca, 0.01)
print(sum(results_test_bgs_pca))
results_test_test_pca <- conduct_test(original_pca, test_pca, 0.01)
print(sum(results_test_test_pca))


results_test_ghost <- conduct_test(encoded_original, encoded_ghost, 0.01)
print(sum(results_test_ghost))
results_test_bgs <- conduct_test(encoded_original, encoded_bgs, 0.01)
print(sum(results_test_bgs))
results_test_test <- conduct_test(encoded_original, encoded_test, 0.01)
print(sum(results_test_test))

# Create a dataframe with the information
data <- data.frame(
  Scenario = c("test", "ghost", "BGS"),
  Detection_Percentage = c(sum(results_test_test), sum(results_test_ghost), sum(results_test_bgs))
)
data$Scenario <- factor(data$Scenario, levels = c("test", "ghost", "BGS"))
color_palette <- c("test" = "#DDCC77", "ghost" = "#332288", "BGS" = "#CC6677")

# Create the bar plot
violations_detectec <- ggplot(data, aes(x = Scenario, y = Detection_Percentage, fill = Scenario)) +
  geom_bar(stat = "identity") +
  labs(y = "Percent Model Violations Detected (%)", x="") +
  scale_fill_manual(values = color_palette) +
  theme_bw() +
  theme(legend.text = element_text(size = 14),
        axis.title = element_text(size=14),
        legend.position = "none",    # Place legend inside at specified coordinates
        #legend.justification = c(0, 1),
        legend.background = element_rect(color = "black", fill = "white"),
        axis.text = element_text(size = 14))
pdf("ViolationsDetection_afs.pdf", height=4, width=4.5)
violations_detectec
dev.off()
