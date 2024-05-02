library(ggplot2)

setwd("~/Documents/GitHub/popAI/megan_testing/")

# get roc curves for ghost
roc_ghost_original <- read.table(file='results/npy/original/ghost_roc.txt', sep =",", header=TRUE)
roc_ghost_cdan <- read.table(file='results/npy/cdan/ghost/ghost_roc.txt', sep =",", header=TRUE)
afs_roc_ghost_original <- read.table(file="results/afs/original/ghost_roc.txt", sep =",", header=TRUE)
afs_roc_ghost_cdan <- read.table(file="results/afs/cdan/ghost/ghost_roc.txt", sep =",", header=TRUE)

name_roc_ghost_original =  "Original Alignments"
name_roc_ghost_cdan = "CDAN Alignments"
name_afs_roc_ghost_original = "Original SFS"
name_afs_roc_ghost_cdan = "CDAN SFS"

roc_ghost_original$scenario<- name_roc_ghost_original
roc_ghost_cdan$scenario <-name_roc_ghost_cdan
afs_roc_ghost_original$scenario <-  name_afs_roc_ghost_original
afs_roc_ghost_cdan$scenario <-  name_afs_roc_ghost_cdan

auc_roc_ghost_original <- roc_ghost_original$aoc[1]
auc_roc_ghost_cdan <- roc_ghost_cdan$aoc[1]
auc_afs_roc_ghost_original <- afs_roc_ghost_original$aoc[1]
auc_afs_roc_ghost_cdan <- afs_roc_ghost_cdan$aoc[1]


ghost_all <- rbind(roc_ghost_original, roc_ghost_cdan, afs_roc_ghost_original, afs_roc_ghost_cdan)

color_palette <- c("CDAN Alignments" = "#AA78D4", 
                   "CDAN SFS" = "#004D40",
                   "Original Alignments" = "#FFC107", 
                   "Original SFS" = "#1E88E5")
ghost_all$scenario <- factor(ghost_all$scenario, levels = c(name_afs_roc_ghost_original, 
                                                            name_afs_roc_ghost_cdan, name_roc_ghost_original, 
                                                            name_roc_ghost_cdan))

# plot for ghost
ghost_plot <- ggplot(ghost_all, aes(x = fpr, y = tpr, color = scenario)) +
  geom_line(lwd=1) + 
  scale_color_manual(values = color_palette,
                     name = NULL,
                     labels = c(paste(name_afs_roc_ghost_original, " (AUC =", round(auc_afs_roc_ghost_original, 2), ")"),
                                paste(name_afs_roc_ghost_cdan, " (AUC =", round(auc_afs_roc_ghost_cdan, 2), ")"),
                                paste(name_roc_ghost_original, " (AUC =", round(auc_roc_ghost_original, 2), ")"),
                                paste(name_roc_ghost_cdan, " (AUC =", round(auc_roc_ghost_cdan, 2), ")")
                                
                                )) + 
  labs(x = "False Positive Rate", y = "True Positive Rate", title="A) Ghost Introgression") +
  theme_bw() + 
  geom_abline(slope=1, intercept=0, linetype = "dashed") + 
  theme(legend.text = element_text(size = 11),
      axis.title = element_text(size=14),
      legend.position = c(0.65, 0.20),    
      legend.background = element_rect(color = "black", fill = "white"),
      axis.text = element_text(size = 12),
      title = element_text(size=14))

# get roc curve for bgs 
roc_bgs_original <- read.table(file='results/npy/original/bgs_roc.txt', sep =",", header=TRUE)
roc_bgs_cdan <- read.table(file='results/npy/cdan/bgs/bgs_roc.txt', sep =",", header=TRUE)
afs_roc_bgs_original <- read.table(file="results/afs/original/bgs_roc.txt", sep =",", header=TRUE)
afs_roc_bgs_cdan <- read.table(file="results/afs/cdan/bgs/bgs_roc.txt", sep =",", header=TRUE)

name_roc_bgs_original =  "Original Alignments"
name_roc_bgs_cdan = "CDAN Alignments"
name_afs_roc_bgs_original = "Original SFS"
name_afs_roc_bgs_cdan = "CDAN SFS"

roc_bgs_original$scenario<- name_roc_bgs_original
roc_bgs_cdan$scenario <-name_roc_bgs_cdan
afs_roc_bgs_original$scenario <-  name_afs_roc_bgs_original
afs_roc_bgs_cdan$scenario <-  name_afs_roc_bgs_cdan

auc_roc_bgs_original <- roc_bgs_original$aoc[1]
auc_roc_bgs_cdan <- roc_bgs_cdan$aoc[1]
auc_afs_roc_bgs_original <- afs_roc_bgs_original$aoc[1]
auc_afs_roc_bgs_cdan <- afs_roc_bgs_cdan$aoc[1]


bgs_all <- rbind(roc_bgs_original, roc_bgs_cdan, afs_roc_bgs_original, afs_roc_bgs_cdan)

color_palette <- c("CDAN Alignments" = "#AA78D4", 
                   "CDAN SFS" = "#004D40",
                   "Original Alignments" = "#FFC107", 
                   "Original SFS" = "#1E88E5")
bgs_all$scenario <- factor(bgs_all$scenario, levels = c(name_afs_roc_bgs_original, 
                                                            name_afs_roc_bgs_cdan, name_roc_bgs_original, 
                                                            name_roc_bgs_cdan))

# plot for bgs
bgs_plot <- ggplot(bgs_all, aes(x = fpr, y = tpr, color = scenario)) +
  geom_line(lwd=1) + 
  scale_color_manual(values = color_palette,
                     name = NULL,
                     labels = c(paste(name_afs_roc_bgs_original, " (AUC =", round(auc_afs_roc_bgs_original, 2), ")"),
                                paste(name_afs_roc_bgs_cdan, " (AUC =", round(auc_afs_roc_bgs_cdan, 2), ")"),
                                paste(name_roc_bgs_original, " (AUC =", round(auc_roc_bgs_original, 2), ")"),
                                paste(name_roc_bgs_cdan, " (AUC =", round(auc_roc_bgs_cdan, 2), ")")
                                )) + 
  labs(x = "False Positive Rate", y = "True Positive Rate", title="B) BGS") +
  theme_bw() + 
  geom_abline(slope=1, intercept=0, linetype = "dashed") + 
  theme(legend.text = element_text(size = 11),
        axis.title = element_text(size=14),
        legend.position = c(0.65, 0.35),    
        legend.background = element_rect(color = "black", fill = "white"),
        axis.text = element_text(size = 12),
        title = element_text(size=14))

pdf('ROC_ghost.pdf', height=5, width=6)
ghost_plot
dev.off()
pdf('ROC_bgs.pdf', height=5, width=6)
bgs_plot
dev.off()

