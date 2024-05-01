import numpy as np
import matplotlib.pyplot as plt

# load data
afs_encoded_tSNE = np.loadtxt('results/afs_encoded_tSNE_original_ghost.txt')
afs_encoded_PCA = np.loadtxt('results/afs_encoded_PCA_original.txt')
afs_encoded = np.loadtxt('results/afs_encoded_original_ghost.txt')

# separate target and source
source_pca = afs_encoded_PCA[:10000]
target_pca = afs_encoded_PCA[-100:]
source_tsne = afs_encoded_tSNE[:10000]
target_tsne = afs_encoded_tSNE[-100:]
source = afs_encoded[:10000]
target = afs_encoded[-100:]

threshold=0.01
print(source_pca.shape, source_tsne.shape, source.shape)
#threshold_pca = threshold / source_pca.shape[1] * 100
threshold_pca = threshold / 1 * 100
threshold_tsne = threshold / source_tsne.shape[1] * 100
threshold_all = threshold / source.shape[1] * 100
print(threshold_all, threshold_pca, threshold_tsne)

print(np.percentile(source_pca, [5,95], axis=0))
print(np.percentile(source_pca, threshold_pca, axis=0))


# Compare each entry in target with the threshold
more_extreme = np.logical_or(target_pca > np.percentile(source_pca, 100-threshold_pca, axis=0), target_pca < np.percentile(source_pca, threshold_pca, axis=0))
#print(more_extreme)
extreme_count = np.sum(more_extreme, axis=1)

# Label rows as 'violation' if count is greater than 0
violation_indices = np.where(extreme_count == source_pca.shape[1])[0]
#violation_indices = np.where(extreme_count > 0)[0]
# Display the results
print("Number of columns:", target.shape[1])
#print("95th Percentiles:", percentile_95)
#print("Extreme count per row:", extreme_count)
#print("Violation rows:", violation_indices)
print(len(violation_indices))
