import pandas as pd
import numpy as np

def gaussian_weight(distance, threshold, sigma):
    # Compute Gaussian weight
    weight = np.exp(-0.5 * ((distance - threshold) / sigma) ** 2)
    return weight

def smooth_delete(data, threshold_lower, threshold_upper, sigma, delete_prob):
    # Compute threshold center
    threshold_center = (threshold_lower + threshold_upper) / 2

    # Compute distance of each sample to the threshold center
    distances = np.abs(data - threshold_center)

    # Compute Gaussian weights
    weights = gaussian_weight(distances, threshold_center, sigma)

    # Delete samples based on weights
    delete_mask = np.random.rand(len(data)) < delete_prob * weights

    # Return indices of deleted samples
    deleted_indices = data.index[delete_mask]

    return deleted_indices

# Load CSV file
df = pd.read_csv("ChEMBLMolsRaw.csv")

# Set thresholds and parameters: directly delete 750-1500;
# for the remaining 500-750 and 1500-2000 ranges, apply Gaussian-probability smoothing deletion
threshold_lower = 500
threshold_upper = 2000
threshold_smooth_lower = 750
threshold_smooth_upper = 1500
sigma = 500  # Standard deviation of the Gaussian distribution, controls the width of the weight distribution
delete_prob = 0.8  # Deletion probability, can be adjusted as needed

# Set random seed to ensure reproducible random numbers
np.random.seed(37)

# Select data outside the threshold range
selected_data = df[(df['standard_value'] > threshold_upper) | (df['standard_value'] < threshold_lower)]

# Process data between thresholds: apply Gaussian deletion to 500-2000 molecules, then remove all 750-1500 molecules

to_delete_data = df[(df['standard_value'] <= threshold_upper) & (df['standard_value'] >= threshold_lower)]  # Gaussian deletion for 500-2000 molecules
deleted_indices_smoothed = smooth_delete(to_delete_data['standard_value'], threshold_lower, threshold_upper, sigma, delete_prob)  # Gaussian deletion for 500-2000 molecules
smoothed_data = to_delete_data.drop(deleted_indices_smoothed)  # Gaussian deletion for 500-2000 molecules

deleted_data = smoothed_data[(smoothed_data['standard_value'] <= threshold_smooth_lower) | (smoothed_data['standard_value'] >= threshold_smooth_upper)]

'''
import seaborn as sns
import matplotlib.pyplot as plt

# Assume your data is in the 'standard_value' column
data = deleted_data['standard_value'].values

# Create KDE plot, adjust figure size
plt.figure(figsize=(8, 5), dpi=300)
sns.histplot(data, kde=True, color='skyblue', bins=100)

# Add text annotation with total count
total_count = len(deleted_data)
plt.text(1500, 25, f'Total count: {total_count}', fontsize=12, ha='right')

# Set x and y axis labels
plt.xlabel('Standard Value')
plt.ylabel('Number of Compounds')

# Adjust title to display on multiple lines
plt.title('IC50 Value Distribution of Compounds\nwith Standard Value between 500 and 2000')

# Set y-axis range
plt.ylim(0, 100)

plt.savefig('After Gaussian deletion.svg', format='svg', dpi=300)

# Show KDE plot
plt.show()


'''

# Merge processed data
final_data = pd.concat([selected_data, deleted_data])

# Save results to a new CSV file
final_data.to_csv('ChemblMolsGS.csv', index=False)
