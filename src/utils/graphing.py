import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
# df = pd.read_csv('output_dir/VideoMamba_tiny_SSv2_all/jepa_r0.csv')
df = pd.read_csv('output_dir/VideoMamba_tiny_SSv2_all/video_classification_frozen/ssv2-16x2x3/jepa_r0.csv')
print(df)
# print("Unique values in 'epoch':", df['epoch'].unique())
# print("Unique values in 'itr':", df['itr'].unique())
df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
df['itr'] = pd.to_numeric(df['itr'], errors='coerce')
df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
print("Inds where epoch is 1", (df['epoch']==1))

last_run_start_index = df[(df['epoch'] == 1) & (df['itr'] == 0)].index[-1]
print(last_run_start_index)

last_run_df = df[last_run_start_index:]

# Assuming 'itr' is the iteration column and 'loss' is the loss column
# Since we're now dealing with the last run only, we don't need to group by epoch again
# We'll directly calculate the mean loss for each epoch within the last run
# grouped_last_run = last_run_df.groupby(last_run_df['itr'] // 300)

# print(grouped_last_run)

# Calculate the mean loss for each epoch in the last run
# mean_loss_per_epoch_last_run = grouped_last_run['loss'].mean()

# Assuming 'itr' is the iteration column and 'loss' is the loss column
# Group by 'epoch' which is defined as every 300 iterations

# Assuming last_run_df is already filtered to the last run
# Ensure the DataFrame is sorted by 'itr'

# Calculate cumulative iterations for accurate x-axis representation
last_run_df['cumulative_itr'] = (last_run_df['epoch'] - 1) * 300 + last_run_df['itr']

downsample_factor = 10  # Adjust based on your dataset size
downsampled_df = last_run_df[::downsample_factor]
last_run_df = last_run_df.sort_values(by=['cumulative_itr'])

print(downsampled_df)

# Plotting
plt.figure(figsize=(12, 8))  # Larger figure size
plt.title('Loss Over Training Iterations (Last Run)')
plt.xlabel('Cumulative Iteration')
plt.ylabel('Loss')
plt.legend()

# Ensure min_loss and max_loss are floats
min_loss = float(downsampled_df['loss'].min())
max_loss = float(downsampled_df['loss'].max())
print(f"min_loss: {min_loss}, max_loss: {max_loss}")
num_ticks = 50

# Generate ticks manually, ensuring coverage across the actual range of loss values
tick_step = (max_loss - min_loss) / (num_ticks - 1)  # Adjust for num_ticks - 1 intervals
yticks = [max_loss - i*tick_step for i in range(num_ticks)]

# Now generate ticks safely
# yticks = np.linspace(max_loss, min_loss, num_ticks)
plt.yticks(yticks)
# Set y-axis limits to match the data range, considering the inversion
# plt.ylim(top=min_loss, bottom=max_loss)  # Note the order due to axis inversion

# Customize y-axis tick appearance
# plt.tick_params(axis='y', which='major', width=1, length=5)
plt.grid(True, linestyle='--', linewidth=0.5)
# plt.gca().invert_yaxis()  # Invert the y-axis
plt.tight_layout()  # Adjust layout to prevent overlap
plt.plot(downsampled_df['cumulative_itr'], downsampled_df['loss'], marker='o', linestyle='', label='Loss')

plt.show()
