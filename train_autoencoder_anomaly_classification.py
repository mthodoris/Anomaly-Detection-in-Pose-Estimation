import numpy as np
import os
from tqdm import tqdm
import os
import datetime
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



current_datetime = datetime.datetime.now()
current_datetime_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")


class PoseDataset(Dataset):
    def __init__(self, poses):
        self.poses = poses

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx]
        return torch.tensor(pose, dtype=torch.float32)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 9, 9]
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=1),  # [B, 32, 3, 3]
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3, padding=1, output_padding=1),  # [B, 16, 9, 9]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 1, 18, 18]
            nn.Sigmoid()  # Sigmoid for values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size//3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            # nn.Softmax()
            nn.Sigmoid(),  # Sigmoid for values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist,squareform
def euclidean_similarity_matrix(keypoints, kpt):
    keypoints_array = np.array(keypoints)
    # Calculate pairwise Euclidean distances between all keypoints
    # distances = cdist(keypoints_array, keypoints_array, 'euclidean')
    distances = distance.euclidean(keypoints_array, keypoints_array)

    # Create similarity matrix from distances
    similarity_matrix = np.exp(-distances ** 2)

    return similarity_matrix

# Function to calculate Euclidean distance between two keypoints
def euclidean_distance(kpt1, kpt2):
    return distance.euclidean(kpt1, kpt2)

# Function to create similarity matrix for a list of keypoints
def create_similarity_matrix(poses):
    num_poses = len(poses)
    similarity_matrix = np.zeros((num_poses, num_poses))

    for i in range(num_poses):
        for j in range(i, num_poses):
            pose1 = poses[i]
            pose2 = poses[j]

            # Calculate average distance between corresponding keypoints
            avg_distance = np.mean([euclidean_distance(pose1[k], pose2[k]) for k in range(len(pose1))])

            # Store symmetrically in the matrix
            similarity_matrix[i][j] = avg_distance
            similarity_matrix[j][i] = avg_distance

    return similarity_matrix


#Load the dataset
df = pd.read_csv('/home/thodoris/Desktop/pose_dataset/train_skeletons_xy.csv')

poses = []
poses2 = []
poses3 = []
poses4 = []
for j in range(len(df)):
    pose = []
    pose2 = []
    pose3 = []

    for i in range(18):
        # print(i)
        x = df[f'kpt_{i}_x'][j]
        y = df[f'kpt_{i}_y'][j]
        pose.append((x, y))
        pose2.append(np.array([x, y]))
        pose3.append([x, y])
    poses.append(pose)
    poses2.append(pose2)
    poses3.append(pose3)
    poses4.append(np.array(pose3))

distance_matrices = []
similarity_matrices = []
for i in range(len(poses4)):
    pairwise_distances = pdist(poses4[i], metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    sigma = 1
    similarity_matrix = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
    distance_matrices.append(distance_matrix)
    similarity_matrices.append(similarity_matrix)
# combined_array = np.stack(distance_matrices)

# input_data = np.stack(distance_matrices)
input_data = np.stack(similarity_matrices)

# Flatten each pose into a single vector
# flattened_poses = [np.array([coord for keypoint in pose for coord in keypoint]) for pose in poses]

# similarity_matrix = euclidean_similarity_matrix(poses[0], poses2[0])

# for i in range(len(poses3)):

# Convert to numpy array
# input_data = np.array(flattened_poses)

input_size = 18
hidden_size = 9
model = Autoencoder(input_size=input_size, hidden_size=hidden_size)
# model = ConvAutoencoder()

input_tensor = torch.tensor(input_data, dtype=torch.float32)
# Assuming you have your flattened pose data in 'input_data'
pose_dataset = PoseDataset(input_data)

# Batch size for training
batch_size = 32

# Create DataLoader
data_loader = DataLoader(dataset=pose_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
pbar = tqdm(total=num_epochs)

log_dir = './logs/autoencoder/' + current_datetime_string
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    pbar.update()
    writer.add_scalar('autoenoder', loss.item(), epoch)


torch.save(model.state_dict(), 'edit_new_conv_simple_autoencoder_18_9_3.pth')
# # Inference: Get reconstructed poses
reconstructed_poses = model(input_tensor).detach().numpy()
# print("a")