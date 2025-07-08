import kagglehub

path = kagglehub.dataset_download("davidgauthier/glassdoor-job-reviews-2")
print("Path to dataset files:", path)

path=''

import os

print("Files in dataset folder:")
print(os.listdir(path))