import os
import subprocess

def download_file_from_google_drive(id, destination):
    """
    Download file from Google Drive using gdown.
    
    Parameters:
    id (str): The ID of the file to be downloaded.
    destination (str): The path where the file should be saved.
    """
    # Construct the gdown command
    command = f"gdown --id {id} -O {destination}"
    
    try:
        print(f"Downloading file with ID {id}...")
        subprocess.check_call(command, shell=True)
        print("Download completed.")
    except Exception as e:
        print(f"Failed to download file with ID {id}. Error: {e}")

# Example usage
folder_id = '16-1DwdrG1hXtcaeMzjpVDDIyOx1r2a3B'  # Replace with your actual folder ID
destination_path = '/jumbo/jinlab/datasets/SSv2'  # Replace with your desired destination path

# Assuming you have a list of file IDs within the folder
# file_ids = ['1PXgLuo_EZxEA85EiNEC61Z8L1bHUSjbM', '1f8lLB2XVZZJXdFUK_G4eyvOCDQLXAdHW', '16Yz9Fp6g5Qhsz1f2MWNcLAc5tJP71O1C', '1YotsI_ZWYStSEK3DQkbgnvWCGIzon1l_', '1Ng3YdU1OcWZ7nmkd_421XPNSL1UmXRks', '1MG8oAQgqLEnvv5qZfyH-yWnLQ2WG9dZx', '12JxImrmQSqWLy_zmSQirDJl7oPdXD-4n', '1f-YiRuEGiajZU445zaPEwkLsi4dLbpgZ', '1QJbgtfXQrhrQ-4XyMmLnorwNDJq8hCsn', '1NaYvIF9GgCG3JRnFQOMrzc9Hibb2u7H2', '1lRU8yIFjtZBoLk4PocZwdt7E4DXA_Lbo', '1f2pGGEVOzdqLxFmlinjJAaT_AOdtUucx', '1Xs1XC5gw_rEtiINfOt24quFKwCKLxX_Y', '15wB0gOFYOT4H_hd1UPs2kB9o9S6Vj787', '1-7mix1s8c4TNZN2MxkjSy6bDdmK8Fmuk', '1Wh3aQts2_m8gLgP0b0zhYywi6ah5dOf-', '1meaSfS0uSsOxpOaepKkftswbHtOVLAve', '1Yi9mfSpG214V_2bavpyXXMA6I24U_z5G', '1Vd95W1U9fqkxSgKMNcHj0q9z3-zxrJJ1', '1YsuBqca2rd6hexcrDpsvO1c78VCthLU-']  # Example file IDs

label_file_ids = ['1ypzrwJofqGJXgbs_xp8XhW_Zf1DPlrud', '1yq1T_HR-wzcT_kbk3afhXFJXlIPc6NX3', '1yr8QWFu5NCHG4xpVdffqNxftxZ98bDh7', '1ytD2_bsz-PGDIpHE_IMD1vprojeShal2']
file_names = ['validation.json', 'train.json', 'test.json', 'labels.json']

# for file_id, file_name in zip(label_file_ids, file_names):
#     # file_name = file_id.split('/')[-1]  # Extracting file name from the ID
#     file_destination = os.path.join(destination_path, file_name)
#     download_file_from_google_drive(file_id, file_destination)

# print("All files have been downloaded.")


# def rename_downloaded_files(source_directory, prefix="renamed_", suffix=""):
#     """
#     Renames all files in the source directory by adding a prefix.

#     Parameters:
#     source_directory (str): The path to the directory containing the files to be renamed.
#     prefix (str): The prefix to add to each filename. Defaults to "renamed_".
#     """
#     for i, filename in enumerate(os.listdir(source_directory)):
#         # Construct the old and new file paths
#         old_file_path = os.path.join(source_directory, filename)
#         new_file_path = os.path.join(source_directory, prefix + str(i) + suffix)

#         # Rename the file
#         os.rename(old_file_path, new_file_path)
#         print(f"Renamed '{filename}' to '{prefix}{i}{suffix}'")

# # Example usage
# source_directory = '/jumbo/jinlab/datasets/SSv2'  # The directory where the files were downloaded
# rename_downloaded_files(source_directory, prefix="20bn-something-something-v2-", suffix=".zip")

import csv
import random
import json

def find_video_files_finetune_ssv2(directory, labels, output_csv, label_map_file):
    video_extensions = ('.webm')

    with open(output_csv, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=' ')
      label_map_file = open(label_map_file, 'r')
      labels_str_to_int = json.load(label_map_file)

      label_file = open(labels, 'r')
      labels_dict = json.load(label_file)
      for label in labels_dict:
        file_name = label['id']
        abs_path = os.path.join(directory, f"{file_name}.webm")
        class_label = label['template'].replace('[','').replace(']','')

        writer.writerow([abs_path, labels_str_to_int[class_label]])

      # for root, dirs, files in os.walk(directory):
      #   for file in files:
      #     abs_path = os.path.join(root, file)
      #     class_label = random.randint(1, 10)
      #     writer.writerow([abs_path, class_label])

directory = "/scratch/SSv2/videos/20bn-something-something-v2"
labels_file = "/scratch/SSv2/labels/train.json"
output_csv_file = "/scratch/SSv2/labels/SSv2_train_filelist.csv"
label_map_file = "/scratch/SSv2/labels/labels.json"
# train file
find_video_files_finetune_ssv2(directory, labels_file, output_csv_file, label_map_file)