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
file_ids = ['1PXgLuo_EZxEA85EiNEC61Z8L1bHUSjbM', '1f8lLB2XVZZJXdFUK_G4eyvOCDQLXAdHW', '16Yz9Fp6g5Qhsz1f2MWNcLAc5tJP71O1C', '1YotsI_ZWYStSEK3DQkbgnvWCGIzon1l_', '1Ng3YdU1OcWZ7nmkd_421XPNSL1UmXRks', '1MG8oAQgqLEnvv5qZfyH-yWnLQ2WG9dZx', '12JxImrmQSqWLy_zmSQirDJl7oPdXD-4n', '1f-YiRuEGiajZU445zaPEwkLsi4dLbpgZ', '1QJbgtfXQrhrQ-4XyMmLnorwNDJq8hCsn', '1NaYvIF9GgCG3JRnFQOMrzc9Hibb2u7H2', '1lRU8yIFjtZBoLk4PocZwdt7E4DXA_Lbo', '1f2pGGEVOzdqLxFmlinjJAaT_AOdtUucx', '']  # Example file IDs

for file_id in file_ids:
    file_name = file_id.split('/')[-1]  # Extracting file name from the ID
    file_destination = os.path.join(destination_path, file_name)
    download_file_from_google_drive(file_id, file_destination)

print("All files have been downloaded.")