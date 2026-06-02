import os
import sys
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = Path(__file__).resolve().parent / "dilated-se-fire-10963f0a6a5a.json"
FILE_TO_UPLOAD = Path(__file__).resolve().parent / "processed.zip"
FOLDER_ID = "1b81qr2Q3Je2sN027C7MTlxonJFKystlJ"

def upload_file():
    if not KEY_FILE.exists():
        print(f"Error: Key file not found at {KEY_FILE}")
        sys.exit(1)
    if not FILE_TO_UPLOAD.exists():
        print(f"Error: Zip file to upload not found at {FILE_TO_UPLOAD}")
        sys.exit(1)

    print("Authenticating with Google Drive API...")
    creds = service_account.Credentials.from_service_account_file(
        str(KEY_FILE),
        scopes=['https://www.googleapis.com/auth/drive']
    )
    drive_service = build('drive', 'v3', credentials=creds)

    # Search if file already exists in folder to avoid duplicate upload
    print(f"Checking if {FILE_TO_UPLOAD.name} already exists in target folder...")
    query = f"'{FOLDER_ID}' in parents and name = '{FILE_TO_UPLOAD.name}' and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id, name, size)").execute()
    files = results.get('files', [])

    if files:
        existing_file = files[0]
        print(f"File already exists on Drive: ID={existing_file.get('id')}, Size={int(existing_file.get('size')) / (1024*1024):.2f} MB")
        print("Skipping upload. If you want to re-upload, please delete the file from Drive first.")
        return

    # Resumable upload metadata
    file_metadata = {
        'name': FILE_TO_UPLOAD.name,
        'parents': [FOLDER_ID]
    }
    
    # 100MB chunk size for stability
    media = MediaFileUpload(
        str(FILE_TO_UPLOAD),
        mimetype='application/zip',
        resumable=True,
        chunksize=100*1024*1024
    )
    
    print(f"Starting resumable upload of {FILE_TO_UPLOAD.name} ({FILE_TO_UPLOAD.stat().st_size / (1024*1024*1024):.2f} GB)...")
    request = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    )
    
    response = None
    last_progress = -1
    
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                if progress != last_progress:
                    print(f"Upload progress: {progress}%")
                    last_progress = progress
        except Exception as e:
            print(f"\nUpload failed: {e}")
            sys.exit(1)
            
    print(f"\n[SUCCESS] Upload complete! File ID on Drive: {response.get('id')}")

if __name__ == '__main__':
    upload_file()
