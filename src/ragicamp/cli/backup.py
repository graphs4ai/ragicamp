"""Backblaze B2 backup and download operations."""

import os
import time
from pathlib import Path
from typing import Optional


def get_b2_client():
    """Get B2 S3 client and credentials.

    Returns:
        Tuple of (s3_client, endpoint, key_id, app_key) or (None, None, None, None) on error.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("Error: boto3 not installed. Install with: uv pip install boto3")
        return None, None, None, None

    key_id = os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    app_key = os.environ.get("B2_APP_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")

    if not key_id or not app_key:
        print("Error: Backblaze credentials not set.")
        print("Set environment variables:")
        print("  export B2_KEY_ID='your-key-id'")
        print("  export B2_APP_KEY='your-application-key'")
        print("  export B2_ENDPOINT='https://s3.us-east-005.backblazeb2.com'  # optional")
        return None, None, None, None

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(signature_version="s3v4"),
    )

    return s3, endpoint, key_id, app_key


def list_backups(bucket: str, limit: int = 20) -> list[str]:
    """List available backups in bucket.

    Returns:
        List of backup names (timestamps), sorted newest first.
    """
    s3, _, _, _ = get_b2_client()
    if s3 is None:
        return []

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix="ragicamp-backup/", Delimiter="/")
        prefixes = response.get("CommonPrefixes", [])
        if not prefixes:
            return []

        backup_names = sorted([p["Prefix"].split("/")[1] for p in prefixes], reverse=True)
        return backup_names[:limit]
    except Exception as e:
        print(f"Error listing backups: {e}")
        return []


def backup(
    dirs_to_backup: list[Path],
    bucket: str,
    prefix: str,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> int:
    """Upload directories to B2.

    Args:
        dirs_to_backup: List of directories to backup
        bucket: B2 bucket name
        prefix: S3 key prefix
        dry_run: If True, only preview files
        continue_on_error: If True, continue on upload errors

    Returns:
        Exit code (0 on success)
    """
    from tqdm import tqdm

    s3, _, _, _ = get_b2_client()
    if s3 is None:
        return 1

    print(f"Backing up to: s3://{bucket}/{prefix}/")
    print(f"Source directories: {', '.join(str(d) for d in dirs_to_backup)}")

    # Collect files to upload
    files_to_upload = []
    for backup_dir in dirs_to_backup:
        for root, dirs, files in os.walk(backup_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for file in files:
                if file.startswith("."):
                    continue
                local_path = Path(root) / file
                relative_path = local_path.relative_to(backup_dir.parent)
                s3_key = f"{prefix}/{relative_path}"
                files_to_upload.append((local_path, s3_key))

    if not files_to_upload:
        print("No files to upload.")
        return 0

    total_bytes = sum(f[0].stat().st_size for f in files_to_upload)
    print(f"Found {len(files_to_upload)} files ({total_bytes / (1024**3):.2f} GB)")

    if dry_run:
        print("\n[DRY RUN] Would upload:")
        for local_path, s3_key in files_to_upload[:10]:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  {s3_key} ({size_mb:.1f} MB)")
        if len(files_to_upload) > 10:
            print(f"  ... and {len(files_to_upload) - 10} more files")
        return 0

    # Upload with byte-level progress
    uploaded_bytes = 0
    uploaded_files = 0
    errors = []
    start_time = time.time()

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Uploading") as pbar:
        for local_path, s3_key in files_to_upload:
            file_size = local_path.stat().st_size
            try:
                s3.upload_file(str(local_path), bucket, s3_key)
                uploaded_files += 1
                uploaded_bytes += file_size
                pbar.update(file_size)
            except Exception as e:
                errors.append((local_path, str(e)))
                pbar.update(file_size)
                if not continue_on_error:
                    print(f"\nError uploading {local_path}: {e}")
                    return 1

    elapsed = time.time() - start_time
    speed_mbps = (uploaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Uploaded {uploaded_files}/{len(files_to_upload)} files to s3://{bucket}/{prefix}/")
    print(f"  Total: {uploaded_bytes / (1024**3):.2f} GB in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)")

    if errors:
        print(f"\n⚠️  {len(errors)} errors:")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    return 0


def download(
    bucket: str,
    backup_name: Optional[str] = None,
    artifacts_only: bool = False,
    outputs_only: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> int:
    """Download backup from B2.

    Args:
        bucket: B2 bucket name
        backup_name: Specific backup to download (None = most recent)
        artifacts_only: Only download artifacts/
        outputs_only: Only download outputs/
        dry_run: If True, only preview files
        continue_on_error: If True, continue on download errors

    Returns:
        Exit code (0 on success)
    """
    from tqdm import tqdm

    s3, _, _, _ = get_b2_client()
    if s3 is None:
        return 1

    # Determine which backup to download
    if backup_name is None:
        backups = list_backups(bucket, limit=1)
        if not backups:
            print("No backups found.")
            return 1
        backup_name = backups[0]
        print(f"Most recent backup: {backup_name}")

    prefix = f"ragicamp-backup/{backup_name}"
    print(f"Downloading from: s3://{bucket}/{prefix}/")

    # List all files in the backup
    files_to_download = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                size = obj["Size"]
                # Extract local path: remove prefix, keep artifacts/... or outputs/...
                relative_path = s3_key[len(prefix) + 1 :]  # +1 for trailing /
                if not relative_path:
                    continue

                # Filter by artifacts-only or outputs-only
                if artifacts_only and not relative_path.startswith("artifacts"):
                    continue
                if outputs_only and not relative_path.startswith("outputs"):
                    continue

                local_path = Path(relative_path)
                files_to_download.append((s3_key, local_path, size))
    except Exception as e:
        print(f"Error listing backup contents: {e}")
        return 1

    if not files_to_download:
        print("No files to download.")
        return 0

    total_bytes = sum(f[2] for f in files_to_download)
    print(f"Found {len(files_to_download)} files ({total_bytes / (1024**3):.2f} GB)")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for _, local_path, size in files_to_download[:15]:
            print(f"  {local_path} ({size / (1024 * 1024):.1f} MB)")
        if len(files_to_download) > 15:
            print(f"  ... and {len(files_to_download) - 15} more files")
        return 0

    # Download with progress
    downloaded_bytes = 0
    downloaded_files = 0
    errors = []
    start_time = time.time()

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for s3_key, local_path, size in files_to_download:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, s3_key, str(local_path))
                downloaded_files += 1
                downloaded_bytes += size
                pbar.update(size)
            except Exception as e:
                errors.append((local_path, str(e)))
                pbar.update(size)
                if not continue_on_error:
                    print(f"\nError downloading {s3_key}: {e}")
                    return 1

    elapsed = time.time() - start_time
    speed_mbps = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Downloaded {downloaded_files}/{len(files_to_download)} files")
    print(
        f"  Total: {downloaded_bytes / (1024**3):.2f} GB in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)"
    )

    if errors:
        print(f"\n⚠️  {len(errors)} errors:")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    # Show what was downloaded
    print("\nDownloaded contents:")
    if Path("artifacts").exists():
        print(f"  artifacts/: {sum(1 for _ in Path('artifacts').rglob('*') if _.is_file())} files")
    if Path("outputs").exists():
        print(f"  outputs/: {sum(1 for _ in Path('outputs').rglob('*') if _.is_file())} files")

    return 0
