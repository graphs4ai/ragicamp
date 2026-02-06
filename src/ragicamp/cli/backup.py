"""Backblaze B2 backup and download operations."""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default number of parallel workers for upload/download
DEFAULT_MAX_WORKERS = 12


@dataclass
class TransferProgress:
    """Thread-safe progress counters for parallel file transfers."""

    total_bytes: int = 0
    total_files: int = 0
    errors: list[tuple[Path, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _early_exit: bool = False

    def record_success(self, file_size: int) -> None:
        with self._lock:
            self.total_files += 1
            self.total_bytes += file_size

    def record_error(self, path: Path, error: str) -> None:
        with self._lock:
            self.errors.append((path, error))

    def request_early_exit(self) -> None:
        with self._lock:
            self._early_exit = True

    @property
    def should_exit(self) -> bool:
        return self._early_exit


def _parallel_transfer(
    files: list[tuple],
    transfer_fn: Callable,
    total_bytes: int,
    desc: str,
    max_workers: int,
    continue_on_error: bool,
) -> TransferProgress:
    """Run a parallel file transfer with progress bar and error handling.

    Args:
        files: List of tuples to pass to transfer_fn.
        transfer_fn: Callable(*file_tuple) -> (success: bool, error: str | None).
        total_bytes: Total bytes for the progress bar.
        desc: Progress bar description (e.g. "Uploading", "Downloading").
        max_workers: Thread pool size.
        continue_on_error: If False, abort on first error.

    Returns:
        TransferProgress with final counters and any errors.
    """
    from tqdm import tqdm

    progress = TransferProgress()

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(transfer_fn, *item): item for item in files
            }

            for future in as_completed(future_to_file):
                item = future_to_file[future]
                file_size = item[-1]  # last element is always size
                success, error = future.result()

                if success:
                    progress.record_success(file_size)
                else:
                    progress.record_error(Path(str(item[0])), error or "Unknown error")
                    if not continue_on_error:
                        progress.request_early_exit()
                        print(f"\nError: {error}")
                pbar.update(file_size)

                if progress.should_exit and not continue_on_error:
                    for f in future_to_file:
                        f.cancel()
                    break

    return progress

# Files/patterns to skip during backup
SKIP_PATTERNS = {
    ".tmp", ".temp", ".pyc", ".pyo", "__pycache__", ".git",
    ".DS_Store", "Thumbs.db", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "*.log", "checkpoint.json", ".lock",
}


def should_skip_file(path: Path) -> bool:
    """Check if file should be skipped during backup."""
    name = path.name
    if name.startswith("."):
        return True
    for pattern in SKIP_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern or pattern in str(path):
            return True
    return False


def get_b2_client():
    """Get B2 S3 client and credentials.

    Returns:
        Tuple of (s3_client, endpoint, key_id, app_key) or (None, None, None, None) on error.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        logger.error("boto3 not installed. Install with: uv pip install boto3")
        return None, None, None, None

    key_id = os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    app_key = os.environ.get("B2_APP_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")

    if not key_id or not app_key:
        logger.error(
            "Backblaze credentials not set. Set environment variables:\n"
            "  export B2_KEY_ID='your-key-id'\n"
            "  export B2_APP_KEY='your-application-key'\n"
            "  export B2_ENDPOINT='https://s3.us-east-005.backblazeb2.com'  # optional"
        )
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
    except Exception:
        logger.exception("Failed to list backups")
        return []


def backup(
    dirs_to_backup: list[Path],
    bucket: str,
    prefix: str,
    dry_run: bool = False,
    continue_on_error: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    sync: bool = False,
) -> int:
    """Upload directories to B2 with parallel uploads.

    Args:
        dirs_to_backup: List of directories to backup
        bucket: B2 bucket name
        prefix: S3 key prefix
        dry_run: If True, only preview files
        continue_on_error: If True, continue on upload errors
        max_workers: Number of parallel upload threads (default 12)
        sync: If True, only upload new/modified files (compare by size)

    Returns:
        Exit code (0 on success)
    """
    from tqdm import tqdm

    s3, _, _, _ = get_b2_client()
    if s3 is None:
        return 1

    print(f"Backing up to: s3://{bucket}/{prefix}/")
    print(f"Source directories: {', '.join(str(d) for d in dirs_to_backup)}")
    print(f"Parallel workers: {max_workers}")
    if sync:
        print("Sync mode: only uploading new/modified files")

    # If sync mode, get existing files from B2
    existing_files: dict[str, int] = {}
    if sync:
        print("Checking existing files in B2...")
        try:
            paginator = s3.get_paginator("list_objects_v2")
            page_count = 0
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                page_count += 1
                contents = page.get("Contents", [])
                for obj in contents:
                    existing_files[obj["Key"]] = obj["Size"]
            print(f"Found {len(existing_files)} existing files in backup (scanned {page_count} pages)")
        except Exception as e:
            print(f"Warning: Could not list existing files: {e}")
            print("Proceeding without sync (will upload all files)")

    # Collect files to upload
    files_to_upload = []
    skipped_files = 0
    skipped_temp = 0
    for backup_dir in dirs_to_backup:
        for root, dirs, files in os.walk(backup_dir):
            # Filter directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", ".git")]
            for file in files:
                local_path = Path(root) / file
                
                # Skip temp/cache files
                if should_skip_file(local_path):
                    skipped_temp += 1
                    continue
                
                relative_path = local_path.relative_to(backup_dir.parent)
                s3_key = f"{prefix}/{relative_path}"
                file_size = local_path.stat().st_size
                
                # In sync mode, skip if file exists with same size
                if sync and s3_key in existing_files:
                    if existing_files[s3_key] == file_size:
                        skipped_files += 1
                        continue
                
                files_to_upload.append((local_path, s3_key, file_size))

    if not files_to_upload:
        if sync and skipped_files > 0:
            print(f"All {skipped_files} files already up to date. Nothing to upload.")
        else:
            print("No files to upload.")
        if skipped_temp > 0:
            print(f"  (Skipped {skipped_temp} temp/cache files)")
        return 0

    total_bytes = sum(f[2] for f in files_to_upload)
    print(f"Found {len(files_to_upload)} files to upload ({total_bytes / (1024**3):.2f} GB)")
    if sync and skipped_files > 0:
        print(f"  Skipped {skipped_files} unchanged files")
    if skipped_temp > 0:
        print(f"  Skipped {skipped_temp} temp/cache files")

    if dry_run:
        print("\n[DRY RUN] Would upload:")
        for local_path, s3_key, size in files_to_upload[:10]:
            size_mb = size / (1024 * 1024)
            print(f"  {s3_key} ({size_mb:.1f} MB)")
        if len(files_to_upload) > 10:
            print(f"  ... and {len(files_to_upload) - 10} more files")
        return 0

    def upload_file(local_path: Path, s3_key: str, file_size: int) -> tuple[bool, str | None]:
        """Upload a single file. Returns (success, error_message)."""
        try:
            s3.upload_file(str(local_path), bucket, s3_key)
            return True, None
        except Exception as e:
            return False, str(e)

    start_time = time.time()

    progress = _parallel_transfer(
        files=files_to_upload,
        transfer_fn=upload_file,
        total_bytes=total_bytes,
        desc="Uploading",
        max_workers=max_workers,
        continue_on_error=continue_on_error,
    )

    elapsed = time.time() - start_time
    speed_mbps = (progress.total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Uploaded {progress.total_files}/{len(files_to_upload)} files to s3://{bucket}/{prefix}/")
    print(f"  Total: {progress.total_bytes / (1024**3):.2f} GB in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)")

    if progress.errors:
        print(f"\n⚠️  {len(progress.errors)} errors:")
        for path, err in progress.errors[:5]:
            print(f"  {path}: {err}")
        if not continue_on_error:
            return 1

    return 0


def download(
    bucket: str,
    backup_name: Optional[str] = None,
    artifacts_only: bool = False,
    outputs_only: bool = False,
    indexes_only: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    skip_existing: bool = False,
    migrate_indexes: bool = True,
) -> int:
    """Download backup from B2 with parallel downloads.

    Args:
        bucket: B2 bucket name
        backup_name: Specific backup to download (None = most recent)
        artifacts_only: Only download artifacts/
        outputs_only: Only download outputs/
        indexes_only: Only download artifacts/indexes/
        dry_run: If True, only preview files
        continue_on_error: If True, continue on download errors
        max_workers: Number of parallel download threads (default 12)
        skip_existing: Skip files that already exist locally with same size
        migrate_indexes: Run index migration after download (default True)

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
    print(f"Parallel workers: {max_workers}")
    if skip_existing:
        print("Skip existing: will skip files that already exist with same size")

    # List all files in the backup
    files_to_download = []
    skipped_existing = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                size = obj["Size"]
                relative_path = s3_key[len(prefix) + 1 :]
                if not relative_path:
                    continue

                # Filter by type
                if indexes_only and not relative_path.startswith("artifacts/indexes"):
                    continue
                if artifacts_only and not relative_path.startswith("artifacts"):
                    continue
                if outputs_only and not relative_path.startswith("outputs"):
                    continue

                local_path = Path(relative_path)
                
                # Skip existing files with same size
                if skip_existing and local_path.exists():
                    if local_path.stat().st_size == size:
                        skipped_existing += 1
                        continue
                
                files_to_download.append((s3_key, local_path, size))
    except Exception as e:
        print(f"Error listing backup contents: {e}")
        return 1

    if not files_to_download:
        if skipped_existing > 0:
            print(f"All {skipped_existing} files already exist. Nothing to download.")
        else:
            print("No files to download.")
        return 0

    total_bytes = sum(f[2] for f in files_to_download)
    print(f"Found {len(files_to_download)} files ({total_bytes / (1024**3):.2f} GB)")
    if skipped_existing > 0:
        print(f"  Skipped {skipped_existing} existing files")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for _, local_path, size in files_to_download[:15]:
            print(f"  {local_path} ({size / (1024 * 1024):.1f} MB)")
        if len(files_to_download) > 15:
            print(f"  ... and {len(files_to_download) - 15} more files")
        return 0

    # Pre-create all directories
    dirs_to_create = set()
    for _, local_path, _ in files_to_download:
        dirs_to_create.add(local_path.parent)
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    def download_file(s3_key: str, local_path: Path, size: int) -> tuple[bool, str | None]:
        """Download a single file. Returns (success, error_message)."""
        try:
            s3.download_file(bucket, s3_key, str(local_path))
            return True, None
        except Exception as e:
            return False, str(e)

    start_time = time.time()

    progress = _parallel_transfer(
        files=files_to_download,
        transfer_fn=download_file,
        total_bytes=total_bytes,
        desc="Downloading",
        max_workers=max_workers,
        continue_on_error=continue_on_error,
    )

    elapsed = time.time() - start_time
    speed_mbps = (progress.total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0

    print(f"\n✓ Downloaded {progress.total_files}/{len(files_to_download)} files")
    print(f"  Total: {progress.total_bytes / (1024**3):.2f} GB in {elapsed:.1f}s ({speed_mbps:.1f} MB/s)")

    if progress.errors:
        print(f"\n⚠️  {len(progress.errors)} errors:")
        for path, err in progress.errors[:5]:
            print(f"  {path}: {err}")
        if not continue_on_error:
            return 1

    # Show what was downloaded
    print("\nDownloaded contents:")
    if Path("artifacts").exists():
        print(f"  artifacts/: {sum(1 for _ in Path('artifacts').rglob('*') if _.is_file())} files")
    if Path("outputs").exists():
        print(f"  outputs/: {sum(1 for _ in Path('outputs').rglob('*') if _.is_file())} files")

    # Auto-migrate indexes after download
    if migrate_indexes and (artifacts_only or indexes_only or not outputs_only):
        indexes_dir = Path("artifacts/indexes")
        if indexes_dir.exists():
            print("\n🔄 Migrating indexes to new format...")
            try:
                from ragicamp.cli.commands import cmd_migrate_indexes
                import argparse
                migrate_args = argparse.Namespace(index_name=None, dry_run=False)
                cmd_migrate_indexes(migrate_args)
            except Exception as e:
                print(f"  Warning: Index migration failed: {e}")
                print("  Run manually: uv run ragicamp migrate-indexes")

    return 0
