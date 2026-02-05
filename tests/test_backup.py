"""Tests for backup CLI functionality."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBackupSync:
    """Test backup sync functionality."""

    def test_should_skip_file_patterns(self):
        """Test that skip patterns work correctly."""
        from ragicamp.cli.backup import should_skip_file

        # Should skip
        assert should_skip_file(Path(".hidden_file")) is True
        assert should_skip_file(Path("file.tmp")) is True
        assert should_skip_file(Path("file.pyc")) is True
        assert should_skip_file(Path("__pycache__/module.py")) is True
        assert should_skip_file(Path(".DS_Store")) is True
        assert should_skip_file(Path("checkpoint.json")) is True

        # Should not skip
        assert should_skip_file(Path("model.safetensors")) is False
        assert should_skip_file(Path("predictions.json")) is False
        assert should_skip_file(Path("index.faiss")) is False

    @patch("ragicamp.cli.backup.get_b2_client")
    def test_list_backups_returns_sorted(self, mock_get_client):
        """Test that list_backups returns backups sorted newest first."""
        from ragicamp.cli.backup import list_backups

        # Mock S3 client
        mock_s3 = MagicMock()
        mock_get_client.return_value = (mock_s3, "endpoint", "key", "secret")

        # Mock response with unsorted prefixes
        mock_s3.list_objects_v2.return_value = {
            "CommonPrefixes": [
                {"Prefix": "ragicamp-backup/20260101-120000/"},
                {"Prefix": "ragicamp-backup/20260204-114504/"},
                {"Prefix": "ragicamp-backup/20260102-080000/"},
            ]
        }

        backups = list_backups("test-bucket", limit=10)

        # Should be sorted newest first
        assert backups == ["20260204-114504", "20260102-080000", "20260101-120000"]

    @patch("ragicamp.cli.backup.get_b2_client")
    def test_list_backups_respects_limit(self, mock_get_client):
        """Test that list_backups respects the limit parameter."""
        from ragicamp.cli.backup import list_backups

        mock_s3 = MagicMock()
        mock_get_client.return_value = (mock_s3, "endpoint", "key", "secret")

        mock_s3.list_objects_v2.return_value = {
            "CommonPrefixes": [
                {"Prefix": "ragicamp-backup/20260101-120000/"},
                {"Prefix": "ragicamp-backup/20260204-114504/"},
                {"Prefix": "ragicamp-backup/20260102-080000/"},
            ]
        }

        backups = list_backups("test-bucket", limit=2)
        assert len(backups) == 2
        assert backups == ["20260204-114504", "20260102-080000"]

    @patch("ragicamp.cli.backup.get_b2_client")
    def test_list_backups_empty_bucket(self, mock_get_client):
        """Test list_backups with no backups."""
        from ragicamp.cli.backup import list_backups

        mock_s3 = MagicMock()
        mock_get_client.return_value = (mock_s3, "endpoint", "key", "secret")

        mock_s3.list_objects_v2.return_value = {"CommonPrefixes": []}

        backups = list_backups("test-bucket")
        assert backups == []

    @patch("ragicamp.cli.backup.get_b2_client")
    def test_list_backups_client_error(self, mock_get_client):
        """Test list_backups when client initialization fails."""
        from ragicamp.cli.backup import list_backups

        mock_get_client.return_value = (None, None, None, None)

        backups = list_backups("test-bucket")
        assert backups == []


class TestCmdBackupLatest:
    """Test cmd_backup with --latest flag."""

    @patch("ragicamp.cli.backup.backup")
    @patch("ragicamp.cli.backup.list_backups")
    def test_latest_flag_uses_most_recent_backup(self, mock_list_backups, mock_backup):
        """Test that --latest flag uses the most recent backup prefix."""
        from ragicamp.cli.commands import cmd_backup

        mock_list_backups.return_value = ["20260204-114504"]
        mock_backup.return_value = 0

        args = argparse.Namespace(
            path=Path("outputs"),
            bucket="test-bucket",
            prefix=None,
            dry_run=False,
            continue_on_error=False,
            workers=12,
            sync=True,
            latest=True,
        )

        result = cmd_backup(args)

        assert result == 0
        mock_list_backups.assert_called_once_with("test-bucket", limit=1)
        mock_backup.assert_called_once()

        # Check that the prefix uses the latest backup
        call_kwargs = mock_backup.call_args[1]
        assert call_kwargs["prefix"] == "ragicamp-backup/20260204-114504"
        assert call_kwargs["sync"] is True

    @patch("ragicamp.cli.backup.list_backups")
    def test_latest_flag_fails_when_no_backups(self, mock_list_backups, capsys):
        """Test that --latest flag fails gracefully when no backups exist."""
        from ragicamp.cli.commands import cmd_backup

        mock_list_backups.return_value = []

        args = argparse.Namespace(
            path=Path("outputs"),
            bucket="test-bucket",
            prefix=None,
            dry_run=False,
            continue_on_error=False,
            workers=12,
            sync=True,
            latest=True,
        )

        result = cmd_backup(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No existing backups found" in captured.out

    @patch("ragicamp.cli.backup.backup")
    @patch("ragicamp.cli.backup.list_backups")
    def test_prefix_overrides_latest(self, mock_list_backups, mock_backup):
        """Test that explicit --prefix is used even if --latest is set."""
        from ragicamp.cli.commands import cmd_backup

        mock_backup.return_value = 0

        args = argparse.Namespace(
            path=Path("outputs"),
            bucket="test-bucket",
            prefix="ragicamp-backup/custom-prefix",
            dry_run=False,
            continue_on_error=False,
            workers=12,
            sync=True,
            latest=False,  # Not using latest
        )

        result = cmd_backup(args)

        assert result == 0
        # list_backups should not be called when --latest is not set
        mock_list_backups.assert_not_called()

        call_kwargs = mock_backup.call_args[1]
        assert call_kwargs["prefix"] == "ragicamp-backup/custom-prefix"


class TestBackupSyncLogic:
    """Test the sync logic in backup function."""

    @patch("ragicamp.cli.backup.get_b2_client")
    def test_sync_skips_existing_files_with_same_size(self, mock_get_client, temp_dir):
        """Test that sync mode skips files with matching size in B2."""
        from ragicamp.cli.backup import backup

        # Create mock S3 client
        mock_s3 = MagicMock()
        mock_get_client.return_value = (mock_s3, "endpoint", "key", "secret")

        # Create test file
        test_file = temp_dir / "artifacts" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test content")
        file_size = test_file.stat().st_size

        # Mock paginator to return existing file with same size
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "test-prefix/artifacts/test.txt", "Size": file_size}
                ]
            }
        ]

        # Run backup in dry-run mode to avoid actual uploads
        result = backup(
            dirs_to_backup=[temp_dir / "artifacts"],
            bucket="test-bucket",
            prefix="test-prefix",
            dry_run=True,
            sync=True,
        )

        assert result == 0
        # With sync mode, the file should be skipped (no files to upload)

    @patch("ragicamp.cli.backup.get_b2_client")
    def test_sync_uploads_files_with_different_size(self, mock_get_client, temp_dir, capsys):
        """Test that sync mode uploads files with different size."""
        from ragicamp.cli.backup import backup

        mock_s3 = MagicMock()
        mock_get_client.return_value = (mock_s3, "endpoint", "key", "secret")

        # Create test file
        test_file = temp_dir / "artifacts" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("new content - different size")

        # Mock paginator to return existing file with different size
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "test-prefix/artifacts/test.txt", "Size": 10}  # Different size
                ]
            }
        ]

        result = backup(
            dirs_to_backup=[temp_dir / "artifacts"],
            bucket="test-bucket",
            prefix="test-prefix",
            dry_run=True,
            sync=True,
        )

        assert result == 0
        captured = capsys.readouterr()
        # File should be in the upload list since size differs
        assert "1 files to upload" in captured.out

    @patch("ragicamp.cli.backup.get_b2_client")
    def test_sync_uploads_new_files(self, mock_get_client, temp_dir, capsys):
        """Test that sync mode uploads files that don't exist in B2."""
        from ragicamp.cli.backup import backup

        mock_s3 = MagicMock()
        mock_get_client.return_value = (mock_s3, "endpoint", "key", "secret")

        # Create test file
        test_file = temp_dir / "artifacts" / "new_file.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("new file content")

        # Mock paginator to return empty (no existing files)
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": []}]

        result = backup(
            dirs_to_backup=[temp_dir / "artifacts"],
            bucket="test-bucket",
            prefix="test-prefix",
            dry_run=True,
            sync=True,
        )

        assert result == 0
        captured = capsys.readouterr()
        # New file should be in the upload list
        assert "1 files to upload" in captured.out


class TestBackupCLIArguments:
    """Test CLI argument parsing for backup command."""

    def test_sync_argument_exists(self):
        """Test that --sync argument is properly configured."""
        from ragicamp.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["backup", "outputs/", "--sync"])

        assert args.sync is True
        assert args.command == "backup"

    def test_sync_short_form(self):
        """Test that -s shorthand works for --sync."""
        from ragicamp.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["backup", "outputs/", "-s"])

        assert args.sync is True

    def test_latest_argument_exists(self):
        """Test that --latest argument is properly configured."""
        from ragicamp.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["backup", "outputs/", "--latest"])

        assert args.latest is True

    def test_latest_short_form(self):
        """Test that -l shorthand works for --latest."""
        from ragicamp.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["backup", "outputs/", "-l"])

        assert args.latest is True

    def test_combined_flags(self):
        """Test that --sync and --latest can be combined."""
        from ragicamp.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(["backup", "outputs/", "--latest", "--sync", "--dry-run"])

        assert args.sync is True
        assert args.latest is True
        assert args.dry_run is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
