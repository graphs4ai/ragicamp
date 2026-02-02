#!/bin/bash
# Download artifacts and indexes from Backblaze B2 backup
#
# Usage:
#   ./download_backup.sh                    # Download most recent backup
#   ./download_backup.sh --list             # List available backups
#   ./download_backup.sh --backup <prefix>  # Download specific backup
#   ./download_backup.sh --dry-run          # Show what would be downloaded
#
# Required environment variables:
#   B2_KEY_ID      - Backblaze B2 Key ID
#   B2_APP_KEY     - Backblaze B2 Application Key
#   B2_ENDPOINT    - (optional) B2 endpoint URL
#   B2_BUCKET      - (optional) Bucket name (default: ragicamp-backups)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[download]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
error() { echo -e "${RED}[error]${NC} $1"; exit 1; }
info() { echo -e "${BLUE}[info]${NC} $1"; }

# Default values
BUCKET="${B2_BUCKET:-ragicamp-backups}"
ENDPOINT="${B2_ENDPOINT:-https://s3.us-east-005.backblazeb2.com}"
BACKUP_PREFIX="ragicamp-backup"
DRY_RUN=false
LIST_ONLY=false
SPECIFIC_BACKUP=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)
            LIST_ONLY=true
            shift
            ;;
        --backup|-b)
            SPECIFIC_BACKUP="$2"
            shift 2
            ;;
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list, -l              List available backups"
            echo "  --backup, -b <prefix>   Download specific backup (e.g., 20240115-143022)"
            echo "  --dry-run, -n           Show what would be downloaded without downloading"
            echo "  --bucket <name>         Override bucket name (default: ragicamp-backups)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  B2_KEY_ID      Required: Backblaze B2 Key ID"
            echo "  B2_APP_KEY     Required: Backblaze B2 Application Key"
            echo "  B2_ENDPOINT    Optional: B2 endpoint (default: https://s3.us-east-005.backblazeb2.com)"
            echo "  B2_BUCKET      Optional: Bucket name (default: ragicamp-backups)"
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac
done

# Check credentials
if [ -z "$B2_KEY_ID" ] || [ -z "$B2_APP_KEY" ]; then
    error "Backblaze credentials not set. Set B2_KEY_ID and B2_APP_KEY environment variables.
    
Example:
  source setup_env.sh  # If you have credentials in setup_env.sh
  
Or:
  export B2_KEY_ID='your-key-id'
  export B2_APP_KEY='your-application-key'"
fi

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    error "AWS CLI not installed. Install with: pip install awscli
Or: sudo apt install awscli"
fi

# Configure AWS CLI for B2
export AWS_ACCESS_KEY_ID="$B2_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$B2_APP_KEY"

# Function to list backups
list_backups() {
    log "Listing backups in s3://$BUCKET/$BACKUP_PREFIX/..."
    
    # List all prefixes under ragicamp-backup/
    aws s3 ls "s3://$BUCKET/$BACKUP_PREFIX/" --endpoint-url "$ENDPOINT" 2>/dev/null | \
        grep "PRE" | \
        awk '{print $2}' | \
        sed 's/\///' | \
        sort -r | \
        head -20
}

# Function to get most recent backup
get_latest_backup() {
    aws s3 ls "s3://$BUCKET/$BACKUP_PREFIX/" --endpoint-url "$ENDPOINT" 2>/dev/null | \
        grep "PRE" | \
        awk '{print $2}' | \
        sed 's/\///' | \
        sort -r | \
        head -1
}

# Function to download backup
download_backup() {
    local backup_name="$1"
    local s3_path="s3://$BUCKET/$BACKUP_PREFIX/$backup_name"
    
    log "Downloading backup: $backup_name"
    info "Source: $s3_path"
    
    # Check what's in the backup
    log "Checking backup contents..."
    local contents=$(aws s3 ls "$s3_path/" --endpoint-url "$ENDPOINT" --recursive 2>/dev/null | head -5)
    
    if [ -z "$contents" ]; then
        error "Backup not found or empty: $backup_name"
    fi
    
    # Count files and size
    local file_count=$(aws s3 ls "$s3_path/" --endpoint-url "$ENDPOINT" --recursive 2>/dev/null | wc -l)
    local total_size=$(aws s3 ls "$s3_path/" --endpoint-url "$ENDPOINT" --recursive 2>/dev/null | awk '{sum += $3} END {print sum}')
    local size_gb=$(echo "scale=2; $total_size / 1073741824" | bc 2>/dev/null || echo "unknown")
    
    info "Found $file_count files (~${size_gb} GB)"
    
    if [ "$DRY_RUN" = true ]; then
        warn "[DRY RUN] Would download to current directory:"
        aws s3 ls "$s3_path/" --endpoint-url "$ENDPOINT" --recursive 2>/dev/null | head -20
        if [ "$file_count" -gt 20 ]; then
            echo "  ... and $((file_count - 20)) more files"
        fi
        return 0
    fi
    
    # Download artifacts/ if exists
    if aws s3 ls "$s3_path/artifacts/" --endpoint-url "$ENDPOINT" &>/dev/null; then
        log "Downloading artifacts/..."
        aws s3 sync "$s3_path/artifacts/" "./artifacts/" --endpoint-url "$ENDPOINT"
    fi
    
    # Download outputs/ if exists
    if aws s3 ls "$s3_path/outputs/" --endpoint-url "$ENDPOINT" &>/dev/null; then
        log "Downloading outputs/..."
        aws s3 sync "$s3_path/outputs/" "./outputs/" --endpoint-url "$ENDPOINT"
    fi
    
    log "✓ Download complete!"
    
    # Show what was downloaded
    echo ""
    if [ -d "artifacts" ]; then
        info "artifacts/ contents:"
        ls -la artifacts/ 2>/dev/null | head -10
    fi
    
    if [ -d "outputs" ]; then
        echo ""
        info "outputs/ contents:"
        ls -la outputs/ 2>/dev/null | head -10
    fi
}

# Main logic
if [ "$LIST_ONLY" = true ]; then
    echo ""
    echo "Available backups (most recent first):"
    echo "======================================="
    list_backups
    echo ""
    echo "To download a specific backup:"
    echo "  $0 --backup <backup-name>"
    echo ""
    echo "To download the most recent:"
    echo "  $0"
    exit 0
fi

# Determine which backup to download
if [ -n "$SPECIFIC_BACKUP" ]; then
    BACKUP_NAME="$SPECIFIC_BACKUP"
else
    log "Finding most recent backup..."
    BACKUP_NAME=$(get_latest_backup)
    
    if [ -z "$BACKUP_NAME" ]; then
        error "No backups found in s3://$BUCKET/$BACKUP_PREFIX/"
    fi
    
    log "Most recent backup: $BACKUP_NAME"
fi

# Download the backup
download_backup "$BACKUP_NAME"
