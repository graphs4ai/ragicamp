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
#
# Optional environment variables:
#   B2_ENDPOINT    - B2 endpoint URL (default: https://s3.us-east-005.backblazeb2.com)
#   B2_BUCKET      - Bucket name (default: masters-bucket)

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

# Default values (must match backup command defaults)
BUCKET="${B2_BUCKET:-masters-bucket}"
ENDPOINT="${B2_ENDPOINT:-https://s3.us-east-005.backblazeb2.com}"
BACKUP_PREFIX="ragicamp-backup"
DRY_RUN=false
LIST_ONLY=false
SPECIFIC_BACKUP=""
ARTIFACTS_ONLY=false
OUTPUTS_ONLY=false

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
        --artifacts-only)
            ARTIFACTS_ONLY=true
            shift
            ;;
        --outputs-only)
            OUTPUTS_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Download artifacts and indexes from Backblaze B2 backup."
            echo ""
            echo "Options:"
            echo "  --list, -l              List available backups"
            echo "  --backup, -b <name>     Download specific backup (e.g., 20240115-143022)"
            echo "  --dry-run, -n           Show what would be downloaded without downloading"
            echo "  --bucket <name>         Override bucket name (default: masters-bucket)"
            echo "  --artifacts-only        Only download artifacts/ (indexes, retrievers)"
            echo "  --outputs-only          Only download outputs/ (experiment results)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  B2_KEY_ID      Required: Backblaze B2 Key ID"
            echo "  B2_APP_KEY     Required: Backblaze B2 Application Key"
            echo "  B2_ENDPOINT    Optional: B2 endpoint (default: https://s3.us-east-005.backblazeb2.com)"
            echo "  B2_BUCKET      Optional: Bucket name (default: masters-bucket)"
            echo ""
            echo "Examples:"
            echo "  $0 --list                      # See available backups"
            echo "  $0                             # Download latest backup"
            echo "  $0 --artifacts-only            # Only download indexes"
            echo "  $0 --backup 20240115-143022    # Download specific backup"
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac
done

# Check credentials
if [ -z "$B2_KEY_ID" ] || [ -z "$B2_APP_KEY" ]; then
    error "Backblaze credentials not set.

Set environment variables:
  export B2_KEY_ID='your-key-id'
  export B2_APP_KEY='your-application-key'

Or source your credentials file:
  source setup_env.sh"
fi

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    error "AWS CLI not installed. Install with:
  uv pip install awscli
Or:
  pip install awscli"
fi

# Configure AWS CLI for B2
export AWS_ACCESS_KEY_ID="$B2_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$B2_APP_KEY"

# Test connection
log "Connecting to B2..."
if ! aws s3 ls "s3://$BUCKET/" --endpoint-url "$ENDPOINT" &>/dev/null; then
    error "Cannot connect to B2. Check your credentials and bucket name.
Bucket: $BUCKET
Endpoint: $ENDPOINT"
fi

# Function to list backups
list_backups() {
    log "Listing backups in s3://$BUCKET/$BACKUP_PREFIX/..."
    echo ""
    
    # List all backup timestamps
    local backups=$(aws s3 ls "s3://$BUCKET/$BACKUP_PREFIX/" --endpoint-url "$ENDPOINT" 2>/dev/null | \
        grep "PRE" | \
        awk '{print $2}' | \
        sed 's/\///' | \
        sort -r)
    
    if [ -z "$backups" ]; then
        warn "No backups found in s3://$BUCKET/$BACKUP_PREFIX/"
        return 1
    fi
    
    echo "Available backups (most recent first):"
    echo "======================================="
    echo "$backups" | head -20
    
    local count=$(echo "$backups" | wc -l)
    if [ "$count" -gt 20 ]; then
        echo "... and $((count - 20)) more"
    fi
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
    local s3_prefix="s3://$BUCKET/$BACKUP_PREFIX/$backup_name"
    
    log "Downloading backup: $backup_name"
    info "Source: $s3_prefix"
    
    # Check backup exists
    if ! aws s3 ls "$s3_prefix/" --endpoint-url "$ENDPOINT" &>/dev/null; then
        error "Backup not found: $backup_name
        
Available backups:
$(get_latest_backup | head -5)"
    fi
    
    # Count files
    local file_info=$(aws s3 ls "$s3_prefix/" --endpoint-url "$ENDPOINT" --recursive --summarize 2>/dev/null | tail -2)
    info "$file_info"
    
    if [ "$DRY_RUN" = true ]; then
        echo ""
        warn "[DRY RUN] Would download:"
        
        if [ "$ARTIFACTS_ONLY" = false ] && [ "$OUTPUTS_ONLY" = false ]; then
            aws s3 ls "$s3_prefix/" --endpoint-url "$ENDPOINT" --recursive 2>/dev/null | head -30
        elif [ "$ARTIFACTS_ONLY" = true ]; then
            aws s3 ls "$s3_prefix/artifacts/" --endpoint-url "$ENDPOINT" --recursive 2>/dev/null | head -30
        else
            aws s3 ls "$s3_prefix/outputs/" --endpoint-url "$ENDPOINT" --recursive 2>/dev/null | head -30
        fi
        echo ""
        echo "(showing first 30 files)"
        return 0
    fi
    
    # Download based on options
    if [ "$ARTIFACTS_ONLY" = true ]; then
        log "Downloading artifacts/ only..."
        aws s3 sync "$s3_prefix/artifacts/" "./artifacts/" --endpoint-url "$ENDPOINT"
    elif [ "$OUTPUTS_ONLY" = true ]; then
        log "Downloading outputs/ only..."
        aws s3 sync "$s3_prefix/outputs/" "./outputs/" --endpoint-url "$ENDPOINT"
    else
        # Download both
        log "Downloading artifacts/..."
        aws s3 sync "$s3_prefix/artifacts/" "./artifacts/" --endpoint-url "$ENDPOINT" 2>/dev/null || true
        
        log "Downloading outputs/..."
        aws s3 sync "$s3_prefix/outputs/" "./outputs/" --endpoint-url "$ENDPOINT" 2>/dev/null || true
    fi
    
    echo ""
    log "✓ Download complete!"
    
    # Show what was downloaded
    echo ""
    if [ -d "artifacts" ] && [ "$OUTPUTS_ONLY" = false ]; then
        info "artifacts/ contents:"
        du -sh artifacts/* 2>/dev/null | head -10 || ls -la artifacts/ 2>/dev/null | head -10
    fi
    
    if [ -d "outputs" ] && [ "$ARTIFACTS_ONLY" = false ]; then
        echo ""
        info "outputs/ contents:"
        du -sh outputs/* 2>/dev/null | head -10 || ls -la outputs/ 2>/dev/null | head -10
    fi
}

# Main logic
if [ "$LIST_ONLY" = true ]; then
    list_backups
    echo ""
    echo "To download the most recent backup:"
    echo "  $0"
    echo ""
    echo "To download a specific backup:"
    echo "  $0 --backup <backup-name>"
    exit 0
fi

# Determine which backup to download
if [ -n "$SPECIFIC_BACKUP" ]; then
    BACKUP_NAME="$SPECIFIC_BACKUP"
else
    log "Finding most recent backup..."
    BACKUP_NAME=$(get_latest_backup)
    
    if [ -z "$BACKUP_NAME" ]; then
        error "No backups found in s3://$BUCKET/$BACKUP_PREFIX/
        
Check that backups exist:
  aws s3 ls s3://$BUCKET/$BACKUP_PREFIX/ --endpoint-url $ENDPOINT"
    fi
    
    log "Most recent backup: $BACKUP_NAME"
fi

# Download the backup
download_backup "$BACKUP_NAME"
