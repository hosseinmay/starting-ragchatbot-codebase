#!/bin/bash

# Development quality checks script
# Usage:
#   ./scripts/quality.sh          - Run all quality checks
#   ./scripts/quality.sh format   - Auto-format code with black
#   ./scripts/quality.sh check    - Check formatting without changing files

set -e

cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${YELLOW}=== $1 ===${NC}"
    echo ""
}

run_black_check() {
    print_header "Checking code formatting with black"
    if uv run black --check .; then
        echo -e "${GREEN}All files are properly formatted${NC}"
        return 0
    else
        echo -e "${RED}Some files need formatting. Run './scripts/quality.sh format' to fix.${NC}"
        return 1
    fi
}

run_black_format() {
    print_header "Formatting code with black"
    uv run black .
    echo -e "${GREEN}Code formatting complete${NC}"
}

run_tests() {
    print_header "Running tests"
    uv run pytest
}

case "${1:-all}" in
    format)
        run_black_format
        ;;
    check)
        run_black_check
        ;;
    test)
        run_tests
        ;;
    all)
        run_black_check
        echo ""
        run_tests
        echo ""
        echo -e "${GREEN}All quality checks passed!${NC}"
        ;;
    *)
        echo "Usage: $0 {format|check|test|all}"
        echo ""
        echo "Commands:"
        echo "  format  - Auto-format code with black"
        echo "  check   - Check formatting without changing files"
        echo "  test    - Run pytest"
        echo "  all     - Run all checks (default)"
        exit 1
        ;;
esac
