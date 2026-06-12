#!/bin/bash

# Default values
VISUALIZE=""
SPEED=2.0

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v           Enable visualization"
    echo "  -s SPEED     Set speed (0.1 to 3.0, default: 2.0)"
    echo "  -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Run with default settings"
    echo "  $0 -v              # Run with visualization"
    echo "  $0 -s 1.5          # Run with speed 1.5"
    echo "  $0 -v -s 2.5       # Run with visualization and speed 2.5"
}

# Function to validate and clip speed
validate_speed() {
    local speed=$1
    # Check if it's a valid number
    if ! [[ $speed =~ ^[0-9]*\.?[0-9]+$ ]]; then
        echo "Error: Speed must be a number" >&2
        exit 1
    fi
    # Clip speed between 0.1 and 3.0
    if (( $(echo "$speed < 0.1" | bc -l) )); then
        speed=0.1
        echo "Warning: Speed clipped to minimum value 0.1" >&2
    elif (( $(echo "$speed > 3.0" | bc -l) )); then
        speed=3.0
        echo "Warning: Speed clipped to maximum value 3.0" >&2
    fi
    echo $speed
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v)
            VISUALIZE="--visualize"
            shift
            ;;
        -s)
            if [[ -n $2 && $2 != -* ]]; then
                SPEED=$(validate_speed $2)
                shift 2
            else
                echo "Error: -s requires a speed value" >&2
                exit 1
            fi
            ;;
        -h)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the python command with the configured options
python "$SCRIPT_DIR/replay_trajectory.py" "$SCRIPT_DIR/data/vega-1_dance.npz" --hz 500 --smooth 0.1 --vel-smooth 0.5 --speed $SPEED $VISUALIZE
