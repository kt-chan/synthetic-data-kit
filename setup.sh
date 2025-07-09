#!/bin/bash

# setup.sh - Install Miniconda, create Conda environment, and use uv for pip packages
# Usage: ./setup.sh [ENV_NAME] [PYTHON_VERSION]

set -e  # Exit on error

# Default parameters
DEFAULT_ENV_NAME="myenv"
DEFAULT_PYTHON_VERSION="3.11"

# Parameter handling
ENV_NAME="${1:-$DEFAULT_ENV_NAME}"
PYTHON_VERSION="${2:-$DEFAULT_PYTHON_VERSION}"

# Validate parameters
validate_parameters() {
    # Check environment name format
    if [[ ! "$ENV_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "ERROR: Invalid environment name '$ENV_NAME'. Use only letters, numbers, hyphens or underscores."
        exit 1
    fi
    
    # Check Python version format
    if [[ ! "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "ERROR: Invalid Python version format '$PYTHON_VERSION'. Use major.minor format (e.g., 3.11)."
        exit 1
    fi
}

# Check for required commands
check_commands() {
    for cmd in wget curl; do
        if ! command -v $cmd &> /dev/null; then
            echo "Installing required system package: $cmd"
            sudo apt install -y $cmd
        fi
    done
}

# Main setup functions
install_miniconda() {
    MINICONDA_PATH="$HOME/miniconda3"
    
    if [ ! -d "$MINICONDA_PATH" ]; then
        echo "=== Installing Miniconda ==="
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $MINICONDA_PATH
        rm miniconda.sh
        
        # Initialize Conda
        eval "$($MINICONDA_PATH/bin/conda shell.bash hook)"
        conda init --all --quiet
        conda config --set auto_activate_base false
        echo "Miniconda installed to $MINICONDA_PATH"
    else
        eval "$($MINICONDA_PATH/bin/conda shell.bash hook)"
        echo "Miniconda already exists at $MINICONDA_PATH"
    fi
}

install_uv() {
    if ! command -v uv &> /dev/null; then
        echo "=== Installing uv ==="
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc
        echo "uv installed successfully"
    else
        echo "uv already installed"
    fi
}

create_conda_env() {
    echo "=== Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION ==="
    
    if conda env list | grep -qw "$ENV_NAME"; then
        echo "Environment '$ENV_NAME' already exists. Skipping creation."
        return
    fi

    if ! conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y; then
        echo "ERROR: Failed to create environment. Possible reasons:"
        echo "  - Invalid Python version '$PYTHON_VERSION' (available versions: $(conda search python | grep -o '^[0-9]\+\.[0-9]\+' | sort -u | tr '\n' ' '))"
        echo "  - Environment name conflict"
        exit 1
    fi
}

install_packages() {
    echo "=== Installing packages in '$ENV_NAME' with uv ==="
    conda activate "$ENV_NAME"
    
    # Core packages
    uv pip install numpy pandas scipy matplotlib jupyter
    
    # Additional validation
    echo -e "\n=== Installation verification ==="
    echo "Python path: $(which python)"
    echo "Python version: $(python --version)"
    echo "uv version: $(uv --version)"
    echo "Installed packages:"
    uv pip list
}

# Main execution
validate_parameters
check_commands
install_miniconda
install_uv
create_conda_env
install_packages

echo -e "\n=== Setup completed successfully! ==="
echo "To activate environment:"
echo "  conda activate $ENV_NAME"
echo "To deactivate:"
echo "  conda deactivate"
