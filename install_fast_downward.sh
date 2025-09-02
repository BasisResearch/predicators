#!/bin/bash

# Install Fast Downward for predicators
echo "Installing Fast Downward..."

# Create external directory if it doesn't exist
mkdir -p external

# Clone Fast Downward if not already present
if [ ! -d "external/downward" ]; then
    echo "Cloning Fast Downward repository..."
    git clone https://github.com/aibasel/downward.git external/downward
fi

# Build Fast Downward
echo "Building Fast Downward..."
cd external/downward
python build.py

# Get the absolute path
FD_PATH=$(pwd)/fast-downward.py

# Go back to predicators root
cd ../..

# Create environment setup script
cat > setup_fd_env.sh << EOF
#!/bin/bash
export FD_EXEC_PATH=$FD_PATH
echo "Fast Downward path set to: $FD_PATH"
EOF

chmod +x setup_fd_env.sh

echo "Fast Downward installed successfully!"
echo "To set the environment variable for your current session, run:"
echo "  source ./setup_fd_env.sh"
echo ""
echo "To make it permanent, add this line to your shell profile (.bashrc, .zshrc, etc.):"
echo "  export FD_EXEC_PATH=$FD_PATH"
