#!/bin/bash
# Easy demo runner script

echo "=========================================="
echo "Autoscaling Policy Demo Runner"
echo "=========================================="
echo ""

# Check if models exist
if [ ! -d "models" ]; then
    echo "❌ Error: models/ directory not found"
    echo "Please train your models first or ensure they're in the models/ directory"
    exit 1
fi

# Function to show menu
show_menu() {
    echo "Choose an option:"
    echo "1) Demo all policies (interactive)"
    echo "2) Demo DQN only"
    echo "3) Demo PPO only"
    echo "4) Demo Q-Learning only"
    echo "5) Demo Threshold only"
    echo "6) Quick comparison (no rendering)"
    echo "7) Long demo (300 steps)"
    echo "8) Random load pattern"
    echo "9) Build Docker image"
    echo "10) Run in Docker"
    echo "0) Exit"
    echo ""
    read -p "Enter choice [0-10]: " choice
}

# Main loop
while true; do
    show_menu
    
    case $choice in
        1)
            echo "Running all policies..."
            python demo_all_models.py
            ;;
        2)
            echo "Running DQN demo..."
            python demo_all_models.py --policy dqn
            ;;
        3)
            echo "Running PPO demo..."
            python demo_all_models.py --policy ppo
            ;;
        4)
            echo "Running Q-Learning demo..."
            python demo_all_models.py --policy qlearning
            ;;
        5)
            echo "Running Threshold demo..."
            python demo_all_models.py --policy threshold
            ;;
        6)
            echo "Running quick comparison (no rendering)..."
            python demo_all_models.py --no-render --steps 200
            ;;
        7)
            echo "Running long demo (300 steps)..."
            python demo_all_models.py --steps 300 --delay 0.1
            ;;
        8)
            echo "Running with random load pattern..."
            python demo_all_models.py --load-pattern RANDOM
            ;;
        9)
            echo "Building Docker image..."
            docker build -f Dockerfile.demo -t gym-scaling-demo .
            echo "✓ Docker image built: gym-scaling-demo"
            ;;
        10)
            echo "Running in Docker..."
            echo "Note: This requires X11 forwarding to be set up"
            docker run -it --rm \
                -e DISPLAY=$DISPLAY \
                -v /tmp/.X11-unix:/tmp/.X11-unix \
                gym-scaling-demo python demo_all_models.py
            ;;
        0)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    
    echo ""
    echo "Press Enter to continue..."
    read
    clear
done
