"""
Configuration file for model training experiments
Use this to easily switch between different training configurations
"""

# Base configuration for all experiments
BASE_CONFIG = {
    'data_yaml': 'data.yaml',
    'project': 'runs/detect',
    'imgsz': 640,
}

# Quick training configuration (for testing)
QUICK_CONFIG = {
    **BASE_CONFIG,
    'name': 'waste_detection_quick',
    'model_size': 'n',
    'epochs': 50,
    'batch': 16,
    'patience': 20,
}

# Standard training configuration
STANDARD_CONFIG = {
    **BASE_CONFIG,
    'name': 'waste_detection_standard',
    'model_size': 'n',
    'epochs': 100,
    'batch': 16,
    'patience': 50,
}

# High accuracy configuration (requires good GPU)
HIGH_ACCURACY_CONFIG = {
    **BASE_CONFIG,
    'name': 'waste_detection_high_acc',
    'model_size': 'm',
    'epochs': 200,
    'batch': 32,
    'patience': 75,
    'imgsz': 640,
}

# Maximum performance configuration
MAX_PERFORMANCE_CONFIG = {
    **BASE_CONFIG,
    'name': 'waste_detection_max',
    'model_size': 'l',
    'epochs': 300,
    'batch': 32,
    'patience': 100,
    'imgsz': 640,
}

# CPU training configuration
CPU_CONFIG = {
    **BASE_CONFIG,
    'name': 'waste_detection_cpu',
    'model_size': 'n',
    'epochs': 50,
    'batch': 2,
    'patience': 20,
    'imgsz': 416,  # Smaller image size for faster processing
}

# Real-time inference optimized
REALTIME_CONFIG = {
    **BASE_CONFIG,
    'name': 'waste_detection_realtime',
    'model_size': 'n',
    'epochs': 100,
    'batch': 16,
    'patience': 50,
    'imgsz': 416,  # Smaller for faster inference
}

# Custom configuration template
CUSTOM_CONFIG = {
    **BASE_CONFIG,
    'name': 'waste_detection_custom',
    'model_size': 's',  # 'n', 's', 'm', 'l', 'x'
    'epochs': 150,
    'batch': 16,
    'patience': 50,
    'imgsz': 640,
}

# Select which config to use
# Change this to switch between configurations
ACTIVE_CONFIG = STANDARD_CONFIG

def get_config(config_name='standard'):
    """
    Get configuration by name
    
    Args:
        config_name: Name of configuration ('quick', 'standard', 'high_accuracy', 'max', 'cpu', 'realtime', 'custom')
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'quick': QUICK_CONFIG,
        'standard': STANDARD_CONFIG,
        'high_accuracy': HIGH_ACCURACY_CONFIG,
        'max': MAX_PERFORMANCE_CONFIG,
        'cpu': CPU_CONFIG,
        'realtime': REALTIME_CONFIG,
        'custom': CUSTOM_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]

if __name__ == "__main__":
    # Display all available configurations
    print("Available Training Configurations:")
    print("=" * 70)
    
    configs = {
        'quick': QUICK_CONFIG,
        'standard': STANDARD_CONFIG,
        'high_accuracy': HIGH_ACCURACY_CONFIG,
        'max': MAX_PERFORMANCE_CONFIG,
        'cpu': CPU_CONFIG,
        'realtime': REALTIME_CONFIG,
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  Model: YOLOv8{config['model_size']}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch: {config['batch']}")
        print(f"  Image Size: {config['imgsz']}")
        print(f"  Output: {config['project']}/{config['name']}")
