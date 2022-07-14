export XRT_TPU_CONFIG="localservice;0;localhost:51011"
accelerate launch main.py --config_path ./config.yaml