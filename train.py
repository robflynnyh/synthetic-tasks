from omegaconf import OmegaConf
import argparse
from transformer import Model

def main(config):
    print(config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='settings.yaml')
    args = parser.parse_args()  
    main(OmegaConf.load(args.config))
