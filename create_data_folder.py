import os, os.path as osp
import argparse

def main(args):
    folder_name = args.name
    os.makedirs(f"data/{folder_name}/albedo", exist_ok=True)
    os.makedirs(f"data/{folder_name}/depth", exist_ok=True)
    os.makedirs(f"data/{folder_name}/normal", exist_ok=True)
    os.makedirs(f"data/{folder_name}/im", exist_ok=True)
    os.makedirs(f"data/{folder_name}/material", exist_ok=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="custom")
    args = parser.parse_args()
    main(args)