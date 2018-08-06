import numpy as np
import argparse


def get_runtime(file):
    runtime = np.load(file)
    print(f'Mean runtime: {np.mean(runtime)}')


def main():
    parser = argparse.ArgumentParser(description='Runtimes')
    parser.add_argument('-f','--file', type=str, help="path to file")
    args = parser.parse_args()

    get_runtime(args.file)


if __name__=='__main__':
    main()
