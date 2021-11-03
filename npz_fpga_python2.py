import argparse
import numpy as np
import cPickle


def app(opt):
    npz = np.load(opt.save, allow_pickle=True)

    state_dict = {}
    for item in npz:
        print(item)
        state_dict[item] = cPickle.loads(npz[item])

    print(state_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', default='data/npz/n3ml_202111021632.npz')

    app(parser.parse_args())
