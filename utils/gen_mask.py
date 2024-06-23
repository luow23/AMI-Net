import numpy as np
import cv2

def gen_mask(k_list, n, im_size):
    while True:
        Ms = []
        for k in k_list:
            N = im_size // k
            rdn = np.random.permutation(N**2)
            additive = N**2 % n
            if additive > 0:
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive))))
            n_index = rdn.reshape(n, -1)
            for index in n_index:
                tmp = [0. if i in index else 1. for i in range(N**2)]
                tmp = np.asarray(tmp).reshape(N, N)
                tmp = tmp.repeat(k, 0).repeat(k, 1)
                Ms.append(tmp)
        yield Ms

if __name__ == '__main__':
    g = gen_mask([2, 4, 8, 16], 3, 256)
    # b = next(next(g))
    b = next(g)
    print(b[0].shape)
    # for b in g:
    # # b[0] = b[0].astype(np.bool)
    #     print(b[0].shape)
    #     cv2.imshow('1', b[0])
    #     # b[1] = b[1].astype(np.bool)
    #     cv2.imshow('2', b[1])
    #     # b[2] = b[2].astype(np.bool)
    #     cv2.imshow('3', b[2])
    #     print(b[0]+b[1]+b[2])
    #
    #     # print(np.sum(~b[0]+~b[1]+~b[2]))
    #     # a = np.array([1, 1])
    #     # print(~a)
    #     cv2.waitKey(0)
    # print(np.random.permutation(4))