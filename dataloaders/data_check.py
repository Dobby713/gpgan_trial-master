from PIL import Image
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    root_path = 'C:/data/gpgan/lfw'
    trA = os.path.join(root_path, 'trainA')
    trB = os.path.join(root_path, 'trainB')

    trA_list = glob(os.path.join(trA, '*.jpg'))
    trB_list = glob(os.path.join(trB, '*.jpg'))

    num_idx = 2
    # A = Image.open(trA_list[num_idx]).convert('RGB')
    # B = Image.open(trB_list[num_idx]).convert('RGB')

    A = Image.open('C:/data/gpgan/lfw\\trainA\\AJ_Cook_0001_A.jpg')
    B = Image.open('C:/data/gpgan/lfw\\trainB\\AJ_Cook_0001_B.jpg')
    npA = np.array(A)
    npB = np.array(B)
    npC = (npA + npB) // 2

    plt.imshow(npA)
    plt.show()
    plt.imshow(npB)
    plt.show()
    plt.imshow(npC)
    plt.show()
    print(1)