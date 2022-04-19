from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Office31(ImageList):
    """Office31 Dataset.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
            ("image_list", "image_list.zip", "https://drive.google.com/uc?export=download&id=1JGjr1bYe0oYkso6prudvKYoFi0FOJcrE"),
            ("amazon", "amazon.tgz", "https://drive.google.com/uc?export=download&id=1xq7gPW14FSLlrerR9nDCryKeSBHNNeFp"),
            ("dslr", "dslr.tgz", "https://drive.google.com/uc?export=download&id=14F7HWvclPehy38aVMNNap1oFLVitUAgA"),
            ("webcam", "webcam.tgz", "https://drive.google.com/uc?export=download&id=11OW_6J7kss6nlgKmbX6jWWQbO3yTmHNV"),
    ]
    image_list = {
        "b": "image_list/b.txt",
        "c": "image_list/c.txt",
        "i": "image_list/i.txt",
        "p": "image_list/p.txt",
    }
    CLASSES = ['251.airplanes-101', '224.touring-bike', '113.hummingbird','197.speed-boat','246.wine-bottle','178.school-bus','252.car-side-101','056.dog','105.horse','046.computer-monitor','145.motorbikes-101','159.people' 
             ]

    def __init__(self, root: str, task: str, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        #if download:
            #list(map(lambda args: download_data(root, *args), self.download_list))#下载相应的数据集到文件夹。
        #else:
            #list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))#检查下载位置有没有想要的数据集。

        super(Office31, self).__init__(root, Office31.CLASSES, data_list_file=data_list_file, **kwargs)
