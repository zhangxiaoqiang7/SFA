from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path

class RMSegmentation(Dataset):
    """
    RM dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('roadmark'),
                 split='train',
                 transform=None
                 ):
        """
        :param base_dir: path to RM dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'image')
        self._cat_dir = os.path.join(self._base_dir, 'gt')
        
        #self.tr_cls_dict = {0:0,1:1,2:2,3:3,4:4,5:7,6:5,7:5,8:5,9:5,10:7,11:7,12:6,13:6,14:6,15:6,16:6,17:6,18:7,19:7}
        self.tr_cls_dict = {0:0,1:1,2:2,3:3,4:6,5:6,6:4,7:4,8:4,9:4,10:6,11:6,12:5,13:5,14:5,15:5,16:5,17:5,18:6,19:6}

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.readlines()

            for ii, line in enumerate(lines):
                line = line.strip()
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, 'gt_'+ line + ".png")
                
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        
        _tmp = np.array(_target)
        for k,v in self.tr_cls_dict.items():
            _tmp[_tmp == k] = v
        _target = Image.fromarray(_tmp)

        return _img, _target

    def __str__(self):
        return 'RM(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import colorize_mask
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])



    rm_train = RMSegmentation(split='train',
                                transform=composed_transforms_tr)

    dataloader = DataLoader(rm_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj])
            segmap = colorize_mask(tmp, dataset='roadmark')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break
    plt.show(block=True)