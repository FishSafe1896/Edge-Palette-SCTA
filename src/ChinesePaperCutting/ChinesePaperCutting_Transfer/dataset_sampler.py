import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader
import torchvision.transforms as T


########################################## Dataset ##########################################

class SimpleDataset(Dataset):
  def __init__(self, dir_path, transforms=None):
    super(SimpleDataset, self).__init__()
    assert os.path.isdir(dir_path), """'dir_path' needs to be a directory path."""
    self.dir_path = dir_path
    self.img_paths = os.listdir(self.dir_path)
    
    if not transforms is None:
      self.transforms = transforms
    else:
      self.transforms = T.ToTensor()

  def __getitem__(self, index):
    file_name = self.img_paths[index]
    img = Image.open(os.path.join(self.dir_path, file_name)).convert('RGB')
    img = self.transforms(img)
    return img

  def __len__(self):
    return len(self.img_paths)


# Example 1
# dataset_COCO = SimpleDataset('../input/styletransfer224/Style_Transfer_224/COCO_224/Train', transforms=T.ToTensor())
# dataset_Wiki = SimpleDataset('../input/styletransfer224/Style_Transfer_224/WikiArt_224/Train', transforms=T.ToTensor())
# print('COCO Lenth: ' + str(len(dataset_COCO)))
# print('Wiki Lenth: ' + str(len(dataset_Wiki)))

# Example 2
# dataset_COCO = SimpleDataset('../input/styletransfer224/Style_Transfer_224/COCO_224/Train', transforms=T.ToTensor())
# img = dataset_COCO[0]
# print(img.shape)

# Example 3
# import matplotlib.pyplot as plt
# dataset_COCO = SimpleDataset('../input/styletransfer224/Style_Transfer_224/COCO_224/Train', transforms=T.ToTensor())
# plt.figure()
# for i in range(2):
#   plt.subplot(1,2,i+1)
#   plt.imshow(dataset_COCO[i].numpy().transpose(1,2,0))
# plt.show()
# plt.close()


########################################## Sampler ##########################################

def InfiniteSampler(n, seed=None):
  """ Generator returning the random number between 0 to n-1
  """
  i = n - 1
  rng = np.random.RandomState(seed) if seed is not None else np.random
  order = rng.permutation(n)
  while True:
    yield order[i]
    i += 1
    if i >= n:
      order = rng.permutation(n)
      i = 0

class InfiniteSamplerWrapper(Sampler):
  def __init__(self, data_source, seed=None):
    self.num_samples = len(data_source)
    self.seed = seed

  def __iter__(self):
    return iter(InfiniteSampler(self.num_samples, self.seed))

  def __len__(self):
    return 2 ** 31


# Example 1
# dataset_COCO = SimpleDataset('../input/styletransfer224/Style_Transfer_224/COCO_224/Train', transforms=T.ToTensor())
# sampler = InfiniteSamplerWrapper(dataset_COCO)
# print('Indice sampled: ')
# for i in range(10):
#   print(next(iter(sampler)))

# Example 2
# dataset_COCO = SimpleDataset('../input/styletransfer224/Style_Transfer_224/COCO_224/Train', transforms=T.ToTensor())
# sampler_COCO = InfiniteSamplerWrapper(dataset_COCO)
# dataloader_COCO = DataLoader(dataset_COCO, batch_size=4, sampler=sampler_COCO, num_workers=0)
# dataloader_COCO_iter = iter(dataloader_COCO)
# for i in range(10):
#   img = next(dataloader_COCO_iter)
#   print(img.shape)

# Example 3
# import matplotlib.pyplot as plt
# dataset_COCO = SimpleDataset('../input/styletransfer224/Style_Transfer_224/COCO_224/Train', transforms=T.ToTensor())
# sampler_COCO = InfiniteSamplerWrapper(dataset_COCO)
# dataloader_COCO = DataLoader(dataset_COCO, batch_size=1, sampler=sampler_COCO, num_workers=0)
# dataloader_COCO_iter = iter(dataloader_COCO)
# plt.figure()
# for i in range(9):
#   img = next(dataloader_COCO_iter).squeeze(dim=0)
#   plt.subplot(3,3,i+1)
#   plt.imshow(img.numpy().transpose(1,2,0))
# plt.show()
# plt.close()
