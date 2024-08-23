import os
import pickle

from torch.utils.data import Dataset
from pyolhccr.dataset import Database

class OLHCCRDataset(Dataset):
    def __init__(self, level, transform):
        '''Load dataset into memory at once since less than 4GB will be used. Cache will used for the second time.
        
        Args:
          level: Name of datasets for different use, such as train, test, validation.
          transform: Function to call when `__getitem__` is called.
        '''
        self.transform = transform
        # load from cache
        if os.path.exists(f"{level}_dataset.pkl"):
            with open(f"{level}_dataset.pkl", "rb") as f:
                self.dataset = pickle.load(f)
            self.tags = sorted(set([i[1] for i in self.dataset]))
        else: # load from files
            database = Database.from_config_file(f"{level}.json")
    
            # digest
            self.writers_num, self.ratios, self.digest_fig, self.tags = database.digest()
    
            # get dataset
            self.dataset = database.raw_dataset()
            with open(f"{level}_dataset.pkl", "wb+") as f:
                pickle.dump(self.dataset, f)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index]
        return self.transform(x[0]), self.tags.index(x[1])