import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from struct import unpack
from typing import Tuple, Dict, List
from tqdm import tqdm

GB2312_SPECIAL_NUM = 682
GB2312_LEVEL1_NUM = 3755
GB2312_LEVEL2_NUM = 3008
GBK_NUM = 23940

class Dataset:
    def __init__(self, name: str, data: Dict[str, List[np.ndarray]]):
        '''
        Args:
          name: Identifier of the writer.
          data: {"a": [[(x, y), ...], ...]} Label-to-strokes map. Each stroke have a shape of (n, 2).
        '''
        self.name = name
        self.data = data

    @classmethod
    def from_pot_file(cls, file_path: str):
        '''Load dataset from a single `.pot` file, format defined by CASIA Online Chinese Handwriting Databases.
        
        Args:
          file_path: `.pot` file path.
        '''
        with open(file_path, "rb") as f:
            header_byte = f.read(8)
            data = {}
            while len(header_byte) == 8:
                # get the header
                header: Tuple[int, bytes, int] = unpack("<H4sH", header_byte)
                size = header[0]
                tag = header[1][::-1].decode("gbk")

                # get the coordinates
                coordinates = unpack(f"<{(size-8)//2}h", f.read(size-8))

                data[tag] = []
                stamp = 0
                for i in range(len(coordinates) // 2):
                    if coordinates[2*i] == -1 and coordinates[2*i + 1] == 0:
                        stroke_points = np.array(coordinates[stamp:2*i]).reshape(-1, 2)
                        data[tag].append(stroke_points)
                        stamp = 2*(i+1)

                # continue read next character
                header_byte = f.read(8)
            
        name = os.path.basename(file_path)
        if name[-4:] == ".pot":
            name = name[:-4]
            
        return cls("CASIA_" + name, data)

    @classmethod
    def from_vectors_file(cls, verctors_file: str, labels_file: str):
        '''Load dataset from `_verctors` and `labels` file, format defined by Harbin Institute of Technology Opening
        Recognition Corpus for Chinese Characters
        
        Args:
          verctors_file: `_verctors` file path.
          labels_file: `labels` file path.
        '''
        # read labels file
        with open(labels_file, "rb") as f:
            header: Tuple[int, int] = unpack("<hb", f.read(3))
            size = header[0] * header[1]
            tags_bytes: bytes = unpack(f"<{size}s", f.read(size))[0]
            tags = []
            for i in range(header[0]):
                tags.append(tags_bytes[header[1]*i : header[1]*(i+1)].decode("gbk"))

        with open(verctors_file, "rb") as f:
            # check match, in case you use wrong labels file
            assert unpack("<i", f.read(4))[0] == header[0], "size of labels do not match size of verctors"
            data = {}
            for i, s in zip(range(header[0]), unpack(f"<{header[0]}h", f.read(header[0]*2))):
                raw_data = list(unpack(f"<{s}b", f.read(s))) # transfer to list to avoid memory copy
                stroke_size_list = raw_data[1 : raw_data[0]+1]
                # points of a character
                raw_data = raw_data[raw_data[0]+1 : ]
                
                data[tags[i]] = []
                stamp = 0
                for stroke_size in stroke_size_list:
                    stroke_points = np.array(raw_data[stamp:2*stroke_size+stamp]).reshape(-1, 2)
                    data[tags[i]].append(stroke_points)
                    stamp += 2*stroke_size
        
        name = os.path.basename(verctors_file)
        if name[-8:] == "_vectors":
            name = name[:-8]
        return cls("HIT-OR3C_" + name, data)

class Database:
    def __init__(self, datasets: List[Dataset]):
        '''
        Args:
          datasets: List of dataset.
        '''
        self.datasets = datasets
    
    @classmethod
    def from_config_file(cls, file_path: str, silent=False):
        '''Load datasets from a config file in json format.
        
        Config file must in format `{"pot_dirs": ["pot_dir", ...], "vectors_dirs": [["vectors_dir", 
        "label_file"], ...], "pot_files": ["pot_path", ...], "vectors_files": [["vectors_path", 
        "label_file"], ...]}`. Files in subdirectory will not be loaded.

        Args:
          file_path: Config file path.
          silent: True if you don't want progress bar and loading hints. (default False)
        '''
        print_ = lambda x: x if silent else print
        tqdm_ = lambda x: x if silent else tqdm

        with open(file_path, "r") as f:
            config: dict = json.load(f)

        datasets = []

        # get pathes of pot files
        pot_files = [] if config.get("pot_files") is None else config.get("pot_files")
        if config.get("pot_dirs") is not None:
            pot_dirs: list = config.get("pot_dirs")
            for dir_path in pot_dirs:
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):
                        pot_files.append(file_path)
        
        print_("Loading CASIA datasets...")
        for path in tqdm_(pot_files):
            datasets.append(Dataset.from_pot_file(path))

        # get pathes of vectors files
        vectors_files = [] if config.get("vectors_files") is None else config.get("vectors_files")
        if config.get("vectors_dirs") is not None:
            vectors_dirs: list = config.get("vectors_dirs")
            for dir_and_labels in vectors_dirs:
                for file_name in os.listdir(dir_and_labels[0]):
                    file_path = os.path.join(dir_and_labels[0], file_name)
                    if os.path.isfile(file_path):
                        vectors_files.append((file_path, dir_and_labels[1]))
        
        print_("Loading HIT-OR3C datasets...")
        for vectors_file, labels_file in tqdm_(vectors_files):
            datasets.append(Dataset.from_vectors_file(vectors_file, labels_file))
        
        return cls(datasets)
    
    def digest(self) -> Tuple[int, List[float], plt.Figure, List[str]]:
        '''Get digest of database.

        Returns:
          digests: Number of writers, coverage ratios in [gb2312 special, gb2312 level1, gb2312 level2, gbk],
          figure of frequency distribution, list of character labels.
        '''
        raw_tags: List[str] = []
        lens: List[int] = []
        strokes: List[int] = []
        for dataset in self.datasets:
            raw_tags.extend(dataset.data.keys())
            strokes.extend([len(i) for i in dataset.data.values()])
            for i in dataset.data.values():
                lens.append(sum([len(j) for j in i]))
        tags = Counter(raw_tags)

        # draw frequence figure
        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(20, 8)

        axes[0].hist(tags.values())
        axes[0].set_xlabel("dataset size")
        axes[0].set_ylabel("number of characters")
        axes[0].set_title("Frequency distribution of dataset size")

        axes[1].hist(lens)
        axes[1].set_xlabel("points number")
        axes[1].set_ylabel("number of samples")
        axes[1].set_title("Frequency distribution of points number")

        axes[2].hist(strokes)
        axes[2].set_xlabel("strokes number")
        axes[2].set_ylabel("number of samples")
        axes[2].set_title("Frequency distribution of strokes number")

        # get tags information
        tags = sorted(tags.keys())
        gb2312_special = 0
        gb2312_level1 = 0
        gb2312_level2 = 0
        for tag in tags:
            try:
                id = tag.encode("gb2312")[0] - 160
                if id > 0 and id < 10:
                    gb2312_special += 1
                elif id > 15 and id < 56:
                    gb2312_level1 += 1
                elif id > 55 and id < 88:
                    gb2312_level2 += 1
            except UnicodeEncodeError:
                pass
        ratios = [gb2312_special / GB2312_SPECIAL_NUM, gb2312_level1 / GB2312_LEVEL1_NUM,
                 gb2312_level2 / GB2312_LEVEL2_NUM, len(tags) / GBK_NUM]
        return len(self.datasets), ratios, fig, tags
    
    def raw_dataset(self) -> List[Tuple[List[np.ndarray], str]]:
        '''Concat all datasets in to one.'''
        result = []
        for dataset in self.datasets:
            for key, value in dataset.data.items():
                result.append((value, key))
        return result