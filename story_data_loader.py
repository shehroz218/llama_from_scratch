import torch
from torch.nn import functional as F
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset, DataLoader


class StoryDataset(Dataset):
    def __init__(self, data, tokenizer, context_window=16):
        """ Args:
        data (str): the dataset
        vectorizer (ReviewVectorizer): vectorizer instantiated from dataset """
        self.data = data 
        self._tokenizer = tokenizer
        self.context_window = context_window
        # self.batch_size = batch_size


        # the data is a large string, need to split it into train, val, and test based on periods
        # self.data = self.data.split('.')
        self.data = torch.tensor(self._tokenizer.encode(self.data))


        self.train_data = self.data[:int(.75 * len(self.data))]
        self.train_size = len(self.data[:int(.75 * len(self.data))])

        self.val_data = self.data[int(.75 * len(self.data)): int(.9 * len(self.data))]
        self.val_size = len(self.data[int(.75 * len(self.data)): int(.9 * len(self.data))])

        self.test_data = self.data[int(.9 * len(self.data)):]
        self.test_size = len(self.data[int(.9 * len(self.data)):])


        self._lookup_dict = {'train': (self.train_data, self.train_size), 
                             'val': (self.val_data, self.val_size),
                             'test': (self.test_data, self.test_size)} 
        self.set_split('train')


    @classmethod
    def load_dataset_and_tokenizer(cls, data_path, tokenizer_path, context_window=16):
        """Load dataset and make a new vectorizer from scratch
        Args:
        review_csv (str): location of the dataset
        Returns:
        an instance of ReviewDataset
        """
        #load and slightly preprocess the dataset
        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        string_to_remove = '<|endoftext|>\n'
        result_list = [x for x in lines if x != string_to_remove]
        #limit lines for local testing
        result_list = result_list[:2000]
        lines = "".join(result_list)
        # load the tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)

        return cls(lines, tokenizer)
    
    def get_tokenizer(self):
        """ returns the vectorizer """
        return self._tokenizer
    
    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe
        Args:
        split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_data, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        Args:
        index (int): the index to the data point
        Returns:
        a dict of the data point's features (x_data) and label (y_target)
        """
        # row = self._target_data.iloc[index]
        # review_vector = self._vectorizer.vectorize(row.review)
        # rating_index =  self._vectorizer.rating_vocab.lookup_token(row.rating)


           # pick random starting points
        ix = torch.randint(0, self._target_data.size(0) - self.context_window - 1, ())
        # end_index = min(ix + self.context_window, len(self._target_data)-self.context_window)

        # Subset the data
        x = self._target_data[ix:ix+self.context_window]
        y = self._target_data[ix+1:ix+1+self.context_window]

        # x = torch.stack([self._target_data[i:i+self.context_window] for i in ix]).long()
        # y = torch.stack([self._target_data[i+1:i+self.context_window+1] for i in ix]).long()
        return x, y
        # return {'x_data': review_vector,
        #         'y_target': rating_index}
    
    def get_num_batches(self):
        """Given a batch size, return the number of batches in the dataset
        Args:
        batch_size (int)
        Returns:
        number of batches in the dataset
        """
        return len(self) // self.batch_size


#generate batches:
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cuda"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
    ensure each tensor is on the write device location. """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    
    # for data_list in dataloader:


    for data_list in dataloader:
        out_data_dict = {}
        out_data_dict['x_data'] = data_list[0].to(device)
        out_data_dict['y_target'] = data_list[1].to(device) 

        # for name, tensor in data_dict.items():
        #     out_data_dict[name] = data_dict[name].to(device) 
        yield out_data_dict