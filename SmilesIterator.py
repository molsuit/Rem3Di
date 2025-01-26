from abc import ABC, abstractmethod

class SmilesIterator(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def close(self):
        pass

class FileSmilesIterator(SmilesIterator):
    def __init__(self, file_path):
        self.file = open(file_path, 'r')
        self.buffer = []
        self.iterator = iter(self.file)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.buffer:
                return self.buffer.pop(0)
            try:
                line = next(self.iterator)
                self.buffer = line.strip().split()
            except StopIteration:
                self.close()
                raise

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()
    
    def __del__(self):
        self.close()

class ListSmilesIterator(SmilesIterator):
    def __init__(self, smiles_list):
        self.buffer = []
        self.iterator = iter(smiles_list)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.buffer:
                return self.buffer.pop(0)
            try:
                line = next(self.iterator)
                self.buffer = line.strip().split()
            except StopIteration:
                raise

    def close(self):
        pass