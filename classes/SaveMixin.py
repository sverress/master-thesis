import os
import pickle
import abc


class SaveMixin(abc.ABC):
    @abc.abstractmethod
    def get_filename(self):
        pass

    def save(self, directory: str, suffix=""):
        # If there is no world_cache directory, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/{self.get_filename()}{suffix}.pickle", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
