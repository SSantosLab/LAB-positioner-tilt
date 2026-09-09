import yaml
import copy

class ConfigFile():

    class NotFoundException(Exception):
        pass

    def __init__(self, file):
        self._file = file

        with open(self._file) as stream:
            self._data = yaml.safe_load(stream)

    def dict(self):
        return copy.deepcopy(self._data)

    def get(self, path):
        parts = path.split(':')

        level = self._data
        for pp, part in enumerate(parts):
            if part not in level:
                subpath = ':'.join(parts[:pp])
                raise self.NotFoundException(f'Entry [{part}] not found under [{subpath}] in file [{self._file}]')

            level = level[part]

        return level

    def getDefault(self, path, default=None):
        try:
            return self.get(path)
        except self.NotFoundException as e:
            return default
