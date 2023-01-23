
from omegaconf import OmegaConf
from pathlib import Path
from typing import Union


class ConstOmegaConfLoader:
    """
    A class for loading an OmegaConf configuration from a file and setting it to read-only.

    Attributes:
        config (omegaconf.DictConfig): The loaded and read-only OmegaConf configuration.

    Methods:
        __init__(file_path: pathlib.Path): Initialize the loader with the path to the configuration file.
        load_config(): Load the configuration from the specified file and set it to read-only.
    """

    def __init__(self, file_path: Union[Path, str]) -> None:
        """
        Initialize the ConstOmegaConfLoader.

        Args:
            file_path (pathlib.Path): The path to the configuration file.
        """
        self._file_path = file_path
        self._config = None

    @property
    def config(self) -> 'omegaconf.DictConfig':
        """
        Getter for the loaded and read-only OmegaConf configuration.

        Returns:
            omegaconf.DictConfig: The loaded and read-only configuration.
        """
        if self._config is None:
            self.load_config()
        return self._config

    def load_config(self) -> None:
        """
        Load the configuration from the specified file and set it to read-only.
        """
        self._config = OmegaConf.load(self._file_path)

        # Set the entire configuration as read-only
        OmegaConf.set_readonly(self._config, True)