"""Base class for Milvus migrations."""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class Migration(ABC):
    """Base class for all migrations.
    
    Each migration should implement up() and down() methods to handle
    upgrading and downgrading the schema.
    """
    
    @abstractmethod
    def up(self):
        """Upgrade to this version."""
        pass
    
    @abstractmethod
    def down(self):
        """Downgrade from this version."""
        pass
    
    def __init__(self):
        """Initialize the migration."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def apply(self):
        """Apply the migration by running the up() method."""
        self.logger.info(f"Applying migration {self.__class__.__name__}")
        try:
            self.up()
            self.logger.info(f"Successfully applied migration {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to apply migration {self.__class__.__name__}: {str(e)}")
            raise
    
    def rollback(self):
        """Rollback the migration by running the down() method."""
        self.logger.info(f"Rolling back migration {self.__class__.__name__}")
        try:
            self.down()
            self.logger.info(f"Successfully rolled back migration {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to roll back migration {self.__class__.__name__}: {str(e)}")
            raise 