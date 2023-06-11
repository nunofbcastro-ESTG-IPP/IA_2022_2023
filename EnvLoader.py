import os
from os.path import join
from dotenv import load_dotenv

class EnvLoader:
    def __init__(self, file_name: str = ".env"):
        dotenv_path = join(os.path.abspath(os.path.dirname(__file__)), file_name)
        load_dotenv(dotenv_path)
    
    def get_env_as_float(self, key: str, default=None) -> float:
        value = self.get_env_as_string(key, None)

        try:
            return float(value)
        except ValueError:
            return default
    
    def get_env_as_int(self, key: str, default=None) -> int:
        value = self.get_env_as_string(key, None)
        
        try:
            return int(value)
        except ValueError:
            return default
        
    def get_env_as_string(self, key: str, default=None) -> str:
        return os.environ.get(key, default)
        
    def get_env_as_bool(self, key: str, default=None) -> bool:
        value = self.get_env_as_string(key, None)
        
        if value == "true":
            return True
        elif value.lower() == "false":
            return False
        return default