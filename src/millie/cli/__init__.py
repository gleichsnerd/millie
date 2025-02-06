"""CLI tooling for Millie, a Milvus ORM with migration support.""" 

from dotenv import load_dotenv, find_dotenv
from .router import cli, add_millie_commands

__all__ = ['cli', 'add_millie_commands']

def main():
    # Load .env from current working directory
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path, override=True)  # Force reload of environment variables
    cli()

if __name__ == "__main__":
    main()
