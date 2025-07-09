"""
A Python Package to convert PEM blog content to usseful information by leveraging LLMs
"""
__version__ = '0.0.8'
from .main import save_to_json_file, make_query
__all__ = ["save_to_json_file", "make_query"]
