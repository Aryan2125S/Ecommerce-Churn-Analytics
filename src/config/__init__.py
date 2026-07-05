"""
Configuration Management Package.

Provides centralized configuration loading, environment variable management,
project-wide constants, and settings for all application modules.

Modules:
    config: Loads environment variables from .env and exposes typed settings
            including database credentials, file paths, model parameters,
            and application flags via a Settings dataclass.
    constants: Defines project-wide immutable values including column name
               registries, color palette hex codes, model identifier strings,
               membership types, payment methods, and UI design tokens.

Dependencies:
    - python-dotenv (for .env file loading)
"""
