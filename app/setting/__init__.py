from .config import DevelopmentConfig, ProductionConfig, TestingConfig

env_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig
}