from abc import ABC, abstractmethod

class BaseModelStrategy(ABC):
    """Interface base para todos os modelos"""

    @abstractmethod
    def build_pipeline(self, preprocessor):
        """Deve retornar um Pipeline do Scikit-Learn"""
        pass
    
    @abstractmethod
    def get_params(self):
        """Retorna os hiperparâmetros atuais"""
        pass