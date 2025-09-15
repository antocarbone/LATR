from mmengine.registry import MODELS

# Importa i moduli
from .latr import LATR
from .latr_head import LATRHead

# Verifica e registra se necessario
def ensure_registration():
    modules_to_register = [
        ('LATR', LATR),
        ('LATRHead', LATRHead)
    ]
    
    for name, cls in modules_to_register:
        if name not in MODELS.module_dict:
            MODELS.register_module(name=name, module=cls)
            print(f"Registered {name}")

ensure_registration()

__all__ = ['LATR', 'LATRHead']