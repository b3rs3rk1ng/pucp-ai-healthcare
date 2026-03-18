"""Configuraciones de canales para el experimento de reducción de electrodos."""

CHANNEL_CONFIGS = {
    64: None,  # Se llena con todos los canales al cargar data
    32: ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
         'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
         'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
         'F3', 'F1', 'Fz', 'F2', 'F4',
         'P3', 'P1', 'Pz', 'P2', 'P4', 'P6'],
    16: ['FC3', 'FC1', 'FCz', 'FC2', 'FC4',
         'C3', 'C1', 'Cz', 'C2', 'C4',
         'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Fz'],
    8:  ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CP4'],
    4:  ['C3', 'Cz', 'C4', 'CPz'],
    2:  ['C3', 'C4'],
}


def get_configs(all_ch_names):
    """Retorna channel_configs con 64ch completado."""
    configs = dict(CHANNEL_CONFIGS)
    configs[64] = list(all_ch_names)
    return configs
