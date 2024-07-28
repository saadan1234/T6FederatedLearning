import hydra
from omegaconf import DictConfig, OmegaConf
from dataProcess import prepare_fldata

@hydra.main(config_path='conf', config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainloaders, valloaders, testloaders = prepare_fldata(cfg.num_clients, cfg.batch_size)

    print(len(trainloaders), len(trainloaders[0].dataset))

