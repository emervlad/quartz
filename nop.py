from omegaconf import OmegaConf
from src import encoder


def main():
    print(sum(p.numel() for p in encoder.QuartzNet(OmegaConf.load('conf/quarznet_5x5_ru.yaml').model.encoder).parameters()))


if __name__ == "__main__":
    main()
