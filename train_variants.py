import logging
from datetime import datetime

import torch

from trainval_net import Trainer


class TrainVariants(object):
    def __init__(self, epochs, bs, nw=1, mGPUs=1):
        self.epochs = epochs
        self.bs = bs
        self.nw = nw
        self.mGPUs = mGPUs
        self.cuda = torch.cuda.is_available()

    def get_dt(self):
        dt = datetime.now().strftime("%y%m%d")
        return dt

    def inet_pascal_pigs(self, dt):
        """
        Trained on pascal then FC layers modified for pigs
        """
        session = f"{dt}1"
        logging.info(f"[{session}] Training on Pascal VOC with FC reconfigured")
        trained_pascal = Trainer({"dataset": "pascal_voc",
                                  "net": "res101",
                                  "max_epochs": self.epochs,
                                  "disp_interval": 100,
                                  "batch_size": self.bs,
                                  "num_workers": self.nw,
                                  "cuda": True,
                                  "mGPUs": self.mGPUs,
                                  "session": session,
                                  "log_path": f"logs/pascal_voc_{session}",
                                  "is_optimising": True,
                                 })
        trained_pascal.train()

        logging.info(f"[{session}] Training on Pigs VOC")
        trained_pigs = Trainer({"dataset": "pigs_voc",
                                "net": "res101",
                                "max_epochs": self.epochs * 2,
                                "disp_interval": 100,
                                "batch_size": self.bs,
                                "num_workers": self.nw,
                                "cuda": True,
                                "mGPUs": self.mGPUs,
                                "session": session,
                                "resume": True,
                                "checksession": session,
                                "checkepoch": self.epochs,
                                "checkpoint": 1251,
                                "resume_dataset": "pascal_voc",
                                "transfer": True,
                                "resume_classes": 21,
                                "log_path": f"logs/pascal_pigs_voc_{session}",
                                "is_optimising": True,
                                "and_test": True,
                               })
        trained_pigs.train()

    def inet_pigs(self, dt):
        """
        Trained on the pig data
        """
        session = f"{dt}2"
        logging.info(f"[{session}] Training on Pigs VOC")
        trained_pigs = Trainer({"dataset": "pigs_voc",
                                  "net": "res101",
                                  "max_epochs": self.epochs,
                                  "disp_interval": 100,
                                  "batch_size": self.bs,
                                  "num_workers": self.nw,
                                  "cuda": True,
                                  "mGPUs": self.mGPUs,
                                  "session": session,
                                  "log_path": f"logs/pigs_voc_{session}",
                                  "is_optimising": True,
                                  "and_test": True,
                                 })
        trained_pigs.train()

    def inet_pascal(self, dt):
        """
        Train on the pascal using classagnostic
        """
        session = f"{dt}3"
        logging.info(f"[{session}] Training on Pascal VOC Class Agnostic")
        trained_pascal_agnostic = Trainer({"dataset": "pascal_pigs",
                                           "net": "res101",
                                           "max_epochs": self.epochs,
                                           "disp_interval": 100,
                                           "batch_size": self.bs,
                                           "num_workers": self.nw,
                                           "cuda": True,
                                           "mGPUs": self.mGPUs,
                                           "session": session,
                                           "class_agnostic": True,
                                           "log_path":
                                              f"logs/pascal_agnostic_{session}",
                                           "is_optimising": True,
                                           "and_test": True,
                                          })
        trained_pascal_agnostic.train()

    def run(self):
        """
        Train 3 models:
        - ImageNet/PascalVOC/PigsVOC
        - ImageNet/PigsVOC
        - ImageNet/PascalVOC (class agnostic)
        """
        dt = self.get_dt()

        # Pascal then transfer to pigs
        self.inet_pascal_pigs(dt)
        # Pigs only
        self.inet_pigs(dt)
        # Pascal Class Agnostic
        self.inet_pascal(dt)


if __name__ == "__main__":
    # import coloredlogs
    # coloredlogs.install(level="DEBUG",
                        # fmt="%(asctime)s %(levelname)s %(module)s" + \
                            # "- %(funcName)s: %(message)s",
                        # datefmt="%Y-%m-%d %H:%M:%S")

    logging.basicConfig(filename=f"./logs/{str(datetime.now()).replace(' ', '_')}_variants.log",
                        level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(module)s" +
                        "- %(funcName)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    TrainVariants(epochs=2, bs=8).run()
