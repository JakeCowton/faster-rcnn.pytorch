import logging
from datetime import datetime
from multiprocessing import cpu_count

import torch

from trainval_net import Trainer


class TrainVariants(object):
    def __init__(self, epochs, bs, nw=cpu_count(), mGPUs=1):
        self.epochs = epochs
        self.bs = bs
        self.nw = nw
        self.mGPUs = mGPUs
        self.cuda = torch.cuda.is_available()

    def generate_session(self, i):
        dt = datetime.now().strftime("%y%m%d")
        session = f"{dt}{i}"
        return session

    def inet_pascal_pigs(self):
        session = self.generate_session(1)
        logging.info(f"[{session}] Training on Pascal VOC")
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
                                "checkpoint": 10021,
                                "resume_dataset": "pascal_voc",
                                "transfer": True,
                                "resume_classes": 21,
                                "log_path": f"logs/pascal_pigs_voc_{session}",
                                "is_optimising": True,
                                "and_test": True,
                               })
        trained_pigs.train()

    def inet_pigs(self):
        session = self.generate_session(2)
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

    def inet_pascal(self):
        session = self.generate_session(3)
        logging.info(f"[{session}] Training on Pigs VOC")
        trained_pascal_agnostic = Trainer({"dataset": "pascal_voc_agnostic",
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
        self.inet_pigs()
        self.inet_pascal_pigs()
        # self.inet_pascal()


if __name__ == "__main__":
    import coloredlogs
    coloredlogs.install(level="DEBUG",
                        fmt="%(asctime)s %(levelname)s %(module)s" + \
                            "- %(funcName)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    TrainVariants(epochs=2, bs=8).run()
