# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Utils for pytorch.
"""

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


class EarlyStopper:
    """Taken from https://stackoverflow.com/a/73704579 and modified to be able to restore the best epoch's weights"""

    def __init__(self, patience=1, min_delta=0, model=None):
        """Initialize the EarlyStopper.

        Args:
            patience (int, optional): The number of epochs with no improvement after which training will be
                stopped. Defaults to 1.
            min_delta (int, optional): The minimum change in the monitored quantity to qualify as an
                improvement. Defaults to 0.
            model (object, optional): The PyTorch model to restore the best weights. Defaults to None, the best
                network weights are not restored.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.model = model

        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_weights = None

    def early_stop(self, validation_loss):
        # An epoch counts as an improvement only if it beats the running best by at least min_delta;
        # otherwise the patience counter advances. This is what makes min_delta a genuine "minimum
        # improvement" threshold -- without it, a validation loss creeping down by negligible amounts
        # keeps setting new minima, the counter never advances, and training never stops.
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.counter = 0
        else:
            self.counter += 1

        # always checkpoint the genuine best, so restoring "best weights" returns the lowest-loss model
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            if self.model is not None:
                self.best_weights = self.model.state_dict()

        if self.counter >= self.patience:
            if self.model is not None and self.best_weights is not None:
                self.model.load_state_dict(self.best_weights)
                LOGGER.info(f"Restored the weights from the best epoch (vali_loss = {self.min_validation_loss:.2f})")
            return True
        return False


def get_lr(optimizer):
    """Get the current learning rate from an optimizer. Taken from https://stackoverflow.com/a/52671057."""

    for param_group in optimizer.param_groups:
        return param_group["lr"]
