from keras.callbacks import LearningRateScheduler
import numpy as np

class OneCycleScheduler(LearningRateScheduler):
    '''
    Sets the learning rate of each parameter group according to the 1cycle learning rate policy. 
    The 1cycle policy anneals the learning rate from an initial learning rate to some maximum 
    learning rate and then from that maximum learning rate to some minimum learning rate much 
    lower than the initial learning rate. This policy was initially described in the paper 
    Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.
    '''
    def __init__(
        self,
        max_lr,
        epochs,
        pct_start = 0.3,
        div_factor = 25.0,
        final_div_factor = 1e5,
        **kwargs
    ):
        super().__init__(self._schedule)
        self.n_warmup_epochs = int(epochs * pct_start)
        self.lr_init = max_lr / div_factor
        self.lr_max = max_lr
        self.lr_min = max_lr / final_div_factor
        self.n_epochs = epochs
    def _schedule(self, epoch, lr):
        if epoch < self.n_warmup_epochs:
            return (self.lr_max - self.lr_init) / self.n_warmup_epochs * epoch + self.lr_init
        else:
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * (epoch - self.n_warmup_epochs) / (self.n_epochs - self.n_warmup_epochs)))