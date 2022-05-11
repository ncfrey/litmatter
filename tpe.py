import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class TrainingSpeedEstimator():
    """Implements training speed estimators from 
    Ru, Robin, et al. "Speedy Performance Estimation for Neural Architecture Search." 
    Advances in Neural Information Processing Systems 34 (2021).
    """

    def __init__(self, E=1, gamma=0.999, norm=True):
        """
        Parameters
        ----------
        E: int
            number of "burn-in" epochs to throw away at beginning of training
            for TSE-E estimator
            
        gamma: float
            hyperparam for exponential moving average
            

        normalize: bool
            boolean for whether or not to normalize loss data before fitting curve; normalization
            is simply dividing by the maximum loss value and generally gives better results (default True)
            
        """

        self.E = E
        self.gamma = gamma
        self.normalize = norm

    def estimate(self, df_train, T, df_energy=None):
        """
        Parameters
        ---------
        df_train: Pandas dataframe
            dataframe with 'epoch' and 'train_loss_step' columns
        T: int
            number of epochs to consider in estimation
        df_energy: Pandas dataframe (Optional)
            dataframe with GPU index, timestamps, and power draw from nvidia-smi
        Returns
        -------
        tse_dict: dict
            Results from three TSE estimation methods for training loss curve
        """

        B = len(df_train[df_train['epoch']==T])  # number of steps (minibatches) in an epoch
        T_end = df_train.iloc[-1].epoch + 1  # number of total epochs

        tse = df_train[df_train['epoch'] < T].train_loss_stp.sum() / B

        tsee = df_train[(df_train['epoch'] >= T - self.E + 1) & (df_train['epoch'] <= T)].train_loss_stp.sum() / B

        tseema = 0
        for t in range(0, T+1):
            sum_losses = df_train[df_train['epoch']==t].train_loss_stp.sum() / B * self.gamma ** (T-t)
            tseema += sum_losses

        if df_energy is not None:
            energies = []
            for idx in df_energy[' index'].unique():
                df0 = df_energy[df_energy[' index']==idx].reset_index(drop=True)  # power by GPU device index
                E = self._compute_energy(df0)
                energies.append(E)
            total_energy = np.sum(energies) / 1e3
        else:
            total_energy = 0
        
        energy_per_epoch = total_energy / T_end
        energy_per_step = total_energy / len(df_train)
        tpe_dict = {'tse': tse, 'tsee': tsee, 'tseema': tseema, 
                    'T_end': T_end, 'energy_per_epoch (kJ)': energy_per_epoch,
                    'energy_per_step (kJ)': energy_per_step}

        return tpe_dict

    def _compute_energy(self, df):
        ts = pd.to_datetime(df['timestamp'])
        ts = ts - ts[0]
        ts = ts.dt.total_seconds().to_numpy()
        # Quadrature by trapezoidal rule
        deltas = ts[1:] - ts[0: -1]
        power = df[' power.draw [W]'].to_numpy()
        avg_powers = 0.5 * (power[1:] + power[0: -1])
        energy = deltas * avg_powers  # units of watts * seconds
        return np.sum(energy)