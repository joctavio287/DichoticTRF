# Standard libraries
import numpy as np
import time

# Specific libraries
from typing import Union
from sklearn.utils import shuffle
from tqdm import tqdm
import torch
import gc

# Modules
from utils.helpers_processing import (
    Normalize, Standarize, shifted_matrix
)

class TorchMtrf:
    def __init__(
        self, 
        alpha:Union[float, np.ndarray], 
        relevant_indexes:np.ndarray, 
        train_indexes:np.ndarray, 
        test_indexes:np.ndarray, 
        attribute_preprocess:str, 
        eeg_preprocess:str, 
        delays:np.ndarray,
        random_permutations:int=3000,
        fit_intercept:bool=False, 
        shuffle:bool=False, 
        validation:bool=False,
        use_gpu:bool=True,
        solver:str='ridge',
        logger:any=None,
    )->None:
        """
        Initialize the TorchMtrf model, a PyTorch implementation of the TimeDelayingRidge of stimulus to predict EEG.

        Parameters
        ----------
        alpha : float or np.ndarray, optional
            Regularization strength. If validation is True, this should be an array of alphas to be swept, by default None.
        relevant_indexes : np.ndarray
            Array of relevant indexes.
        train_indexes : np.ndarray
            Array of training indexes.
        test_indexes : np.ndarray
            Array of testing indexes.
        attribute_preprocess : str
            Preprocessing solver for stimuli.
        eeg_preprocess : str
            Preprocessing solver for EEG data.
        fit_intercept : bool, optional
            Whether to fit the intercept, by default False.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        validation : bool, optional
            Whether to perform validation, by default False.
        use_gpu : bool, optional
            Whether to use the GPU (CUDA) for computation, by default True.
        solver : str, optional
            Whether to apply Tikhonov regularization, by default False.

        Returns
        -------
        None
        
        Raises
        ------
        None
        """
        assert solver in ['ridge', 'ridge-laplacian', 'fourier-ridge'], f"solver {solver} is not supported. Use 'ridge', 'ridge-laplacian' or 'fourier-ridge'."
        self.solver = solver
        self.relevant_indexes = relevant_indexes
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
        self.alpha = alpha
        self.attribute_preprocess = attribute_preprocess
        self.eeg_preprocess = eeg_preprocess    
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.validation = validation
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.delays = delays
        self.random_permutations = random_permutations
    
    def fit(
        self, 
        stims:np.ndarray, 
        eeg:np.ndarray
    )->None:
        """
        Fit the TorchMtrf model to the given stimuli and EEG data.

        This function constructs the design matrix from the stimuli, applies the relevant indexes,
        and separates the data into training and testing sets. It then standardizes and normalizes
        the data, and fits a Ridge regression model to the training data. If validation is enabled,
        it further splits the training data into training and validation sets and fits the model
        accordingly.

        Parameters
        ----------
        stims : np.ndarray
            The input stimuli data, shape (n_samples, n_features).
        eeg : np.ndarray
            The EEG response data, shape (n_samples, n_channels).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the input data shapes are not compatible with the model.
        """
        
        # Construct design matrix and transform for GPU computation
        X_train, X_pred = shifted_matrix(
            indices_to_keep=self.relevant_indexes,
            train_indexes=self.train_indexes,
            pred_indexes=self.test_indexes,
            optimized_shifted=True,
            delays=self.delays, 
            use_gpu=self.use_gpu,
            output_torch=True,
            features=stims
        )
        del stims
        
        if self.relevant_indexes is None:
            self.relevant_indexes = np.arange(X_train.shape[0]+X_pred.shape[0])
        
        # Remove rows with all zeros
        mask = ~(torch.all(X_train == 0, dim=1))
        X_train = X_train[mask]
        n_featuresbyn_delays = X_train.shape[1]
        number_of_channels = eeg.shape[1]
        n_features = n_featuresbyn_delays // len(self.delays)

        # Get relevant indexes and transform to device, if available. If not, transform to CPU
        try:
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to(self.device)
            del eeg
            y_train = y_temp[self.train_indexes]
            y_train = y_train[mask]
            y_test = y_temp[self.test_indexes]
        except:
            X_train = X_train.cpu()
            X_pred =  X_pred.cpu()
            
            y_temp = torch.tensor(eeg[self.relevant_indexes]).to(torch.float32).to('cpu')
            del eeg            
            y_train = y_temp[self.train_indexes]
            y_train = y_train[mask.cpu()]
            y_test = y_temp[self.test_indexes]
        del y_temp
        
        if self.validation:
            # Delete held out
            del X_pred, y_test
            
            # Make split for validation: validation sets, fixing the train percent of data
            train_percent = .8
            self.train_cutoff = int(train_percent * len(self.train_indexes))
            try:
                X_train_for_val = X_train[:self.train_cutoff]
                X_val = X_train[self.train_cutoff:]
                del X_train
                y_train_for_val = y_train[:self.train_cutoff]
                y_val = y_train[self.train_cutoff:]
                del y_train
            except:
                X_train = X_train.cpu()
                y_train = y_train.cpu()
                
                X_train_for_val = X_train[:self.train_cutoff]
                X_val = X_train[self.train_cutoff:]
                del X_train
                y_train_for_val = y_train[:self.train_cutoff]
                y_val = y_train[self.train_cutoff:]
                del y_train
                        
            # Standarize and normalize 
            X_train_for_val, y_train_for_val, X_pred, y_val = self._standarize_normalize(
                X_train=X_train_for_val, 
                y_train=y_train_for_val, 
                X_pred=X_val, 
                y_test=y_val
            )
            correlations = torch.zeros(
                len(self.alpha), 
                device=self.device, 
                dtype=torch.float32
            )
            correlations_train = torch.zeros(
                len(self.alpha), 
                device=self.device, 
                dtype=torch.float32
            )

            trfs = torch.zeros(
                len(self.alpha), 
                len(self.delays), 
                device=self.device, 
                dtype=torch.float32
            )
            if self.logger and self.logger.level <= 10:
                iterator = tqdm(enumerate(self.alpha), total=len(self.alpha), desc='Sweeping progress', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            else:
                iterator = enumerate(self.alpha)
            for i_alpha, alph in iterator:

                # Fit the model
                mtrfs = self._solver(
                    solver=self.solver,
                    alpha=alph, 
                    X_train=X_train_for_val, 
                    y_train=y_train_for_val
                )
                y_predicted = X_pred @ mtrfs
                y_predicted_train = X_train_for_val @ mtrfs
                
                # Compute correlation
                try:
                    correlations[i_alpha] = self._compute_correlation(
                        y_1=y_predicted,
                        y_2=y_val
                    ).mean()
                except RuntimeWarning:
                    correlations[i_alpha] = 0
                try:
                    correlations_train[i_alpha] = self._compute_correlation(
                        y_1=y_predicted_train,
                        y_2=y_train_for_val
                    ).mean()
                except RuntimeWarning:
                    correlations_train[i_alpha] = 0
                
                trfs[i_alpha] = mtrfs.view(n_features, len(self.delays), mtrfs.shape[-1]).permute(2, 0, 1).mean(dim=0).mean(dim=0) # shape n_chans, feats, delays --> delays
            del X_train_for_val, y_train_for_val, y_predicted, y_predicted_train, y_val, X_pred, mtrfs

            # Let GPU free memory
            gc.collect()
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if n_features>32:
                    time.sleep(.1)
            return trfs.detach().cpu().numpy(), correlations.detach().cpu().numpy(), correlations_train.detach().cpu().numpy()
        else:
            if self.shuffle:
                iterations = np.arange(self.random_permutations)
                number_of_indices = X_train.shape[0]
                
                coefs = torch.zeros(
                    size=(self.random_permutations, number_of_channels, n_features, len(self.delays)), 
                    device=self.device, 
                    dtype=torch.float32
                    )
                correlations = torch.zeros(
                    size=(self.random_permutations, number_of_channels), 
                    device=self.device, 
                    dtype=torch.float32
                    )
                
                X_train, y_train, X_pred, y_test = self.standarize_normalize(
                    X_train=X_train, 
                    X_pred=X_pred, 
                    y_train=y_train, 
                    y_test=y_test
                )
                # Shuffle the data, by requierment of random permutations
                for s in tqdm(iterations, desc='Performing permutations', bar_format="{desc}: {percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                    X_train_p = X_train[torch.randperm(number_of_indices)] # TODO Shufflear y en vez de X
                   
                    # Fit the model
                    mtrfs = self._solver(
                        solver=self.solver,
                        alpha=self.alpha, 
                        X_train=X_train_p, 
                        y_train=y_train
                    )
                    del X_train_p
                    y_predicted = X_pred @ mtrfs
                    coefs[s] = mtrfs.view(n_features, len(self.delays), mtrfs.shape[-1]).permute(2, 0, 1)
                    del mtrfs
                    
                    try:
                        correlations[s] = self._compute_correlation(
                            y_1=y_predicted,
                            y_2=y_test
                        )
                    except RuntimeWarning:
                        correlations[s] = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)
                    
                del X_train, y_train, X_pred, y_test, y_predicted
                # Let GPU free memory   
                gc.collect()
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if n_features>32:
                        time.sleep(.1)
                return coefs.cpu().numpy(), correlations.cpu().numpy()
            else:
                # Standarize and normalize
                X_train, y_train, X_pred, y_test = self._standarize_normalize(
                    X_train=X_train, 
                    X_pred=X_pred, 
                    y_train=y_train, 
                    y_test=y_test
                )

                # Fit the model 
                mtrfs = self._solver(
                    solver=self.solver,
                    alpha=self.alpha, 
                    X_train=X_train, 
                    y_train=y_train
                )
                del X_train, y_train
                
                # Perform predictions
                y_predicted = X_pred @ mtrfs
                del X_pred
                if torch.all(y_predicted==0):
                    print(f'\n\t\tFold prediction is null, this may be due to the sparsity of weights. If there are\n\t\ttoo many zeros when making product with selected stimuli, the product may be null.')
                
                # Store mtrfs
                mtrfs = mtrfs.view(n_features, len(self.delays), mtrfs.shape[-1]).permute(2, 0, 1) # shape n_chans, feats, delays

                # Calculates and saves correlation of each channel # TODO HACER SOLO DE 0  EN ADELANTE
                try:
                    correlation_matrix = self._compute_correlation(
                        y_1=y_predicted,
                        y_2=y_test
                    )
                except RuntimeWarning:
                    correlation_matrix = torch.zeros(y_predicted.shape[1], device=self.device, dtype=torch.float32)

                # Let GPU free memory   
                gc.collect()
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if mtrfs.shape[1]>32:
                        time.sleep(.1)
                return mtrfs.cpu().numpy(), correlation_matrix.cpu().numpy()
    
    def _solver(
        self, 
        X_train:torch.Tensor, 
        y_train:torch.Tensor,
        alpha:Union[float, int]=1.0, 
        solver:str='ridge'
        )->torch.Tensor:
        """
        Solve the regression problem using the specified solver.

        Parameters
        ----------
        X_train : torch.Tensor
            Training data, shape (n_samples, n_features).
        y_train : torch.Tensor
            Target values, shape (n_samples, n_channels).
        alpha : float or np.ndarray, optional
            Regularization strength, by default 1.0.
        solver : str, optional
            The solver to use for solving the regression problem, by default 'ridge'.

        Returns
        -------
        torch.Tensor
            The coefficients of the regression model.
        """
        if solver == 'ridge':
            XTX_reg = X_train.T @ X_train + torch.tensor(alpha, dtype=torch.float32) *  torch.eye(X_train.shape[1], device=self.device) # X^T * X + alpha*laplacian_matrix
            return torch.linalg.solve(XTX_reg, X_train.T @ y_train)

        elif solver == 'ridge-laplacian': # (X^T X + alpha * M) * mtrfs = X^T * y_train_p 
            n_features = X_train.shape[1]
            laplacian_matrix = torch.diag(torch.full((n_features,), 2.0, device=self.device, dtype=torch.float32))
            laplacian_matrix += torch.diag(torch.full((n_features-1,), -1.0, device=self.device, dtype=torch.float32), diagonal=1)
            laplacian_matrix += torch.diag(torch.full((n_features-1,), -1.0, device=self.device, dtype=torch.float32), diagonal=-1)
            laplacian_matrix[n_features-1, n_features-1] = 1.0
            laplacian_matrix[0, 0] = 1.0
            XTX_reg = X_train.T @ X_train + torch.tensor(alpha, dtype=torch.float32) *  laplacian_matrix # X^T * X + alpha*laplacian_matrix
            return torch.linalg.solve(XTX_reg, X_train.T @ y_train)
        
    def _compute_correlation(
        self,
        y_1:torch.Tensor, 
        y_2:torch.Tensor
        ):
        """
        Compute mean correlation between predicted and true values using centered Pearson correlation.

        Parameters
        ----------
        y_1 : torch.Tensor
            Predicted values, shape (n_samples, n_channels)
        y_2 : torch.Tensor
            True values, shape (n_samples, n_channels)

        Returns
        -------
        float
            Mean correlation across channels, or 0 if std is zero.
        """
        y_1_centered = y_1 - y_1.mean(dim=0, keepdim=True)
        y_2_centered = y_2 - y_2.mean(dim=0, keepdim=True)
        y_1_std = y_1_centered.std(dim=0, unbiased=True)
        y_2_std = y_2_centered.std(dim=0, unbiased=True)
        covariance = (y_1_centered*y_2_centered).mean(dim=0)

        if torch.all(y_2_std == 0) or torch.all(y_2_std == 0):
            print("\n Error: null standard deviation")
            return 0
        else:
            return (covariance / (y_1_std * y_2_std))
        
    def _standarize_normalize(
        self, 
        X_train:np.ndarray, 
        X_pred:np.ndarray, 
        y_train:np.ndarray=None, 
        y_test:np.ndarray=None
        ):
        """
        Standarize|Normalize training and test data.
        Parameters
        ----------
        X_train : np.ndarray
            Fatures to be normalized. Its dimensions should be samples x features 
        y_train : np.ndarray
            EEG samples to be normalized. Its dimensions should be samples x features

        Returns
        -------
        tuple
            A tuple containing the standardized/normalized training and test data: (X_train, y_train, X_pred, y_test).
        """
        # Instances of normalize and standarize
        normalization = Normalize(
            axis=0, 
            porcent=5, 
            by_gpu=self.use_gpu
        )
        standarization = Standarize(
            axis=0,
            by_gpu=self.use_gpu
        )

        # Iterates to normalize|standarize over features
        if self.attribute_preprocess=='Standarize':
            X_train = standarization.fit_standarize_train(train_data=X_train) 
            X_pred = standarization.fit_standarize_test(test_data=X_pred)
            # for feat in range(X_train.shape[1]):
            #     X_train[:, feat] = standarization.fit_standarize_train(train_data=X_train[:, feat]) 
            #     X_pred[:, feat] = standarization.fit_standarize_test(test_data=X_pred[:, feat])
        if self.attribute_preprocess=='Normalize':
            X_train = normalization.fit_normalize_train(train_data=X_train) 
            X_pred = normalization.fit_normalize_test(test_data=X_pred)
        if y_train is None or y_test is None:                
            return X_train, X_pred
        else:
            if self.eeg_preprocess=='Standarize':
                y_train=standarization.fit_standarize_train(train_data=y_train)
                y_test=standarization.fit_standarize_test(test_data=y_test)
            if self.eeg_preprocess=='Normalize':
                y_train=normalization.fit_normalize_percent(data=y_train)
                y_test=normalization.fit_normalize_test(test_data=y_test)
            return X_train, y_train, X_pred, y_test

def fold_model(
    fold:int, 
    alpha:Union[float, np.ndarray],  
    stims:np.ndarray, 
    eeg:np.ndarray, 
    relevant_indexes:np.ndarray,
    train_indexes:np.ndarray, 
    test_indexes:np.ndarray, 
    delays:np.ndarray,
    validation:bool=False, 
    shuffle:bool=False,
    statistical_test:bool=False, 
    path_null:str=None, 
    logger:any=None,
    solver:str='ridge',
    attribute_preprocess:str='Standarize',
    eeg_preprocess:str='Standarize',
    use_gpu:bool=True
    ) -> tuple:
    """
    Perform parallel fold model training and evaluation. 
    This function is design to be run in parallel over folds.
    
    Parameters
    ----------
    fold : int
        The fold number.
    alpha : float or np.ndarray
        Regularization parameter for the model. If validation is True, it can be an array of alphas.
    stims : np.ndarray
        Stimuli data array.
    eeg : np.ndarray
        EEG data array.
    relevant_indexes : np.ndarray
        Array of relevant indexes.
    train_indexes : np.ndarray
        Array of training indexes.
    test_indexes : np.ndarray
        Array of test indexes.
    validation : bool, optional
        Whether to perform validation (default is False).
    shuffle : bool, optional
        Whether to perform permutations in order to construct null model (default is False).
    statistical_test : bool, optional
        Whether to perform statistical tests (default is False).
    path_null : str, optional
        Path to null data for statistical tests (default is None).
    
    Returns
    -------
    tuple
        If statistical_test is True and shuffle is False, returns (fold, weights, correlation_matrix, root_mean_square_error, p_corr, p_rmse, significant_corr_count, significant_rmse_count, null_correlation_per_channel).
        Else if statistical_test is False and shuffle is False, returns (fold, weights, correlation_matrix, root_mean_square_error).
        Otherwise (statistical_test False, shuffle True), returns (iteration, fold, weights, correlation_matrix, root_mean_square_error).
    """
    
    mtrf = TorchMtrf(
        relevant_indexes=relevant_indexes if relevant_indexes is not None else None,
        attribute_preprocess=attribute_preprocess, 
        eeg_preprocess=eeg_preprocess,
        train_indexes=train_indexes, 
        test_indexes=test_indexes, 
        use_gpu=use_gpu,
        delays=delays,
        random_permutations=3000,
        validation=validation,
        fit_intercept=False,
        shuffle=shuffle, 
        alpha=alpha,
        solver=solver,
        logger=logger
    )
            
    if shuffle:        
        # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
        weights, correlation_matrix = mtrf.fit( # n_iterations, n_chans, feats, delays; # n_iterations, n_chans
            stims, 
            eeg
        )
        return weights, correlation_matrix
    elif validation:
        return mtrf.fit(stims, eeg)
    else:
        weights, correlation_matrix = mtrf.fit(stims, eeg) 
            
        # Perform statistical test
        if statistical_test:
            print("\n Statistical test is not implemented yet.")
            # # Null Hypothesis (H0): There is no significant relationship between the predicted and actual EEG data. The test statistic (e.g., correlation or RMSE) follows the null distribution.
            # # Alternative Hypothesis (H1): There is a significant relationship between the predicted and actual EEG data. The test statistic follows the alternative distribution.
            # null_data = load_pickle(path=os.path.join(path_null, f'null_metrics_ses_{session}_sub_{subject}_{random_permutations}.pkl'))
            # null_correlation_per_channel = null_data['null_correlation_per_channel_per_fold']
            # iterations =  null_correlation_per_channel.shape[1]

            # # Correlation and RMSE (n_iterations, n_channels)
            # null_correlation_matrix = null_correlation_per_channel[fold]

            # # p-values for both tests: probability of getting a value equal or greater than the measured value, given the null hypothesis distribution (P(X>=X_obs|H0))
            # # (null_correlation_matrix > correlation_matrix) is the number of iterations that surpasses the measured values for each channel (n_channels)
            # p_corr = ((null_correlation_matrix > correlation_matrix).sum(axis=0) + 1) / (iterations + 1) # +1 to avoid division by zero, right tail test
            # return fold, weights, correlation_matrix, p_corr, null_correlation_per_channel
        else:
            return weights, correlation_matrix