import numpy as np

def preprocess_baringhaus(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Only considers imputed data. Undo 50% overlap ucihar data
    and reshape to (N, n_features).
    """
    n_samples, n_timesteps, n_features = x.shape
    
    new_x = np.zeros((n_samples + 1, n_timesteps//2, n_features))
    new_x[0, :, :] = x[0, :n_timesteps//2, :]
    new_x[1:, :, :] = x[:, n_timesteps//2:, :]
   
    new_mask = np.zeros((n_samples + 1, n_timesteps//2, n_features), dtype=bool)
    new_mask[0, :, :] = mask[0, :n_timesteps//2, :]
    new_mask[1:, :, :] = mask[:, n_timesteps//2:, :]
    
    new_x = np.delete(new_x, new_mask.reshape(-1))
    return new_x.reshape(-1, n_features)


def euc_dist(a,b):
    """euclidian distance between 2 variables with n dimensions.
    """
    return np.linalg.norm(a-b)


def compute_stat_Baringhaus(first_data, second_data):
    try:
        first_data=first_data.reshape([first_data.shape[1], first_data.shape[0]])
        second_data=second_data.reshape([second_data.shape[1], second_data.shape[0]])
        len_seq_first_data=first_data.shape[1]
        len_seq_sec_data=second_data.shape[1]
    except IndexError:
        len_seq_first_data=len(first_data)
        len_seq_sec_data=len(second_data)
        vector=True
    nb_variables_first_data=first_data.shape[0]
    nb_variables_second_data=second_data.shape[0]


    mult_lens=len_seq_sec_data*len_seq_first_data
    sum_lens=len_seq_sec_data+len_seq_first_data
    sum_all_euc_dist=0
    for i in (range(nb_variables_first_data)):
        for j in range((nb_variables_second_data)):

            sum_all_euc_dist+=(euc_dist(first_data[i],second_data[j]))

    sum_euc_dist_intra_first_data=0
    for k in range(nb_variables_first_data):
        for l in range(nb_variables_first_data):
            sum_euc_dist_intra_first_data+=euc_dist(first_data[k],first_data[l])
    sum_euc_dist_intra_second_data=0
    for m in range(nb_variables_second_data):
        for n in range(nb_variables_second_data):
            sum_euc_dist_intra_second_data+=euc_dist(second_data[m],second_data[n])


    return ((mult_lens/sum_lens) * (
        ((1/mult_lens)*sum_all_euc_dist)-
        ((1/(2*len_seq_first_data*len_seq_first_data)) * sum_euc_dist_intra_first_data)-
        ((1/(2*len_seq_sec_data*len_seq_sec_data)) * sum_euc_dist_intra_second_data)
    ))

