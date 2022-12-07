import numpy as np
import pandas as pd

weigh_errors = True

def evaluate_model(shf_data, weigh_errors=weigh_errors,
    return_dataframe=False, save_dir=None):
    #if save_dir is None:
    #    save_dir = 'Output/'
    #Output Dataframe
    #shf_df = shf_data.assign(SHF_model=shf_interpolated)
    #diffs = shf_df['model_values'] - shf_df['data_values']
    shf_df = shf_data.assign(SHF_diff=(shf_data['SHF_model'] - shf_data['SHF']))
    #RMSE
    if weigh_errors is True:
        data_errors = shf_data['Error']/100*shf_data['SHF']
        meanerr = data_errors.mean()
        data_errors = data_errors.fillna(meanerr)
        # shf_df = shf_df.dropna(subset=['data_errors'])
        rmse = calc_rmse_weighted(
            shf_df['SHF_model'], shf_df['SHF'], data_errors,
                    meanerr)
        mse, sigmas = sigma_weighted(
            shf_df['SHF_model'], shf_df['SHF'], data_errors,
                    meanerr)
    else:
        #shf_df = shf_df.drop(columns=['data_error'])
        rmse = calc_rmse(shf_df['SHF_model'], shf_df['SHF'])
        mse, sigmas = sigma(shf_df['SHF_model'], shf_df['SHF'])
    estimators = {'rmse': rmse, 'mse': mse, 'sigmas': sigmas}
    #print_estimators_table(estimators, save_dir + 'estimadores')
    #Standard deviation
    return_value = estimators
    if return_dataframe:
        return_value = [estimators, shf_df]
    return return_value

def calc_rmse(model, data):
    diff = model - data
    rmse = np.sqrt((diff**2).mean())
    return rmse

def calc_rmse_weighted(model, data, data_error, meanerr):
    data_weight =  meanerr / data_error
    n_weight = data_weight/np.sum(data_weight)
    diff = model - data
    #print(n_weight*(diff**2).sum())
    rmse = np.sqrt(sum(n_weight*(diff**2)))
    return rmse

def calc_rmse_weighted_aggressive(model, data, data_error):
    diff = model - data
    data_min = data + data_error
    data_max = data - data_error
    data_salida = np.zeros(len(diff))
    for i in range(len(diff)):
        if abs(diff[i]) < abs(data_error[i]):
            data_salida[i] = model[i]
        elif diff[i] > 0:
            data_salida[i] = data_max[i]
        else:
            data_salida[i] = data_min[i]
    rmse, diff2 = calc_rmse(model, data_salida)
    #np.savetxt('shf_data.txt', data)
    #np.savetxt('ishf.txt', model)
    #np.savetxt('shf_data_error.txt', data_salida)
    #np.savetxt('diff.txt', diff)
    return rmse, data_salida

def sigma(shf_interpolated, data):
    diff = shf_interpolated - data
    #mae = mean_absolute_error(data, shf_interpolated)
    mse = diff.mean()
    sigma = np.std(diff) #np.sqrt(((diff-diff.mean())**2).mean())
    n_1_sigma = diff.mean() - sigma
    p_1_sigma = diff.mean() + sigma
    n_2_sigma = diff.mean() - 2*sigma
    p_2_sigma = diff.mean() + 2*sigma
    sigmas = {
        'p_1_sigma': p_1_sigma, 'n_1_sigma': n_1_sigma,
        'p_2_sigma': p_2_sigma, 'n_2_sigma': n_2_sigma}
    #moda = np.nanmax(abs(diff))
    return mse, sigmas

def sigma_weighted(shf_interpolated, data, data_error, meanerr):
    diff = shf_interpolated - data
    data_weight = meanerr / data_error
    n_weight = data_weight / np.sum(data_weight)
    mse = np.average(diff, weights = n_weight)
    V1 = sum(data_weight)
    V2 = sum(data_weight**2)
    sigma = np.sqrt(sum(data_weight*((diff-mse)**2))/(V1-(V2/V1)))
    n_1_sigma = mse - sigma
    p_1_sigma = mse + sigma
    n_2_sigma = mse - 2*sigma
    p_2_sigma = mse + 2*sigma
    sigmas = {
        'p_1_sigma': p_1_sigma, 'n_1_sigma': n_1_sigma,
        'p_2_sigma': p_2_sigma, 'n_2_sigma': n_2_sigma}
    #moda = np.nanmax(abs(diff))
    return mse, sigmas


def print_estimators_table(estimators, filename):
    if filename == None:
        filename = 'Output/estimadores'
    df = pd.DataFrame.from_dict(estimators)
    writer = pd.ExcelWriter(filename + '.xlsx')
    df.to_excel(writer, 'Estimadores')
    writer.save()
