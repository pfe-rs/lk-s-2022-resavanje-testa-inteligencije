from train import *

dataset_path = "C:\\Users\\danil\\Desktop\\human tests"
epoch = 5
if __name__ == '__main__':

    metrics_test = test(5)

    acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count']) #preciznost za test set

    time_now = datetime.now().strftime('%H:%M:%S')

    with open(save_log_name, 'a') as f:
        f.write('Epoch {:02d}: Accuracy: {:.3f}, Time: {:s}\n'.format(
            5, acc_test, time_now)) #zapisujemo broj epoha, preciznost na test setu i vreme kada se testiranje zavrsilo