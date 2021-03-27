import os
import time

window_size = 10
num_keypoints = 16
project_dirname = '/Users/dcongtinh/Workspace/endless-runner'  # os.getcwd()
# project_dirname = os.path.join(project_dirname, 'train', 'CNN')
project_dirname = os.path.join(project_dirname)  # Run code Jupyter

epochs = 200
batch_size = 16
learning_rate = 1e-5
weight_decay = 1e-4

# Configure early stopping
patience = 10
str_time = time.strftime('%Y%m%d_%H%M%S')
result_dir = os.path.join(project_dirname, 'results', str_time)

model_save_path = os.path.join(result_dir, str_time + 'model.h5')
history_json_save_path = os.path.join(result_dir, str_time + 'history.json')
log_path = os.path.join(result_dir, str_time + '.log.csv')
cfg_path = os.path.join(result_dir, str_time + '.cfg')
acc_plot_path = os.path.join(result_dir, 'ACC_plot.png')
auc_plot_path = os.path.join(result_dir, 'AUC_plot.png')
loss_plot_path = os.path.join(result_dir, 'LOSS_plot.png')
cm_plot_path = os.path.join(result_dir, 'CM_plot.png')

classes = os.listdir('pose/datasets')
classes.sort()
classes = [_class for _class in classes if _class != '.DS_Store']
n_classes = len(classes)

# date_result_sorted = sorted(os.listdir(
#     os.path.join(project_dirname, 'results', dataset_name)))
# date_result = date_result_sorted[-1]
# # date_result = '20200731_211006'  # '20200731_211006', '20200725_160559', '20200721_193325'
# date_result = '20200925_114620'
# model_load_path = os.path.join(
#     project_dirname, 'results', dataset_name, date_result, date_result + 'model.keras')
# model_json_load_path = os.path.join(
#     project_dirname, 'results', dataset_name, date_result, date_result + 'model.json')
