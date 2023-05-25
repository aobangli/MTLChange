import torch
import torch.nn as nn
from Source.ChangePrediction.TaskType import TaskType


train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_device = 'cpu'

# 原始数据集路径
original_path = '../data/Eclipse.csv'
# data_path = '../data/Eclipse.csv'
# data_path = '../data/Libreoffice.csv'
# data_path = '../data/OpenStack.csv'
# data_path = '../data/Libreoffice_new.csv'
data_path = '../data/Libreoffice_total_emb.csv'
result_output_path = '../data/output'

# 筛选指定时间的PR
end_date = "2022-01-01"


# 将subject、comment、msg转成384维的向量，每个维度为一列
subject_emb_features_cols = [f'subject_emb{i + 1}' for i in range(384)]
comment_emb_features_cols = [f'comment_emb{i + 1}' for i in range(384)]
msg_emb_features_cols = [f'msg_emb{i + 1}' for i in range(384)]
emb_features_cols = subject_emb_features_cols + comment_emb_features_cols + msg_emb_features_cols

# emb_features_cols = []

features_group = {
    'author': ['author_experience', 'author_is_reviewer', 'author_change_num', 'author_participation',
               'author_changes_per_week', 'author_avg_rounds', 'author_avg_duration', 'author_avg_scores',
               'author_merge_proportion',
               'author_degree_centrality', 'author_closeness_centrality', 'author_betweenness_centrality',
               'author_eigenvector_centrality', 'author_clustering_coefficient', 'author_k_coreness'],
    'reviewer': ['reviewer_experience',
                 # 'reviewer_is_author',
                 'reviewer_change_num', 'reviewer_participation',
                 'reviewer_avg_comments', 'reviewer_avg_files', 'reviewer_avg_rounds', 'reviewer_avg_duration',
                 'reviewer_avg_scores', 'reviewer_merge_proportion',
                 'reviewer_degree_centrality', 'reviewer_closeness_centrality', 'reviewer_betweenness_centrality',
                 'reviewer_eigenvector_centrality', 'reviewer_clustering_coefficient', 'reviewer_k_coreness'],
    'change': ['directory_num', 'subsystem_num', 'language_num', 'file_type_num', 'has_test',
               'has_feature', 'has_bug', 'has_document', 'has_improve', 'has_refactor',
               'subject_length', 'subject_readability',
               # 'subject_embedding',
               'msg_length', 'msg_readability',
               # 'msg_embedding',
               'lines_added', 'lines_deleted', 'segs_added', 'segs_deleted', 'segs_updated',
               'files_added', 'files_deleted', 'files_updated', 'modify_proportion', 'modify_entropy',
               'test_churn', 'non_test_churn',
               'reviewer_num', 'bot_reviewer_num',
               'comment_num', 'comment_length',
               # 'comment_embedding',
               'last_comment_mention'],
    'project': ['project_age', 'project_language_num', 'project_change_num', 'open_changes', 'project_author_num',
                'project_reviewer_num', 'project_team_size',
                'project_changes_per_author', 'project_changes_per_reviewer', 'project_changes_per_week',
                'project_change_avg_lines', 'project_change_avg_segs', 'project_change_avg_files',
                'project_add_per_week', 'project_del_per_week',
                'project_merge_proportion', 'project_avg_reviewers', 'project_avg_comments', 'project_avg_rounds',
                'project_avg_duration', 'project_avg_scores',
                'project_avg_rounds_merged', 'project_avg_duration_merged',
                'project_avg_churn_merged', 'project_avg_file_merged', 'project_avg_comments_merged',
                'project_avg_rounds_abandoned', 'project_avg_duration_abandoned',
                'project_avg_churn_abandoned', 'project_avg_file_abandoned', 'project_avg_comments_abandoned'],
    'embedding': emb_features_cols
}


def get_initial_feature_list() -> [str]:
    features = []
    for group in features_group:
        features.extend(features_group[group])
    return features


# 部分模型需要区分稀疏特征（分类型）
sparse_features_cols = ['author_is_reviewer', 'has_test', 'has_feature', 'has_bug', 'has_document', 'has_improve', 'has_refactor', 'last_comment_mention']

# 部分模型需要区分稠密特征（数值型）
dense_features_cols = list(filter(lambda x: x not in sparse_features_cols and x not in emb_features_cols, get_initial_feature_list()))

# 每个稀疏特征的值的类别数，部分模型中需要指定，用于为每个稀疏特征构造embedding的参数
# sparse_features_val_num = [2, 2, 2]
sparse_features_val_num = [2 for _ in range(len(sparse_features_cols))]

# 稠密特征个数
num_of_dense_feature = len(dense_features_cols)
# 向量数值特征个数
num_of_emb_feature = len(emb_features_cols)
# 数值型特征个数
num_of_numerical_feature = num_of_dense_feature + num_of_emb_feature

# 原始数据集里的所有label
# all_labels = ['num_of_reviewers', 'rounds', 'time', 'avg_score', 'status']
# 原始数据为回归型的label
regression_labels = ['num_of_reviewers', 'rounds', 'time', 'avg_score']
# 需要预测的label
target_labels = ['rounds', 'time', 'avg_score', 'status']
# target_labels = ['rounds']
# 要预测的label数量
num_of_labels = len(target_labels)
# 是否对回归任务label进行归一化
apply_minmax_to_regression = True
# 指定每种label的任务类型，与上面的labels一一对应
# label_types = [
#     TaskType.Regression,
#     TaskType.Regression,
#     TaskType.Regression,
#     TaskType.Binary_Classification
# ]
label_types = [
    TaskType.Multiple_Classification,
    TaskType.Multiple_Classification,
    TaskType.Multiple_Classification,
    TaskType.Binary_Classification
]
# label_types = [
#     TaskType.Multiple_Classification
# ]
# 指定每个label的激活函数，与上面labels一一对应
# 注意：对于二分类任务，widedeep和deepcross模型使用BCELoss()，tabtransformer模型使用BCEWithLogitsLoss()
# loss_functions_by_label = [AsymmetricLoss(gamma_neg=1, gamma_pos=2, clip=0)]
# loss_functions_by_label = [nn.BCELoss()]
# loss_functions_by_label = [nn.BCEWithLogitsLoss()]
# loss_functions_by_label = [nn.CrossEntropyLoss()]
# loss_functions_by_label = [
#     nn.MSELoss(),
#     nn.MSELoss(),
#     nn.MSELoss(),
#     nn.BCELoss()
# ]
loss_functions_by_label = [
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.BCELoss()
]
# 将数值型标签转换为二分类时的阈值，大于阈值取1，反之取0
binary_classification_label_threshold = {'num_of_reviewers': 1, 'rounds': 2, 'time': 2, 'avg_score': 1}

# 将数值型标签转换为多分类的阈值区间，从0开始取值，除了两侧的区间外，每个区间左开右闭
multi_classification_label_threshold = \
    {'num_of_reviewers': [2, 4], 'rounds': [1, 6], 'time': [1, 7], 'avg_score': [1, 1.75]}
# multi_classification_label_threshold = \
#     {'num_of_reviewers': [2, 4], 'rounds': [1, 6], 'time': [0.0417, 1, 7, 30], 'avg_score': [1, 1.75]}


# 根据阈值区间，将给定数值型数据映射为多分类特征
def classify_by_multi_threshold(data, thresholds):
    for index, threshold in enumerate(thresholds):
        if data <= threshold:
            return index
    return len(thresholds)


# 根据任务类型，获取输出维度list，二分类任务与回归任务为1，多分类任务为类别数
def get_task_out_dims():
    dims = []
    for index, task_name in enumerate(target_labels):
        task_type = label_types[index]
        if task_type == TaskType.Multiple_Classification:
            dim = len(multi_classification_label_threshold[task_name]) + 1
        else:
            dim = 1
        dims.append(dim)
    return dims


task_out_dims = get_task_out_dims()

# 所有特征列表
feature_list = dense_features_cols + emb_features_cols + sparse_features_cols
# 所有特征和标签列表
feature_label_list = feature_list + target_labels

# 对label做归一化时，保存对应的scaler，方便后面对模型预测结果还原
scalers_buffer = {}

# 训练时是否打印日志到控制台
print_batch_log = False
print_epoch_log = True
