import torch


# ########## Path Config ##########
projects = ['Libreoffice', 'Eclipse', 'Gerrithub', 'OpenStack', 'Qt', 'Android', 'OpenDaylight']
project = projects[3]

# data_folder = "/Users/aobang/Documents/学习资料/毕业设计/数据集/my_data/Data"
data_folder = "E:/毕业设计/Data"
# data_folder = "/Volumes/Extreme SSD/毕业设计/Data"
# data_folder = "../../Data"


root = f"{data_folder}/{project}"
change_folder = "change"
change_directory_path = f'{root}/{change_folder}'
changes_root = f"{root}/changes"
diff_root = f'{root}/diff'
comment_root = f'{root}/comment'

result_folder = "../../Results"
result_project_folder = f"{result_folder}/{project}"

account_list_filepath = f'{root}/{project}_account_list.csv'
change_list_filepath = f'{root}/{project}_change_list.csv'
selected_change_list_filepath = f'{root}/{project}_selected_change_list.csv'

comment_list_filepath = f'{root}/{project}_comment_list.csv'

change_list_for_feature_cal_filepath = f'{root}/{project}_change_list_for_feature_cal.csv'


# ########## Feature Config ##########
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
                'project_avg_churn_abandoned', 'project_avg_file_abandoned', 'project_avg_comments_abandoned']
}

def get_initial_feature_list() -> [str]:
    feature_list = []
    for group in features_group:
        feature_list.extend(features_group[group])
    return feature_list


initial_feature_list = get_initial_feature_list()

train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
