import os
from datetime import timedelta

import numpy as np
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import textstat
from tqdm import tqdm

from Source.DataProcess.SimpleParser import *

from Source.DataProcess.SimpleParser import Change
from Source.Util import *

account_list_df = pd.read_csv(account_list_filepath)
account_list_df['registered_on'] = pd.to_datetime(account_list_df['registered_on'])
account_list_df['name'] = account_list_df['name'].apply(str)

if os.path.exists(change_list_for_feature_cal_filepath):
    change_list_df = joblib.load(change_list_for_feature_cal_filepath)
else:
    print("prepare change list...")

    change_list_df = joblib.load(selected_change_list_filepath)
    change_list_df = change_list_df.sort_values(by=['change_id']).reset_index(drop=True)
    for col in ['created', 'updated', 'closed']:
        change_list_df.loc[:, col] = change_list_df[col].apply(pd.to_datetime)

    # 按时间筛选
    # start_date = pd.to_datetime('2021-01-01')
    # end_date = pd.to_datetime('2022-12-31')
    # change_list_df = change_list_df[(change_list_df['created'] >= start_date) & (change_list_df['created'] <= end_date)]
    # print(f"Reduce to {change_list_df.shape[0]} changes")

    for change_number in tqdm(change_list_df['change_id']):
        diff_path = os.path.join(diff_root, f"{project}_{change_number}_diff.json")
        diff_json = json.load(open(diff_path, 'r'))
        files = list(diff_json.values())[0].values()
        total_lines = 0
        total_segs = 0
        for file in files:
            total_lines += file['meta_a']['lines'] if 'meta_a' in file.keys() else 0
            total_segs += len(file['content'])
        change_list_df['line_num'] = total_lines
        change_list_df['seg_num'] = total_segs
    joblib.dump(change_list_df, change_list_for_feature_cal_filepath)

comment_list_df = pd.read_csv(comment_list_filepath)
comment_list_df['updated'] = pd.to_datetime(comment_list_df['updated'])
comment_list_df['message'] = comment_list_df['message'].apply(str)

lookback = 60
social_network_lookback = 60

# 默认值
default_merge_ratio = 0.5
default_duration = 1
default_rounds = 1
default_score = 1


def main():
    print("start feature calculation...")

    features_list = initial_feature_list
    file_header = ["project", "change_id", 'created', 'subject'] + features_list + \
                  ['avg_score', 'status', 'time', 'rounds'] + ['comment_content', 'msg_content']

    output_file_name = f"{root}/{project}.csv"

    change_numbers = change_list_df['change_id'].values
    if os.path.exists(output_file_name):
        old_pd = pd.read_csv(output_file_name)
        old_change_numbers = old_pd['change_id'].values
        current_change_number = change_numbers[~np.in1d(change_numbers, old_change_numbers)]
    else:
        initialize(output_file_name, file_header)
        current_change_number = change_numbers

    csv_file = open(output_file_name, "a", newline='', encoding='utf-8')
    file_writer = csv.writer(csv_file, dialect='excel')

    # it is important to calculate in sorted order of created.
    # Change numbers are given in increasing order of creation time
    count = 0
    for change_number in tqdm(current_change_number):
        # print(change_number)

        filename = f'{project}_{change_number}_change.json'
        filepath = os.path.join(changes_root, filename)
        if not os.path.exists(filepath):
            print(f'{filename} does not exist')
            continue

        change = Change(json.load(open(filepath, "r")))
        if not change.is_real_change():
            continue

        current_date = pd.to_datetime(change.first_revision.created)
        calculator = FeatureCalculator(change, current_date)

        author_features = calculator.author_features
        reviewer_features = calculator.reviewer_features
        change_features = calculator.change_features
        project_features = calculator.project_features

        status = 1 if change.status == 'MERGED' else 0
        feature_vector = [
            change.project, change.change_number, change.created, change.subject,

            author_features['author_experience'], author_features['author_is_reviewer'],
            author_features['author_change_num'], author_features['author_participation'],
            author_features['author_changes_per_week'], author_features['author_avg_rounds'],
            author_features['author_avg_duration'], author_features['author_avg_scores'],
            author_features['author_merge_proportion'],
            author_features['author_degree_centrality'], author_features['author_closeness_centrality'],
            author_features['author_betweenness_centrality'], author_features['author_eigenvector_centrality'],
            author_features['author_clustering_coefficient'], author_features['author_k_coreness'],

            reviewer_features['reviewer_experience'], reviewer_features['reviewer_change_num'],
            reviewer_features['reviewer_participation'], reviewer_features['reviewer_avg_comments'],
            reviewer_features['reviewer_avg_files'], reviewer_features['reviewer_avg_rounds'],
            reviewer_features['reviewer_avg_duration'], reviewer_features['reviewer_avg_scores'],
            reviewer_features['reviewer_merge_proportion'],
            reviewer_features['reviewer_degree_centrality'], reviewer_features['reviewer_closeness_centrality'],
            reviewer_features['reviewer_betweenness_centrality'], reviewer_features['reviewer_eigenvector_centrality'],
            reviewer_features['reviewer_clustering_coefficient'], reviewer_features['reviewer_k_coreness'],

            change_features['directory_num'], change_features['subsystem_num'], change_features['language_num'],
            change_features['file_type_num'], change_features['has_test'],
            change_features['has_feature'], change_features['has_bug'], change_features['has_document'],
            change_features['has_improve'], change_features['has_refactor'],
            change_features['subject_length'], change_features['subject_readability'],
            change_features['msg_length'], change_features['msg_readability'],
            change_features['lines_added'], change_features['lines_deleted'],
            change_features['segs_added'], change_features['segs_deleted'], change_features['segs_updated'],
            change_features['files_added'], change_features['files_deleted'], change_features['files_updated'],
            change_features['modify_proportion'], change_features['modify_entropy'],
            change_features['test_churn'], change_features['non_test_churn'],
            reviewer_features['reviewer_num'], reviewer_features['bot_reviewer_num'],
            change_features['comment_num'], change_features['comment_length'], change_features['last_comment_mention'],

            project_features['project_age'], project_features['project_language_num'],
            project_features['project_change_num'], project_features['open_changes'],
            project_features['project_author_num'], project_features['project_reviewer_num'],
            project_features['project_team_size'], project_features['project_changes_per_author'],
            project_features['project_changes_per_reviewer'], project_features['project_changes_per_week'],
            project_features['project_change_avg_lines'], project_features['project_change_avg_segs'],
            project_features['project_change_avg_files'],
            project_features['project_add_per_week'], project_features['project_del_per_week'],
            project_features['project_merge_proportion'], project_features['project_avg_reviewers'],
            project_features['project_avg_comments'], project_features['project_avg_rounds'],
            project_features['project_avg_duration'], project_features['project_avg_scores'],
            project_features['project_avg_rounds_merged'], project_features['project_avg_duration_merged'],
            project_features['project_avg_churn_merged'], project_features['project_avg_file_merged'],
            project_features['project_avg_comments_merged'],
            project_features['project_avg_rounds_abandoned'], project_features['project_avg_duration_abandoned'],
            project_features['project_avg_churn_abandoned'], project_features['project_avg_file_abandoned'],
            project_features['project_avg_comments_abandoned'],

            change.avg_score,
            status,
            day_diff(change.closed, change.created),
            len(change.revisions),

            # for embedding
            change_features['comment_content'],
            change_features['msg_content']
        ]
        file_writer.writerow(feature_vector)

        count += 1
        if count % 100 == 0:
            csv_file.flush()
            # break

    csv_file.close()

    features = pd.read_csv(output_file_name)
    features.drop_duplicates(['change_id'], inplace=True)
    features.sort_values(by=['change_id']).to_csv(output_file_name, index=False, float_format='%.2f')


class FeatureCalculator:
    def __init__(self, change, current_date):
        self.change = change
        self.project = change.project
        self.current_date = current_date
        self.lookback_date = current_date - timedelta(days=lookback)
        self.changes_now = change_list_df[change_list_df['created'] < self.current_date]
        # self.changes_during_lookback = self.changes_now[self.changes_now['updated'] >= self.lookback_date]

    @property
    def author_features(self):
        features = {}
        owner = self.change.owner
        registered_on = account_list_df[account_list_df['account_id'] == owner]['registered_on'].values
        authors_changes = self.changes_now[self.changes_now['owner'] == owner]
        active_changes = self.changes_now

        features['author_change_num'] = authors_changes.shape[0]

        if len(registered_on) == 0 or registered_on[0] > self.current_date:
            if authors_changes.shape[0] > 0:
                features['author_experience'] = max(0, day_diff(self.current_date, authors_changes['created'].min()))
            else:
                features['author_experience'] = 0
        else:
            features['author_experience'] = day_diff(self.current_date, registered_on[0])

        features['author_is_reviewer'] = active_changes[active_changes['reviewers']
            .apply(lambda x: owner in x)].shape[0] > 0

        if active_changes.shape[0] > 0:
            features['author_participation'] = float(authors_changes.shape[0]) / active_changes.shape[0]
        else:
            features['author_participation'] = 0

        finished_works = authors_changes[authors_changes['updated'] <= self.current_date]
        merged_works = finished_works[finished_works['status'] == 'MERGED']

        if finished_works.shape[0] > 0:
            features['author_merge_proportion'] = float(merged_works.shape[0]) / finished_works.shape[0]
        else:
            features['author_merge_proportion'] = default_merge_ratio

        first_date = authors_changes['created'].min() if authors_changes.shape[0] > 0 else self.lookback_date
        weeks = max(1, day_diff(self.current_date, max(first_date, self.lookback_date)) / 7.0)
        features['author_changes_per_week'] = float(finished_works.shape[0]) / weeks

        features['author_avg_rounds'] = np.mean(authors_changes['revision_num'].values) if authors_changes.shape[0] > 0 \
            else (np.mean(active_changes['revision_num'].values) if active_changes.shape[0] > 0 else default_rounds)

        features['author_avg_duration'] = np.mean(authors_changes['duration'].values) if authors_changes.shape[0] > 0 \
            else (np.mean(active_changes['duration'].values) if active_changes.shape[0] > 0 else default_duration)

        features['author_avg_scores'] = np.mean(authors_changes['avg_score'].values) if authors_changes.shape[0] > 0 \
            else (np.mean(active_changes['avg_score'].values) if active_changes.shape[0] > 0 else default_score)

        social_network_start_time = self.current_date - timedelta(days=social_network_lookback)
        authors_social_features = \
            cal_author_social_features(active_changes, owner, social_network_start_time, self.project)
        features['author_degree_centrality'] = authors_social_features['degree_centrality']
        features['author_closeness_centrality'] = authors_social_features['closeness_centrality']
        features['author_betweenness_centrality'] = authors_social_features['betweenness_centrality']
        features['author_eigenvector_centrality'] = authors_social_features['eigenvector_centrality']
        features['author_clustering_coefficient'] = authors_social_features['clustering_coefficient']
        features['author_k_coreness'] = authors_social_features['k_coreness']

        # features['author_degree_centrality'] = 0
        # features['author_closeness_centrality'] = 0
        # features['author_betweenness_centrality'] = 0
        # features['author_eigenvector_centrality'] = 0
        # features['author_clustering_coefficient'] = 0
        # features['author_k_coreness'] = 0
        return features

    @property
    def reviewer_features(self):
        features = {}
        reviewer_list = self.change.reviewers
        active_changes = self.changes_now

        avg_experience = 0
        real_reviewers = []
        bot = 0
        # 所有评审人评审过的PR的id的集合
        related_change_ids = set()
        for reviewer_id in reviewer_list:
            reviewer = account_list_df[account_list_df['account_id'] == reviewer_id]
            registered_on = reviewer['registered_on'].values
            if len(registered_on) == 0 or self.current_date < registered_on[0]:
                continue

            if is_bot(self.project, reviewer['name'].values[0]):
                bot += 1
                continue
            real_reviewers.append(reviewer_id)

            experience = day_diff(self.current_date, registered_on[0])
            avg_experience += experience

            if active_changes.shape[0] > 0:
                related_changes = active_changes[active_changes['reviewers'].apply(lambda x: reviewer_id in x)]
                related_change_ids.update(related_changes['change_id'].values)

        if len(real_reviewers) > 0:
            avg_experience /= len(real_reviewers)

        features['reviewer_experience'] = avg_experience

        features['reviewer_num'] = len(real_reviewers)
        features['bot_reviewer_num'] = bot

        total_related_changes = active_changes[active_changes['change_id'].isin(related_change_ids)]

        features['reviewer_change_num'] = total_related_changes.shape[0]

        features['reviewer_participation'] = float(total_related_changes.shape[0]) / active_changes.shape[0] \
            if active_changes.shape[0] > 0 else 0

        merged_related_changes = total_related_changes[total_related_changes['status'] == 'MERGED']

        features['reviewer_merge_proportion'] = float(merged_related_changes.shape[0]) / total_related_changes.shape[0] \
            if total_related_changes.shape[0] > 0 else default_merge_ratio

        features['reviewer_avg_comments'] = np.mean(total_related_changes['comment_num'].values) \
            if total_related_changes.shape[0] > 0 \
            else (np.mean(active_changes['comment_num'].values) if active_changes.shape[0] > 0 else 0)

        features['reviewer_avg_files'] = np.mean(total_related_changes['file_num'].values) \
            if total_related_changes.shape[0] > 0 \
            else (np.mean(active_changes['file_num'].values) if active_changes.shape[0] > 0 else 0)

        features['reviewer_avg_rounds'] = np.mean(total_related_changes['revision_num'].values) \
            if total_related_changes.shape[0] > 0 \
            else (np.mean(active_changes['revision_num'].values) if active_changes.shape[0] > 0 else default_rounds)

        features['reviewer_avg_duration'] = np.mean(total_related_changes['duration'].values) \
            if total_related_changes.shape[0] > 0 \
            else (np.mean(active_changes['duration'].values) if active_changes.shape[0] > 0 else default_duration)

        features['reviewer_avg_scores'] = np.mean(total_related_changes['avg_score'].values) \
            if total_related_changes.shape[0] > 0 \
            else (np.mean(active_changes['avg_score'].values) if active_changes.shape[0] > 0 else default_score)

        social_network_start_time = self.current_date - timedelta(days=social_network_lookback)
        reviewer_social_features = \
            cal_reviewers_social_features(active_changes, real_reviewers, social_network_start_time, self.project)
        features['reviewer_degree_centrality'] = reviewer_social_features['degree_centrality']
        features['reviewer_closeness_centrality'] = reviewer_social_features['closeness_centrality']
        features['reviewer_betweenness_centrality'] = reviewer_social_features['betweenness_centrality']
        features['reviewer_eigenvector_centrality'] = reviewer_social_features['eigenvector_centrality']
        features['reviewer_clustering_coefficient'] = reviewer_social_features['clustering_coefficient']
        features['reviewer_k_coreness'] = reviewer_social_features['k_coreness']

        return features

    @property
    def change_features(self):
        features = {}

        subject_features = self.subject_features
        features.update(subject_features)

        message_features = self.message_features
        features.update(message_features)

        features['comment_num'] = self.change.comment_num
        current_comments_df = comment_list_df[comment_list_df['change_id'] == self.change.change_number]
        current_comments_df = current_comments_df[current_comments_df['patch_set'] == 1]
        if current_comments_df.shape[0] == 0:
            features['comment_length'] = 0
            features['last_comment_mention'] = False
            # for comment embedding
            features['comment_content'] = ''
        else:
            comment_word_num = 0
            current_comments_df = current_comments_df.sort_values(by=['updated'])
            comment_messages = current_comments_df['message']
            # for comment embedding
            features['comment_content'] = ''.join(comment_messages)
            for message in comment_messages:
                comment_word_num += len(message.split())
            features['comment_length'] = comment_word_num

            last_comment_message = comment_messages.iloc[-1]
            features['last_comment_mention'] = ('@' in last_comment_message)

        file_features = self.file_features
        features.update(file_features)
        return features

    @property
    def subject_features(self):
        subject = self.change.subject.lower()

        features = {'has_feature': False, 'has_bug': False, 'has_document': False, 'has_improve': False,
                    'has_refactor': False, 'subject_length': len(subject.split()),
                    'subject_readability': textstat.textstat.coleman_liau_index(self.change.subject)}

        # subject embedding

        for word in ['feat', 'feature']:
            if word in subject:
                features['has_feature'] = True
        for word in ['fix', 'bug', 'defect']:
            if word in subject:
                features['has_bug'] = True
                return features
        for word in ['doc', 'copyright', 'license']:
            if word in subject:
                features['has_document'] = True
                return features
        for word in ['improve']:
            if word in subject:
                features['has_improve'] = True
                return features
        for word in ['refactor']:
            if word in subject:
                features['has_refactor'] = True
        # features['has_feature'] = True
        return features

    @property
    def message_features(self):
        features = {}
        first_revision = self.change.first_revision
        msg = first_revision.commit_message if first_revision is not None else ''
        features['msg_length'] = len(msg.split())
        features['msg_readability'] = textstat.textstat.coleman_liau_index(msg)

        # for message embedding
        features['msg_content'] = msg

        return features

    @property
    def file_features(self):
        features = {}
        files = self.change.files

        lines_added = lines_deleted = 0
        files_added = files_deleted = files_updated = 0
        test_lines_added = test_lines_deleted = 0
        non_test_lines_added = non_test_lines_deleted = 0

        directories = set()
        subsystems = set()
        for file in files:
            if 'test' in file.name.lower():
                test_lines_added += file.lines_inserted
                test_lines_deleted += file.lines_deleted
            else:
                non_test_lines_added += file.lines_inserted
                non_test_lines_deleted += file.lines_deleted
            lines_added += file.lines_inserted
            lines_deleted += file.lines_deleted

            if file.status == 'D': files_deleted += 1
            if file.status == 'A': files_added += 1
            if file.status == 'M': files_updated += 1

            names = file.path.split('/')
            if len(names) > 1:
                directories.update([names[-2]])
                subsystems.update(names[0])

        features['directory_num'] = len(directories)
        features['subsystem_num'] = len(subsystems)

        features['test_churn'] = test_lines_added + test_lines_deleted
        features['non_test_churn'] = non_test_lines_added + non_test_lines_deleted

        features['has_test'] = (test_lines_added + test_lines_deleted > 0)

        features['lines_added'] = lines_added
        features['lines_deleted'] = lines_deleted

        features['files_added'] = files_added
        features['files_deleted'] = files_deleted
        features['files_updated'] = files_updated

        # Entropy is defined as: −Sum(k=1 to n)(pk∗log2pk). Note that n is number of files
        # modified in the change, and pk is calculated as the proportion of lines modified in file k among
        # lines modified in this code change.
        modify_entropy = 0
        if lines_added + lines_deleted > 0:
            for file in files:
                lines_changed_in_file = file.lines_deleted + file.lines_inserted
                if lines_changed_in_file:
                    pk = float(lines_changed_in_file) / (lines_added + lines_deleted)
                    modify_entropy -= pk * np.log2(pk)

        features['modify_entropy'] = modify_entropy

        features['language_num'] = len(self.change.language_set)
        features['file_type_num'] = self.change.file_type_num

        diff_path = os.path.join(diff_root, f"{project}_{self.change.change_number}_diff.json")
        diff_json = json.load(open(diff_path, 'r'))

        segs_added = segs_deleted = segs_updated = 0
        prev_total_line = 0

        files = list(diff_json.values())[0].values()
        for file in files:
            prev_total_line += file['meta_a']['lines'] if 'meta_a' in file.keys() else 0
            for content in file['content']:
                change_type = list(content.keys())
                if change_type == ['a']:
                    segs_deleted += 1
                elif change_type == ['a', 'b']:
                    segs_updated += 1
                elif change_type == ['b']:
                    segs_added += 1

        features['modify_proportion'] = float((lines_added + lines_deleted) / prev_total_line)\
            if prev_total_line > 0 else 1
        features['segs_added'] = segs_added
        features['segs_deleted'] = segs_deleted
        features['segs_updated'] = segs_updated

        return features

    @property
    def project_features(self):
        features = {}
        project_changes = self.changes_now[self.changes_now['project'] == self.project]

        features['project_age'] = day_diff(self.current_date, project_changes['created'].min()) \
            if project_changes.shape[0] > 0 else 0

        language_set = set()
        for languages in project_changes['languages']:
            language_set = language_set.union(languages)
        features['project_language_num'] = len(language_set)

        features['project_change_num'] = project_changes.shape[0]

        features['open_changes'] = project_changes[self.current_date < project_changes['closed']].shape[0]

        author_set = set(project_changes['owner'])
        reviewer_set = set()
        for reviewers in project_changes['reviewers']:
            reviewer_set = reviewer_set.union(reviewers)
        features['project_author_num'] = len(author_set)
        features['project_reviewer_num'] = len(reviewer_set)

        features['project_team_size'] = len(author_set.union(reviewer_set))

        features['project_changes_per_author'] = float(project_changes.shape[0]) / len(author_set) \
            if len(author_set) > 0 else 0

        features['project_changes_per_reviewer'] = float(project_changes.shape[0]) / len(reviewer_set) \
            if len(reviewer_set) > 0 else 0

        weeks = max(1, day_diff(self.current_date, max(project_changes['created'].min(), self.lookback_date)) / 7.0) \
            if project_changes.shape[0] > 0 else 1

        features['project_changes_per_week'] = float(project_changes.shape[0]) / weeks

        features['project_change_avg_lines'] = np.mean(project_changes['line_num']) \
            if project_changes.shape[0] > 0 else 0

        features['project_change_avg_segs'] = np.mean(project_changes['seg_num']) \
            if project_changes.shape[0] > 0 else 0

        features['project_change_avg_files'] = np.mean(project_changes['file_num']) \
            if project_changes.shape[0] > 0 else 0

        features['project_add_per_week'] = float(np.sum(project_changes['added_lines'])) / weeks
        features['project_del_per_week'] = float(np.sum(project_changes['deleted_lines'])) / weeks

        merged_project_changes = project_changes[project_changes['status'] == 'MERGED']
        abandoned_project_changes = project_changes[project_changes['status'] == 'ABANDONED']

        features['project_merge_proportion'] = float(merged_project_changes.shape[0]) / project_changes.shape[0] \
            if project_changes.shape[0] > 0 else default_merge_ratio

        features['project_avg_reviewers'] = np.mean(len(project_changes['reviewers'])) \
            if project_changes.shape[0] > 0 else 0
        features['project_avg_comments'] = np.mean(project_changes['comment_num']) \
            if project_changes.shape[0] > 0 else 0
        features['project_avg_rounds'] = np.mean(project_changes['revision_num']) \
            if project_changes.shape[0] > 0 else default_rounds
        features['project_avg_duration'] = np.mean(project_changes['duration']) \
            if project_changes.shape[0] > 0 else default_duration
        features['project_avg_scores'] = np.mean(project_changes['avg_score']) \
            if project_changes.shape[0] > 0 else default_score

        features['project_avg_rounds_merged'] = np.mean(merged_project_changes['revision_num']) \
            if merged_project_changes.shape[0] > 0 else default_rounds

        features['project_avg_duration_merged'] = np.mean(merged_project_changes['duration']) \
            if merged_project_changes.shape[0] > 0 else default_duration

        features['project_avg_churn_merged'] = \
            np.mean(merged_project_changes['added_lines']) + np.mean(merged_project_changes['deleted_lines']) \
                if merged_project_changes.shape[0] > 0 else 0

        features['project_avg_file_merged'] = np.mean(merged_project_changes['file_num']) \
            if merged_project_changes.shape[0] > 0 else 0

        features['project_avg_comments_merged'] = np.mean(merged_project_changes['comment_num']) \
            if merged_project_changes.shape[0] > 0 else 0

        features['project_avg_rounds_abandoned'] = np.mean(abandoned_project_changes['revision_num']) \
            if abandoned_project_changes.shape[0] > 0 else default_rounds

        features['project_avg_duration_abandoned'] = np.mean(abandoned_project_changes['duration']) \
            if abandoned_project_changes.shape[0] > 0 else default_duration

        features['project_avg_churn_abandoned'] = \
            np.mean(merged_project_changes['added_lines']) + np.mean(abandoned_project_changes['deleted_lines']) \
                if abandoned_project_changes.shape[0] > 0 else 0

        features['project_avg_file_abandoned'] = np.mean(abandoned_project_changes['file_num']) \
            if abandoned_project_changes.shape[0] > 0 else 0

        features['project_avg_comments_abandoned'] = np.mean(abandoned_project_changes['comment_num']) \
            if abandoned_project_changes.shape[0] > 0 else 0

        return features


def cal_author_social_features(df, author, start_time, current_project=None):
    graph_df = df[df['project'] == current_project] if current_project is not None else df
    graph_df = graph_df[graph_df['created'] >= start_time]
    owners, reviewers_list = graph_df['owner'].values, graph_df['reviewers'].values

    graph = nx.Graph()
    for index in range(graph_df.shape[0]):
        owner, reviewers = owners[index], reviewers_list[index]
        for reviewer in reviewers:
            if reviewer == owner:
                continue
            try:
                graph[owner][reviewer]['weight'] += 1
            except (KeyError, IndexError):
                graph.add_edge(owner, reviewer, weight=1)

    network = SocialNetwork(graph, author)
    # network.show_graph()
    return {
        'degree_centrality': network.degree_centrality(),
        'closeness_centrality': network.closeness_centrality(),
        'betweenness_centrality': network.betweenness_centrality(),
        'eigenvector_centrality': network.eigenvector_centrality(),
        'clustering_coefficient': network.clustering_coefficient(),
        'k_coreness': network.k_coreness()
    }


def cal_reviewers_social_features(df, reviewers, start_time, current_project=None):
    graph_df = df[df['project'] == current_project] if current_project is not None else df
    graph_df = graph_df[graph_df['created'] >= start_time]

    owners, reviewers_list = graph_df['owner'].values, graph_df['reviewers'].values
    graph = nx.Graph()
    for index in range(graph_df.shape[0]):
        owner, reviewers = owners[index], reviewers_list[index]
        for reviewer in reviewers:
            if reviewer == owner:
                continue
            try:
                graph[owner][reviewer]['weight'] += 1
            except (KeyError, IndexError):
                graph.add_edge(owner, reviewer, weight=1)

    degree_centrality = 0
    closeness_centrality = 0
    betweenness_centrality = 0
    eigenvector_centrality = 0
    clustering_coefficient = 0
    k_coreness = 0

    for reviewer in reviewers:
        network = SocialNetwork(graph, reviewer)
        degree_centrality += network.degree_centrality()
        closeness_centrality += network.closeness_centrality()
        betweenness_centrality += network.betweenness_centrality()
        eigenvector_centrality += network.eigenvector_centrality()
        clustering_coefficient += network.clustering_coefficient()
        k_coreness += network.k_coreness()

    # network.show_graph()
    return {
        'degree_centrality': float(degree_centrality) / len(reviewers) if len(reviewers) > 0 else 0,
        'closeness_centrality': float(closeness_centrality) / len(reviewers) if len(reviewers) > 0 else 0,
        'betweenness_centrality': float(betweenness_centrality) / len(reviewers) if len(reviewers) > 0 else 0,
        'eigenvector_centrality': float(eigenvector_centrality) / len(reviewers) if len(reviewers) > 0 else 0,
        'clustering_coefficient': float(clustering_coefficient) / len(reviewers) if len(reviewers) > 0 else 0,
        'k_coreness': float(k_coreness) / len(reviewers) if len(reviewers) > 0 else 0
    }


class SocialNetwork:
    def __init__(self, graph, owner):
        self.graph = graph
        self.owner = owner
        self.lcc = self.largest_connected_component()

    def show_graph(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()

    def largest_connected_component(self):
        try:
            return self.graph.subgraph(max(nx.connected_components(self.graph), key=len))
        except:
            return self.graph

    def degree_centrality(self):
        nodes_dict = nx.degree_centrality(self.lcc)
        try:
            return nodes_dict[self.owner]
        except:
            return 0

    def closeness_centrality(self):
        try:
            return nx.closeness_centrality(self.lcc, u=self.owner)
        except:
            return 0

    def betweenness_centrality(self):
        nodes_dict = nx.betweenness_centrality(self.lcc, weight='weight')
        try:
            return nodes_dict[self.owner]
        except:
            return 0

    def eigenvector_centrality(self):
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.lcc)
            try:
                return eigenvector_centrality[self.owner]
            except:
                return 0
        except:
            return 0

    def clustering_coefficient(self):
        try:
            return nx.clustering(self.lcc, nodes=self.owner, weight='weight')
        except:
            return 0

    def k_coreness(self):
        nodes_dict = nx.core_number(self.lcc)
        try:
            return nodes_dict[self.owner]
        except:
            return 0


if __name__ == '__main__':
    main()
