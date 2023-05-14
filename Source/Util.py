import csv
import json
import os
import re

import pandas as pd

from Source.DataProcess.DataProcessConfig import *


def initialize_dirs():
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    for project in projects:
        if not os.path.exists(f'{data_folder}/{project}'):
            os.mkdir(f'{data_folder}/{project}')

        if not os.path.exists(f'{result_folder}/{project}'):
            os.mkdir(f'{result_folder}/{project}')

        for dirname in ['change', 'changes', 'diff', 'profile']:
            if not os.path.exists(f'{data_folder}/{project}/{dirname}'):
                os.mkdir(f'{data_folder}/{project}/{dirname}')


def initialize(path, file_header):
    with open(path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', dialect='excel')
        writer.writerow(file_header)
        csvfile.close()


def day_diff(date1, date2):
    if type(date1) == str:
        date1 = pd.to_datetime(date1)
    if type(date2) == str:
        date2 = pd.to_datetime(date2)

    diff = date1 - date2
    return diff.days + diff.seconds / 86400.0


def is_bot(project, name):
    project = project.lower()
    name = name.lower()
    if project in name or name == 'do not use':
        return True

    words = name.split()
    for word in ['bot', 'chatbot', 'ci', 'jenkins']:
        if word in words:
            return True

    return False


def safe_drop_column(df, columns):
    for col in columns:
        if col in df.columns:
            df.drop([col], axis=1, inplace=True)
        else:
            print("Error: column {0} not found".format(col))
    return df


def is_change_file(filename: str) -> bool:
    status = ["open", "closed", "merged", "abandoned"]
    for s in status:
        if s in filename:
            return True
    return False


def is_profile_file(filename: str) -> bool:
    pattern = r'profile_[0-9]+.json'
    return bool(re.fullmatch(pattern, filename))


def is_profile_details_file(filename: str) -> bool:
    pattern = r'profile_details_[0-9]+.json'
    return bool(re.fullmatch(pattern, filename))


def make_date(date):
    # date = re.sub(r"\.[0-9]+", "", date)
    # return datetime.datetime.strptime(date, format='%Y-%m-%d %H:%M:%S')
    return pd.to_datetime(date)


'''
these jsons are generally in format list of jsons
but sometimes they are list of one list of jsons
'''


def load_change_jsons(input_file):
    change_json = json.load(input_file)
    while type(change_json) == list and len(change_json) != 0 and type(change_json[0]) == list:
        change_json = change_json[0]
    return change_json


def toJSON(object):
    return json.dumps(object, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def subsystem_of(file_path):
    str_list = file_path.split('/')
    if len(str_list) == 1:
        return ''
    else:
        if str_list[0] == '':
            return str_list[1]
        return str_list[0]


def directory_of(file_path):
    return os.path.dirname(file_path)


def is_nonhuman(author_name):
    return 'CI' in author_name.split(' ') or \
           author_name == 'jenkins' or \
           author_name == 'Jenkins' or \
           author_name == 'Eclipse Genie'
    # try:
    #     reviewers.remove(author_id)
    #     project_set_instance.non_natural_human.add(author_id)
    # except KeyError:
    #     pass
