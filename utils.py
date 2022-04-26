import pandas as pd


def get_dataframe_from_csv(file_name):
    df = pd.read_csv(file_name)
    return df


def save_dataframe_to_csv(df, file_name, headers=[]):
    df.to_csv(file_name, headers=headers)


def calculate_mean_average_precision(predicted_keywords, true_keywords):
    all_precisions = []
    for i, keywords in enumerate(predicted_keywords):
        temp_precisions = []
        temp_correct = 0
        for j, kw in enumerate(keywords):
            if kw in true_keywords[i]:
                temp_correct += 1
                temp_precisions.append(temp_correct / (j + 1))
        if temp_correct == 0:
            all_precisions.append(0)
        else:
            all_precisions.append(sum(temp_precisions) / temp_correct)
    mean_average_precision = sum(all_precisions) / len(predicted_keywords)
    return mean_average_precision


def print_kwargs(**kwargs):
    print(kwargs['printt'])
    for x in kwargs['list']:
        print(x)
    for val in kwargs.values():
        print('VAVAV {}'.format(val))

# grouped_docs = ['ab', 'cd ', 'efg']
# temp_combined_doc = ''
# for doc in grouped_docs:
#     temp_combined_doc = temp_combined_doc + ' ' + doc
#
# quick = ' '.join(grouped_docs)
# print(temp_combined_doc)
# print(quick)
#
# print_kwargs(printt='PRINT THIS', list=['x', 'y', 'z'])