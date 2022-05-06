import os
import pandas as pd
import pickle


def prep_and_pickle(file_path, save_name):
    all_file_names, all_text, all_keywords = [], [], []

    for file in os.listdir(file_path):
        if file.endswith(".txt"):
            full_path = os.path.join(file_path, file)
            all_file_names.append(full_path)
            with open(full_path, "r") as file:
                temp_text = file.read()
                all_text.append(temp_text)

            key_path = full_path.replace(".txt", ".key")

            with open(key_path, "r") as file:
                temp_keywords = file.readlines()
                stripped_keywords = [keyword.strip() for keyword in temp_keywords]
                all_keywords.append(stripped_keywords)

    df = pd.DataFrame(list(zip(all_file_names, all_text, all_keywords)))
    df.to_csv("{}_with_keywords.csv".format(save_name))

    marujo_data = (all_file_names, all_text, all_keywords)

    with open('{}.pickle'.format(save_name), 'wb') as handle:
        pickle.dump(marujo_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('marujo.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

    # print(marujo_data[0][0], marujo_data[1][0], marujo_data[2][0])
    # print("++++++++++++++++++++++++++")
    # print(b[0][0], b[1][0], b[2][0])


#prep_and_pickle("data/Marujo/", "Marujo")
# prep_and_pickle("data/Krapivin2009/", "Kravpivin2009")
prep_and_pickle("data/kdd/", "kdd-science")





#
# y = []
# with open("2.key", "r") as file:
#     txt = file.readlines()
#     print(file.read())
#
# print(type(txt))
# # for i in range(len(txt)):
# #     print('{}: {}'.format(i, txt[i]))
#
# for i, t in enumerate(txt):
#     print('{}: {}'.format(i, t))
#
# y.append(txt)