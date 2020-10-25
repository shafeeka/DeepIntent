import pickle
import os

def save_pkl_data(path, data):
    with open(path, 'wb') as fo:
        pickle.dump(data, fo)

def load_pkl_data(path):
    with open(path, 'rb') as fi:
        data = pickle.load(fi)
    return data

def insert_outlier_label(data):
    outlier_label = [-1,-1,-1,-1,-1,-1,-1,-1] #all inliers
    for i in range(len(data)):
        data[i].append(outlier_label)
    return data

def insert_unique_id(data):
    for i in range(len(data)):
        data[i].append(i)
    return data

def prepare_training(path_training):
    result = load_pkl_data(path_training) #tuple,list
    v_l_id, data = result #data a list of : img_data as tuple, tokens as list, perms as list
    data_with_label = insert_outlier_label(data)
    data_id = insert_unique_id(data_with_label) #[(img_data),[tokens], [perms], [outlier_labels], id]

    return v_l_id,data_id

def prepare_testing(path_testing):
    result = load_pkl_data(path_testing) #tuple,list
    v_l_id, data = result #data a list of : img_data as tuple, tokens as list, perms as list, outlier_labels as list
    data_id = insert_unique_id(data) #[(img_data),[tokens], [perms], [outlier_labels], id]

    return v_l_id,data_id
    
def pkl_files():
    path_current = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(path_current, 'data', 'total')
    # path
    path_training=os.path.join(path_data, 'outlier.training.pkl')
    path_testing_benign=os.path.join(path_data, 'outlier.testing.benign.pkl')
    path_testing_malicious=os.path.join(path_data, 'outlier.testing.malicious.pkl')
    path_combined_output = os.path.join(path_data, 'outlier.combined.pkl')

    v_l_id_training_benign, data_training_benign = prepare_training(path_training)
    v_l_id_testing_benign, data_testing_benign = prepare_testing(path_testing_benign)
    v_l_id_testing_malicious, data_testing_malicious = prepare_testing(path_testing_malicious)
    #tuple are the same for all 3, including vocab dict
    data_training_benign.extend(data_testing_benign)
    data_training_benign.extend(data_testing_malicious)
    save_pkl_data(path_combined_output, [v_l_id_training_benign,data_training_benign])




def main():
    pkl_files()


if __name__ == '__main__':
    main()