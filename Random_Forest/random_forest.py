import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# split_train_test splits the data into training set and test set
def split_train_test(data, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * data.shape[0])
    indices = data.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    data_test = data.loc[test_indices]
    data_train = data.drop(test_indices)
    return data_train, data_test

# ==================================Decision tree=================================
# classify_min_data returns the label with max counts
def classify_min_data(data):
    labels,counts = np.unique(data.iloc[:,-1], return_counts = True)
    return labels[counts.argmax()]

# is_pure returns True if the data is pure
def is_pure(data):
    if len(np.unique(data.iloc[:,-1])) == 1:
        return True
    else:
        return False

# split_data returns 2 groups of data, one of which is below the split_value and the other one is above the split_value
def split_data(data, split_feature, split_value):
    split_fecture_values = data[split_feature]
    return data[split_fecture_values <= split_value], data[split_fecture_values > split_value]

# calculate_entropy calculates the entropy H(Y|feature <= threshold) or H(Y|feature > threshold)
def calculate_entropy(data):
    _, unique_classes_counts = np.unique(data.iloc[:,-1], return_counts = True)
    probabilities = unique_classes_counts/unique_classes_counts.sum()
    return sum(probabilities*-np.log2(probabilities))

# calculate_overall_entropy calculates the entropy H(Y|feature)
def calculate_overall_entropy(data_below, data_above):
    p_data_below = len(data_below)/(len(data_below)+len(data_above))
    p_data_above = len(data_above)/(len(data_below)+len(data_above))
    return p_data_below * calculate_entropy(data_below)+p_data_above * calculate_entropy(data_above)

# determine_best_split returns the best split point
def determine_best_split(data, feature_importance, num_samples_total, random_features, random_splits = None):
    overall_entropy = float('inf')
    original_entropy = calculate_entropy(data)
    if random_splits == None:
        for split_feture in random_features:
            for split_value in np.unique(data[split_feture]).tolist():
                data_below, data_above = split_data(data, split_feture, split_value)
                curr_overall_entropy = calculate_overall_entropy(data_below, data_above)
                if curr_overall_entropy <= overall_entropy:
                    overall_entropy = curr_overall_entropy
                    best_split_feature = split_feture
                    best_split_value = split_value
                    best_group = (data_below, data_above)
        if best_split_feature in feature_importance:
            feature_importance[best_split_feature][0] += (original_entropy - overall_entropy)*(len(data)/num_samples_total)
            feature_importance[best_split_feature][1] += 1
        else:
            feature_importance[best_split_feature] = [(original_entropy - overall_entropy)*(len(data)/num_samples_total), 1]
    else:
        # random_split_features = random.sample(population=random_features, k=random_splits)
        for random_split_feature in random_features:
            random_split_value = random.choice(np.unique(data[random_split_feature]).tolist())
            data_below, data_above = split_data(data, random_split_feature, random_split_value)
            curr_overall_entropy = calculate_overall_entropy(data_below, data_above)
            if curr_overall_entropy <= overall_entropy:
                overall_entropy = curr_overall_entropy
                best_split_feature = random_split_feature
                best_split_value = random_split_value
                best_group = (data_below, data_above)
        if best_split_feature in feature_importance:
            feature_importance[best_split_feature][0] += (original_entropy - overall_entropy)*(len(data)/num_samples_total)
            feature_importance[best_split_feature][1] += 1
        else:
            feature_importance[best_split_feature] = [(original_entropy - overall_entropy)*(len(data)/num_samples_total), 1]
    return best_split_feature, best_split_value, best_group, feature_importance

# build_decision_tree build the decision tree model with random features, and return the feature importance
def build_decision_tree(data, feature_importance, num_samples_total, current_depth = 0, min_sample_size = 2, max_depth = 1000, random_features = None, random_splits = None):
    features_indices = data.columns.tolist()[:-1]
    if current_depth == 0: 
        if random_features != None and random_features <= len(features_indices):
            random_features = random.sample(population=features_indices, k=random_features)
        else:
            random_features = features_indices
    
    # if current_depth == 0:
    #     num_samples_total = len(data)
    
    if is_pure(data) or data.shape[0] < min_sample_size or current_depth == max_depth:
        return classify_min_data(data),feature_importance
    else:
        current_depth += 1
        split_feature, split_value, split_group, feature_importance = determine_best_split(data, feature_importance, num_samples_total, random_features, random_splits)
        data_below, data_above = split_group
        if len(data_below) == 0 or len(data_above) == 0:
            return classify_min_data(data),feature_importance
        else:
            question = str(split_feature) + " <= " + str(split_value)
            decision_subtree = {question: []}
            yes_answer,feature_importance = build_decision_tree(data_below, feature_importance, num_samples_total, current_depth, min_sample_size, max_depth, random_features, random_splits)
            no_answer,feature_importance = build_decision_tree(data_above, feature_importance, num_samples_total, current_depth, min_sample_size, max_depth, random_features, random_splits)
            if yes_answer == no_answer:
                decision_subtree = yes_answer
            else:
                decision_subtree[question].append(yes_answer)
                decision_subtree[question].append(no_answer)
            return decision_subtree, feature_importance

# classify_sample classify one test sample with the learned decision tree
def classify_sample(sample, decision_tree):
    if not isinstance(decision_tree, dict):
        return decision_tree
    question = list(decision_tree.keys())[0]
    feature, value = question.split(" <= ")
    if float(sample[feature]) <= float(value):
        answer = decision_tree[question][0]
    else:
        answer = decision_tree[question][1]
    return classify_sample(sample, answer)

# decision_tree_predictions returns the predictions using decision tree
def decision_tree_predictions(data, decision_tree):
    predictions = data.apply(classify_sample, axis = 1, args = (decision_tree,))
    return predictions

# ==================================Random Forest=================================
# bootstrap returns the random sub-samples
def bootstrap(data, bootstrap_size):
    if isinstance(bootstrap_size, float):
        bootstrap_size = round(bootstrap_size * data.shape[0])
    indices = data.index.tolist()
    random_samples = random.sample(population=indices, k=bootstrap_size)
    oob_samples = []
    for sample in indices:
        if sample not in random_samples:
            oob_samples.append(sample)
    return data.loc[random_samples], data.loc[oob_samples]

# build_random_forest build random forest model
def build_random_forest(data, feature_importance, bootstrap_size, random_features, random_splits, forest_size = 20, max_depth = 1000):
    random_forest = []
    oob_scores = []
    for i in range(forest_size):
        bootstrapped_data, oob_data = bootstrap(data, bootstrap_size)
        num_samples_total = len(bootstrapped_data)
        decision_tree, feature_importance = build_decision_tree(bootstrapped_data, feature_importance, num_samples_total, max_depth=max_depth, random_features=random_features, random_splits=random_splits)
        random_forest.append(decision_tree)

        oob_prediction = decision_tree_predictions(oob_data, decision_tree)
        oob_score = calculate_accuracy(oob_prediction, oob_data["Disease_state"].tolist())
        oob_scores.append(oob_score)
    for feature in feature_importance.keys():
        feature_importance[feature] = feature_importance[feature][0]/feature_importance[feature][1] 
    average_oob_score = np.mean(oob_scores)  
    return random_forest, feature_importance, average_oob_score

# random_forest_predictions returns the prediction using random forest
def random_forest_predictions(data, random_forest):
    predictions = {}
    for i in range(len(random_forest)):
        column = "decision tree " + str(i)
        predictions[column] = decision_tree_predictions(data, random_forest[i])
    predictions = pd.DataFrame(predictions)
    return predictions.mode(axis = 1)[0]

# calculate_accuracy calculates the accuracy given the predictions and true values
def calculate_accuracy(predictions, true):
    correct = predictions == true
    return correct.mean()

# ==================================Feature importance (method 2, permutaion)=================================
def convert_class_to_int(classes):
    classes_int = []
    for i in range(len(classes)):
        if classes[i] == 'T':
            classes_int.append(1)
        else:
            classes_int.append(0)
    return classes_int

def rmse(predictions, true):
    return np.sqrt(((np.array(predictions)-np.array(true))**2).mean())

def calculate_feature_importance(data, random_forest):
    feature_importance = {}
    true = convert_class_to_int(data.iloc[:,-1])
    base = rmse(convert_class_to_int(random_forest_predictions(data, random_forest)),true)
    for feature in data.columns[:-1]:
        data_copy = data.copy()
        data_copy[feature] = np.random.choice(data_copy[feature], len(data_copy))
        feature_importance[feature] = np.abs(rmse(convert_class_to_int(random_forest_predictions(data_copy, random_forest)), true)-base)
    return feature_importance


if __name__ == "__main__":
    total_data =  pd.read_csv('./data/processed_data/log2_Normalized_DEG_p0.001_FC2.csv', sep=',')

    total_data.index = total_data["geneName"]
    total_data = total_data.drop("geneName", axis=1)
    total_data = total_data.T
    label = total_data["Disease_state"].tolist()

    columns_indices = total_data.columns[1:].tolist()
    columns_indices.append("Disease_state")
    total_data = total_data[columns_indices]
    print(total_data.shape)
    # 808 features (columns) + 1 label (column)
    # 797 samples (rows)
    # total_data = total_data.iloc[:,-101:]

    data_train, data_test = split_train_test(total_data, 0.3)
    feature_importance = {}
    random.seed(123)

    forest_sizes = [1,10,20,50,100]
    oob_scores = []
    train_accuracies = []
    test_accuracies = []
    for forest_size in forest_sizes:
        feature_importance = {}
        random_forest, feature_importance, average_oob_score = build_random_forest(data_train, feature_importance, bootstrap_size=0.632, random_features=800, random_splits=800, forest_size=forest_size)
        oob_scores.append(average_oob_score)
        predictions_test = random_forest_predictions(data_test, random_forest).tolist()
        test_accuracies.append(calculate_accuracy(predictions_test, data_test["Disease_state"]))
        predictions_train = random_forest_predictions(data_train, random_forest).tolist()
        train_accuracies.append(calculate_accuracy(predictions_train, data_train["Disease_state"]))
    
    plt.figure(figsize=(8,8))
    plt.plot(forest_sizes, oob_scores, color = 'r', label='oob score')
    plt.plot(forest_sizes, train_accuracies, color = 'g', label='train accuracy')
    plt.plot(forest_sizes, test_accuracies, color = 'b', label='test accuracy')
    plt.legend()
    plt.xlabel("Number of trees")
    plt.savefig("accuracy_plot_new_new.png", dpi=500)
    plt.close()

    # when number of trees = 100:
    # random_forest, feature_importance, average_oob_score = build_random_forest(data_train, feature_importance, bootstrap_size=0.632, random_features=800, random_splits=800, forest_size=100)
    predictions_test = random_forest_predictions(data_test, random_forest).tolist()
    predictions_train = random_forest_predictions(data_train, random_forest).tolist()
    print("The train accyracy (number of trees = 100) is:", calculate_accuracy(predictions_train, data_train["Disease_state"]))
    print("The test accuracy (number of trees = 100) is:", calculate_accuracy(predictions_test, data_test["Disease_state"]))
    print("The oob score is (number of trees = 100):", average_oob_score)

    # plot confusion matrix
    true_and_predicted = pd.DataFrame(columns=['True', 'Predicted'])
    true_and_predicted['True'] = data_test["Disease_state"].tolist()
    true_and_predicted['Predicted'] = random_forest_predictions(data_test, random_forest).tolist()
    confusion_matrix = pd.crosstab(true_and_predicted['True'], true_and_predicted['Predicted'])
    print(confusion_matrix)
    plt.figure(figsize=(8,8))
    sns.heatmap(confusion_matrix, square=True, cmap="Reds")
    plt.title("confusion matrix")
    plt.savefig("confusion_matrix_100ntree.png", dpi=500)
    plt.close()

    # plot feature importance
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    marker_genes = []
    importances = []
    for i in range(50):
        marker_genes.append(feature_importance[i][0])
        importances.append(feature_importance[i][1])
    print(feature_importance[:50])
    print("The marker genes are (according to Gini importance):", marker_genes)

    plt.figure(figsize=(20,15))
    plt.bar(x=marker_genes, height = importances, color='firebrick')
    plt.xticks(rotation=60)
    plt.xlabel("features (genes)")
    plt.title("feature importance")
    plt.savefig("feature_importance_100ntree_50features.png", dpi=500)
    plt.close()
    
    # marker_genes2 = []
    # for feature in feature_importance2[:20]:
    #     marker_genes2.append(feature[0])
    # print(feature_importance2[:20])
    # print("The marker genes are (method2):", marker_genes2)



