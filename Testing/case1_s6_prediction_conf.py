# from s4_cell_conf import population_threshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import pickle
import sys
import numpy as np
import warnings
from sklearn.utils.class_weight import compute_class_weight
import random

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
with open('./Testing/result.pkl', 'rb') as f:
    result = pickle.load(f)
    
with open('./Training/df_aoi.pkl', 'rb') as f:
    df_aoi = pickle.load(f)

with open('./Training/df_not_aoi.pkl', 'rb') as f:
    df_not_aoi = pickle.load(f)
        
with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)
# print(df_aoi.columns)

to_drop = result[1]
prediction_accuracies = []
std_errors = []
predictions_till_now  = []
t1 = df_aoi.shape[0]
t2 = df_not_aoi.shape[0]

class PredictionCall():
    def __init__(self):
        self.models = {
            'Logistic Regression' : logreg
        }
        self.t1 = t1
        self.t2 = t2
        self.count = 0
        self.all_coordinates = [(x, y) for x in range(grid_size) for y in range(grid_size)]
 
    
    def train_test_newAOI(self, agents, md_agents):
        # md_agents = [agent for agent in agents if agent.agent_type_md.startswith('MD')]
        # print("md_agents len : ", len(md_agents))
        for agent in md_agents :
            # print("### : ", agent.agent_type)
            df = agent.agent_df
            if df.empty : 
                print("skipping ")
                continue
            value_counts = df.is_aoi.value_counts()
            if 1 not in value_counts.index:
                t1 = self.t1 + 0
                self.t1 = min(df_aoi.shape[0], t1)
            elif 0 not in value_counts.index:
                t2 = self.t2 + 0
                self.t2 = min(df_not_aoi.shape[0], t2)
            else:
                t1 = self.t1 + 0
                t2 = self.t2 + 0
            
            df_aoi_sample = df_aoi.sample(n=self.t1, random_state=42)
            df_not_aoi_sample = df_not_aoi.sample(n=self.t1,random_state=42)
            selected_columns = df.columns.intersection(df_aoi_sample.columns)
            df_aoi_sample_selected = df_aoi_sample[selected_columns]
            df_not_aoi_sample_selected = df_not_aoi_sample[selected_columns]
            df_training_data = pd.concat([df_aoi_sample_selected, df_not_aoi_sample_selected], ignore_index=True)
            df = pd.concat([df, df_training_data], ignore_index=True)

            # Feature Engineering
            df['one_hop_merged'] = df['one_hopAOICount'] + df['one_hopSCount']
            df['one_hop_merged_fs'] = df['one_hop_fraction'] + df['one_hopSCount']
            df['one_hop_merged_af'] = df['one_hopAOICount'] + df['one_hop_fraction']
            df['two_hop_merged'] = df['two_hopAOICount'] + df['two_hopSCount']
            df['two_hop_merged_fs'] = df['two_hop_fraction'] + df['two_hopSCount']
            df['two_hop_merged_af'] = df['two_hopAOICount'] + df['two_hop_fraction']
            df['three_hop_merged'] = df['three_hopAOICount'] + df['three_hopSCount']
            df['three_hop_merged_fs'] = df['three_hop_fraction'] + df['three_hopSCount']
            df['three_hop_merged_af'] = df['three_hopAOICount'] + df['three_hop_fraction']
            df['one_two_merged'] = df['one_hop_merged'] + df['two_hop_merged']
            df['two_three_merged'] = df['three_hop_merged'] + df['two_hop_merged']
            df['three_one_merged'] = df['three_hop_merged'] + df['one_hop_merged']
            
            model_weights = []
            y = df['is_aoi']
            X = df.drop(to_drop, axis=1)
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # print("****************************")
            # print("X_train shape : ", X_train.shape)
            # print("Y_train shape : ", Y_train.shape)
            # print("X_test shape : ", X_test.shape)
            # print("Y_test shape : ", Y_test.shape)
            i = 1 
            for model_name, model in self.models.items():
                # if agent.agent_model_weights:
                #     model.coef_ = agent.agent_model_weights[0]
                #     model.intercept_ = agent.agent_model_weights[1]
                # else:
                #     model.coef_ = logreg.coef_
                #     model.intercept_ = logreg.intercept_
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)
                results_mean = np.mean(y_pred)
                results_std = np.std(y_pred)
                accuracy = accuracy_score(Y_test, y_pred)
                precision = precision_score(Y_test, y_pred, zero_division=1)
                recall = recall_score(Y_test, y_pred, zero_division=1)
                f1 = f1_score(Y_test, y_pred, zero_division=1)
                conf_matrix = confusion_matrix(Y_test, y_pred)
                if conf_matrix.shape == (2, 2):
                    tn = conf_matrix[0, 0]
                    fp = conf_matrix[0, 1]
                    fn = conf_matrix[1, 0]
                    tp = conf_matrix[1, 1]
                else:
                    tn = conf_matrix[0, 0]
                    fp = 0
                    fn = 0
                    tp = 0
                # Print the evaluation metrics
                # print("Mean score:", results_mean)
                # print(f'{agent.idx} Accuracy is {accuracy}')
                # print("Precision:", precision)
                # print("Recall:", recall)
                # print("F1 Score:", f1)
                # print("Confusion Matrix:")
                # print(conf_matrix)
                # print("True negatives (TN):", tn)
                # print("False positives (FP):", fp)
                # print("False negatives (FN):", fn)
                # print("True positives (TP):", tp)
                i += 1
                coefficients = model.coef_  # Coefficients for each feature
                intercept = model.intercept_  # Intercept term
                model_weights.append([coefficients, intercept])
                agent.agent_model_weights = model_weights
                agent.agent_mse.append([results_mean, accuracy, precision, recall, f1, tn, fp, fn, tp])
        return 


    def get_all_neighbors_for_coordinates_with_size(self, traversed_path):
        traversed_path = [tup[1] for tup in traversed_path]
        untraversed_coordinates = [coord for coord in self.all_coordinates if coord not in traversed_path]
        return untraversed_coordinates

    
    def get_neighbors_for_coordinates_with_size(self, coords_list, n, path):
        neighbors = []
        hop_range = 10  # Define the hop range as 3 for 3-hop neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for i, center_point, surv, aoi, steps in coords_list:
            x, y = center_point
            for dx in range(-hop_range, hop_range + 1):
                for dy in range(-hop_range, hop_range + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the center point itself
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in path:
                        neighbors.append((nx, ny))
        return neighbors


    def get_untraversed_neighbors(self, agents, n):
        for agent in agents:
            agent.agent_untraversed_neighbors = self.get_all_neighbors_for_coordinates_with_size(agent.agent_traversed_path)
            # agent.agent_untraversed_neighbors = self.get_neighbors_for_coordinates_with_size(agent.agent_traversed_path, n, agent.agent_path)
            agent.agent_untraversed_neighbors = list(set(agent.agent_untraversed_neighbors))
            # print("agent.agent_untraversed_neighbors len = ", len(agent.agent_untraversed_neighbors))
        return 
    

    def predict_newAOI_logreg(self, md_agents, agents, predictions):
        num_agents = len(agents)
        merged_predicted_AOI = []
        all_true_labels = []
        all_predicted_labels = []
        traversed_locations = []
        for agent in agents:
            traversed_locations.extend(agent.agent_path)
        data = pd.DataFrame()
        for md_agent in md_agents:
            model = logreg
            
            if md_agent.agent_model_weights :
                # Set model weights
                model.coef_ = md_agent.agent_model_weights[0][0]
                model.intercept_ = md_agent.agent_model_weights[0][1]
            
            warnings.filterwarnings("ignore", category=UserWarning)

            # Prepare the data for prediction
            if md_agent.agent_untraversed_df.empty : 
                continue
            data = md_agent.agent_untraversed_df.copy()
            data.columns = ['location', 'latitude', 'longitude', 'elevation', 'one_hopAOICount', 'one_hopSCount', 'one_hop_fraction', 'two_hopAOICount', 'two_hopSCount', 'two_hop_fraction', 'three_hopAOICount', 'three_hopSCount', 'three_hop_fraction', 'probability', 'population', 'is_aoi']
            data['one_hop_merged'] = data['one_hopAOICount'] + data['one_hopSCount']
            data['one_hop_merged_fs'] = data['one_hop_fraction'] + data['one_hopSCount']
            data['one_hop_merged_af'] = data['one_hopAOICount'] + data['one_hop_fraction']
            data['two_hop_merged'] = data['two_hopAOICount'] + data['two_hopSCount']
            data['two_hop_merged_fs'] = data['two_hop_fraction'] + data['two_hopSCount']
            data['two_hop_merged_af'] = data['two_hopAOICount'] + data['two_hop_fraction']
            data['three_hop_merged'] = data['three_hopAOICount'] + data['three_hopSCount']
            data['three_hop_merged_fs'] = data['three_hop_fraction'] + data['three_hopSCount']
            data['three_hop_merged_af'] = data['three_hopAOICount'] + data['three_hop_fraction']
            data['one_two_merged'] = data['one_hop_merged'] + data['two_hop_merged']
            data['two_three_merged'] = data['three_hop_merged'] + data['two_hop_merged']
            data['three_one_merged'] = data['three_hop_merged'] + data['one_hop_merged']
            locations = data['location']
            true_labels = data['is_aoi']
            filtered_data = data[true_labels == 1]
            data.drop(columns=to_drop, inplace=True)
            
            predicted_values = model.predict(data.values)
            probability_scores = model.predict_proba(data.values)

            acc = accuracy_score(true_labels, predicted_values)
            md_agent.prediction_accuracies.append(acc)

            all_true_labels.extend(true_labels)
            all_predicted_labels.extend(predicted_values)

            for loc, value_array, prob_scores in zip(locations, predicted_values, probability_scores):
                predictions[loc]['prob_value'] = value_array
                predictions[loc]['prob_score'] = prob_scores[1]
                if value_array == 1 :
                    if predictions[loc]['visit_status'] == 0 and loc not in traversed_locations:
                        md_agent.predicted_AOI.append(loc)
                        predictions[loc]['prediction_status'] = 1
                else:
                    continue
            md_agent.predicted_AOI = self.remove_duplicates(traversed_locations, md_agent.predicted_AOI)
            unique_predicted_AOI = set(filter(lambda x: x != (), md_agent.predicted_AOI))
            merged_predicted_AOI.extend(unique_predicted_AOI)
            merged_predicted_AOI = list(set(merged_predicted_AOI))
            
        # if len(merged_predicted_AOI) > 50:
        #     num_entries = max(1, len(merged_predicted_AOI) * 50 // 100)  # Ensure at least 1 entry
        #     merged_predicted_AOI = random.sample(merged_predicted_AOI, num_entries)
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        self.count += 1
        prediction_accuracies.append(accuracy)
        # Calculate standard error of the mean accuracy
        if len(prediction_accuracies) > 1:
            std_error = np.std(prediction_accuracies) / np.sqrt(len(prediction_accuracies))
            std_errors.append(std_error)
        merged_predicted_AOI = list(set(filter(lambda x: x != (), merged_predicted_AOI)))
        return merged_predicted_AOI, predictions, prediction_accuracies, std_errors

    def remove_duplicates(self, A, B):
        set_A = set(A)
        B_filtered = [item for item in B if item not in set_A]
        return B_filtered
 