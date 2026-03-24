# Author : Eoin McLoughlin
import os
import json
import shutil
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from itertools import cycle
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sns.set(style='darkgrid')
colour_list = ["blue", "magenta", "red", "green", "gray", "lime", "maroon", "navy", "olive", "purple", "silver", "fuchsia", "teal", "yellow", "aqua", "black"]


class GraphPlotter:
    ''' Class ussd for creating visualisations 
    '''
    def __init__(self, file_name):
        self.file_name = file_name


    def plot_len_vs_acc(self, lengths, y_true, y_pred, bins=5):
        ''' Graphs to see how different lengths of actions effect the models accuracy
        Args:
            lengths :
            y_true (list) : List of true labels 
            y_pred (list) : List of predicted labels 
            bins (Int) : Number of bins to group results into
        '''
        lengths = np.array(lengths)
        correct = (np.array(y_true) == np.array(y_pred))
        # print(lengths)
        # print(correct)
        # Bin by quantiles and compute mean accuracy per bin
        df = pd.DataFrame({"len": lengths, "correct": correct})
        df["len_bin"] = pd.qcut(df["len"], q=bins, duplicates="drop")
        acc = df.groupby("len_bin")["correct"].mean()
        # Plot
        acc.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.ylabel("Accuracy")
        plt.xlabel("Sequence length range")
        plt.title("Accuracy vs. Sequence Length")
        plt.tight_layout()
        plt.show()



    def plot_reliability(self, y_true, logits):
        ''' 
        '''
        p = torch.softmax(torch.tensor(logits), dim=1).numpy()
        conf = p.max(axis=1); pred = p.argmax(axis=1)
        correct = (pred == np.array(y_true))
        prob_true, prob_pred = calibration_curve(correct.astype(int), conf, n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'--')
        plt.xlabel("Predicted confidence")
        plt.ylabel("Empirical accuracy")
        plt.title("Reliability"); 
        plt.grid(True)
        plt.show()
    


    def plot_BIC_hist(self, results_dict:dict):
        ''' Plots the Bayesian Information criterion for each all most important features
        Args:
            results_dict (Dict) : Dict with BIC results for all GMM's
        '''
        swing_types = list(results_dict.keys())
        bics = [res['bic'] for res in results_dict.values()]
        ks = [res['k'] for res in results_dict.values()]
        plt.figure(figsize=(7, 4))
        bars = plt.bar(swing_types, bics, color='skyblue', edgecolor='black')
        plt.xlabel("Swing Type")
        plt.ylabel("Best BIC (Lower == Better)")
        plt.title("Best BIC per Action Class")
        # Annotate bars with k values
        for bar, k in zip(bars, ks):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"k={k}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        plt.close()


    def plot_report_per_action_type(self, y_test:list, pred_labels:list):
        ''' Plot barchart of each actions results 
        Args:
            y_test (List) : List of true labels
            pred_labels (List) : List of predicted labels
        '''
        # Classification report per action type
        report = classification_report(y_test, pred_labels, output_dict=True)
        pd.DataFrame(report).T[['precision', 'recall', 'f1-score']].plot.bar(figsize=(8,4), title="Per Action Metrics", grid=True)
        plt.tight_layout()
        plt.show()
        plt.close()


    def plot_params_vs_BIC(self, results_dict:dict):
        ''' Plot the numebr of params vs the number
        ArgsL
            results_dict (Dict) : Dict of  features for eahc mixture
        '''
        # Plot mixture model params vs BIC
        plt.figure(figsize=(6,4))
        for cls, res in results_dict.items():
            plt.plot(res['k_list'], res['bic_list'], label=cls, marker='o')
        plt.title("BIC vs Number of Components per Class")
        plt.xlabel("Number of Components (k)")
        plt.xticks(rotation=45)
        plt.ylabel("BIC (lower = better)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


    def plot_confusion_matrix(self, y_pred:list, y_true:list, title:str):
        ''' Plot the confusion matrix ofa set of results
        Args:
            y_test (List) : List of true labels
            pred_labels (List) : List of predicted labels
            title (String) : Title for the plot
        '''
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # in case a row sums to 0

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_true))
        # raw_conf = confusion_matrix(y_test, pred_labels)
        # norm_conf = raw_conf.astype(float) / raw_conf.sum(axis=1, keepdims=True)
        disp.plot(cmap="viridis", colorbar=True, values_format=".2f")
        plt.grid(False)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()


    def plot_PCA_embeddings(self, y_test:list, X_test_pca:list, pca_df:pd.DataFrame):
        ''' Plots a 2d scatter graph showing how principal components groups samples 
        Args:
            y_test (List) : List of true labels
            X_test_pca (List) : 
            pca_df (DataFrame) : DF containing principal components
        '''
        # Plot PCA embeddings
        plt.figure(figsize=(7,5))
        for lbl in np.unique(y_test):
            mask = np.array(y_test) == lbl
            plt.scatter(X_test_pca[mask,0], X_test_pca[mask,1], label=lbl, alpha=0.7)
        plt.title("2D PCA Embedding Colored by True Class")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        pca_df['Feature_Num'] = [f'Feature_{int(i)}' for i in range(0, len(pca_df))]
        print(pca_df)
        # # df_imp = pd.DataFrame(pca_df, columns=['Feature', 'MeanAbsLoading'])
        pca_df.head(20).plot.bar(x='Feature_Num', y='mean_abs_loading', color='teal', figsize=(8,4), title='Most Influential Pose Features')
        # plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        plt.close()


    def plot_kp_over_time(self, data, uid):
        ''' Used to plot a dynamic number of signal on top of one another
        Args:
            data (Dict) : A dictionary where each key is a string with the name of the signal & value is the signal in a list
            uid (String) : A string containing the users ID for reference
        '''
        colors = cycle(colour_list)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle(uid, fontsize=16)

        for key, val in data.items():
            ax.plot(val, label= key, color=next(colors))

        ax.set_xlabel("Frame Index", fontsize=15)
        ax.set_ylabel("Signal vals", fontsize=18)
        ax.legend(loc="best")
        ax.margins(0.1)
        fig.tight_layout()
        plt.show()



    def plot_side_by_side(self, img1_path, img2_path, title1, title2):
        ''' Used to plot the an original and plotted frame
        Args:
            img1_path (String) : Path to the first image 
            img2_path (String) : Path to the second image
            title1 (String) : Title for the first image 
            title2 (String) : Title for the second image
        '''
        figsize=(10, 5)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        # Plot first image
        axes[0].imshow(img1)
        axes[0].set_title(title1, fontsize=13, pad=10)
        axes[0].axis("off")

        # Plot second image
        axes[1].imshow(img2)
        axes[1].set_title(title2, fontsize=13, pad=10)
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()



    
    def plot_multiple_signals(self, nested_data):
        ''' Function for plotting multiple signals in seperate plots to examine more signals at the same time 
            NESTED_DATA_STRUCTURE = [ ( {key : [list of values], key2 : [ list2 ]}, name_of_graph_1 ), ( {key : [list of values], key2 : [ list2 ]}, name_of_graph_2 )]
        '''
        ## Plot the initial arm angles
        f,ax = plt.subplots(len(nested_data),1,figsize=(20,5),sharex=True)
        for index, (data, data_title) in enumerate(nested_data):
            ax[index].set(title = data_title.title()) #, ylabel='Degrees')
            colors = cycle(colour_list)
            # Plot each signal in data
            print()
            print("-"*50)
            print(f"Graph {index}:")
            for kp_name, signal_vals in data.items():
                colour=next(colors)
                print(kp_name, ":", colour)
                ax[index].plot(signal_vals, color=colour, label=kp_name)

        plt.tight_layout()
        plt.show()



    # def plot_3d_frame(df, aspect_ratio):
    def plot_3d_frame(df, height, width):
        '''  Main function for plotting in 3d
        Links : 
            How to plot in 3D : https://pythonnumericalmethods.berkeley.edu/notebooks/chapter12.02-3D-Plotting.html
            How to set custom aspect ratio : https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
            Same problem try this tomorrow : https://copyprogramming.com/howto/python-matplotlib-set-plot-aspect-ratio-code-example

        Notes:
            Need to scale the 3D graph to the screen ratio
        '''
        # print("ASPECT RATIO :", aspect_ratio)
        fig = plt.figure(figsize = (8,8))
        ax = plt.axes(projection='3d')
        # ax.set_box_aspect((np.ptp(1), np.ptp(aspect_ratio), np.ptp(1)))  # aspect ratio is 1:1:1 in data space
        ax.grid()

        row = df.iloc[0]
        for (vec_start, vec_end) in vector_connections:
            ax.plot([float(row[f'{vec_start}_X']), float(row[f'{vec_end}_X'])], [float(row[f'{vec_start}_Z']), float(row[f'{vec_end}_Z'])],zs=[float(row[f'{vec_start}_Y']), float(row[f'{vec_end}_Y'])], color="blue")

        ax.set_title('3D Pose Plot')
        # Set axes label
        ax.set_xlabel('X-axis', labelpad=40)
        ax.set_ylabel('Z-axis', labelpad=40)
        ax.set_zlabel('Y-axis', labelpad=40)
        plt.show()



    # def plot_3d_frame(df, aspect_ratio):
    def plot_2d_frame(df):
        '''  Main function for plotting in 3d
        Links : 
            How to plot in 3D : https://pythonnumericalmethods.berkeley.edu/notebooks/chapter12.02-3D-Plotting.html
            How to set custom aspect ratio : https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
            Same problem try this tomorrow : https://copyprogramming.com/howto/python-matplotlib-set-plot-aspect-ratio-code-example

        Notes:
            Need to scale the 3D graph to the screen ratio
        '''
        row = df.iloc[0]
        for (vec_start, vec_end) in vector_connections:
            plt.plot([float(row[f'{vec_start}_X']) * -1, float(row[f'{vec_end}_X']) * -1], [float(row[f'{vec_start}_Y']), float(row[f'{vec_end}_Y'])], color="blue")

        plt.show()
      


    def graph_point_phase_signals(df, side1, kp1, axis1, side2, bp2, axis2):
        ''' Used to display the relationship between two signals, Plots 3 figures:
                1) The original signals plotted ontop of one another 
                2) The phase angles of both signals plotted ontop of one another 
                3) The phase synchrony between the two hase angle time series
        Args:
            df (DataFrame) : A DataFrame containing all of the angle info needed for calculating the phase portrait
            side1 (String) : A string referencing which side of the body the keypoint lies on (LEFT / RIGHT)
            kp1 (String) : A String denoting the first keypoint we will plotting
            side2 (String) : A string referencing which side of the body the keypoint lies on (LEFT / RIGHT)
            bp2 (String) : A String denoting the second keypoint we will plotting
        '''
        # Extract all neccessary columns for plotting

        # Apply butterworth filter to remove noise
        df[f'BUTTERWORTH_{side1}_{kp1}_{axis1}'] = butterworth_filter(list(df[f'{side1}_{kp1}_{axis1}']), 2, 1, 'low', 5)
        df[f'BUTTERWORTH_{side2}_{bp2}_{axis2}'] = butterworth_filter(list(df[f'{side2}_{bp2}_{axis2}']), 2, 1, 'low', 5)

        r_arm_angles = df[f'BUTTERWORTH_{side1}_{kp1}_{axis1}']
        l_arm_angles = df[f'BUTTERWORTH_{side2}_{bp2}_{axis2}']
        r_phase = df[f'{side1}_{kp1}_{axis1}_PHASE_ANGLES_NORM_0_1']
        l_phase = df[f'{side2}_{bp2}_{axis2}_PHASE_ANGLES_NORM_0_1']
        phase_synchrony = df[f'{side1}_{kp1}_{axis1}_{side2}_{bp2}_{axis2}_CRP']

        # convert radians to degrees
        r_phase = np.rad2deg(r_phase)
        l_phase = np.rad2deg(l_phase)

        # Filtering the signals for cleaner graphs
        r_phase_butter = butterworth_filter(list(r_phase), 5, 1, 'low', 5)
        l_phase_butter = butterworth_filter(list(l_phase), 5, 1, 'low', 5)
        phase_synchrony_butter = butterworth_filter(list(phase_synchrony), 5, 1, 'low', 5)

        # Plot the initial arm angles
        f,ax = plt.subplots(3,1,figsize=(20,5),sharex=True)
        ax[0].set(title='Normalised Angles at each Timepoint', ylabel='Degrees')
        ax[0].plot(r_arm_angles,color='r', label=f'{side1}_{kp1}')
        ax[0].plot(l_arm_angles,color='b', label=f'{side2}_{bp2}')
        ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)

        # Plot the phase angles
        ax[1].set(title='Phase Angles at each Timepoint', ylabel='Degrees')
        ax[1].plot(r_phase_butter,color='r', label=f'{side1}_{kp1}')
        ax[1].plot(l_phase_butter,color='b', label=f'{side2}_{bp2}')

        # plot the phase synchrony of the two phase angles over time
        ax[2].set(title='Instantaneous Phase Synchrony',xlabel='Time',ylabel='Degrees')
        ax[2].plot(phase_synchrony_butter)
        plt.tight_layout()
        plt.show()



    def graph_phase_signals(df, side1, kp1, side2, bp2):
        ''' Used to display the relationship between two signals, Plots 3 figures:
                1) The original signals plotted ontop of one another 
                2) The phase angles of both signals plotted ontop of one another 
                3) The phase synchrony between the two hase angle time series
        Args:
            df (DataFrame) : A DataFrame containing all of the angle info needed for calculating the phase portrait
            side1 (String) : A string referencing which side of the body the keypoint lies on (LEFT / RIGHT)
            kp1 (String) : A String denoting the first keypoint we will plotting
            side2 (String) : A string referencing which side of the body the keypoint lies on (LEFT / RIGHT)
            bp2 (String) : A String denoting the second keypoint we will plotting
        '''
        # Extract all neccessary columns for plotting

        # Apply butterworth filter to remove noise
        df[f'BUTTERWORTH_{side1}_{kp1}_ANGLES'] = butterworth_filter(list(df['{}_{}_ANGLES'.format(side1, kp1)]), 2, 1, 'low', 5)
        df[f'BUTTERWORTH_{side2}_{bp2}_ANGLES'] = butterworth_filter(list(df['{}_{}_ANGLES'.format(side2, bp2)]), 2, 1, 'low', 5)

        r_arm_angles = df[f'BUTTERWORTH_{side1}_{kp1}_ANGLES']
        l_arm_angles = df[f'BUTTERWORTH_{side2}_{bp2}_ANGLES']
        r_phase = df['{}_{}_PHASE_ANGLES_NORM_0_1'.format(side1, kp1)]
        l_phase = df['{}_{}_PHASE_ANGLES_NORM_0_1'.format(side2, bp2)]
        phase_synchrony = df['{}_{}_{}_{}_CRP'.format(side1, kp1, side2, bp2)]

        # convert radians to degrees
        r_phase = np.rad2deg(r_phase)
        l_phase = np.rad2deg(l_phase)

        # Filtering the signals for cleaner graphs
        r_phase_butter = butterworth_filter(list(r_phase), 5, 1, 'low', 5)
        l_phase_butter = butterworth_filter(list(l_phase), 5, 1, 'low', 5)
        phase_synchrony_butter = butterworth_filter(list(phase_synchrony), 5, 1, 'low', 5)

        # Plot the initial arm angles
        f,ax = plt.subplots(3,1,figsize=(20,5),sharex=True)
        ax[0].set(title='Normalised Angles at each Timepoint', ylabel='Degrees')
        ax[0].plot(r_arm_angles,color='r', label=f'{side1}_{kp1}')
        ax[0].plot(l_arm_angles,color='b', label=f'{side2}_{bp2}')
        ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),ncol=2)

        # Plot the phase angles
        ax[1].set(title='Phase Angles at each Timepoint', ylabel='Degrees')
        ax[1].plot(r_phase_butter,color='r', label=f'{side1}_{kp1}')
        ax[1].plot(l_phase_butter,color='b', label=f'{side2}_{bp2}')

        # plot the phase synchrony of the two phase angles over time
        ax[2].set(title='Instantaneous Phase Synchrony',xlabel='Time',ylabel='Degrees')
        ax[2].plot(phase_synchrony_butter)
        plt.tight_layout()
        plt.show()
        plt.close()