import csv
import math
import numpy as np
import pandas as pd
import shap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
class SpectrumInterpreter:
    def __init__(self, use_defaults=False):
        self.use_d = use_defaults
        self.model = None
    def fit(self, model):
        self.model = model
        df, self.X, self.y = self.read_in()
        X_train, X_test, y_train, y_test, X_val, y_val = self.split_data()
        if self.model == 'KNN':
            self.fit_KNN(X_train, y_train, X_test, y_test)
            self.model_sp =self.neigh
        if self.model == 'LogReg':
            self.fit_LogReg(X_train, y_train, X_test, y_test)
            self.model_sp = self.multi_label
        if self.model == 'Random Forest':
            self.fit_RandomForest(X_train, y_train, X_test, y_test)
            self.model_sp = self.clf

        if self.model == 'RBF SVC':
            self.fit_SVC(X_train, y_train, X_test, y_test, type='rbf')
            self.model_sp = self.SVC
        if self.model == 'Linear SVC':
            self.fit_SVC(X_train, y_train, X_test, y_test, type='linear')
            self.model_sp = self.SVC
        if self.model == 'Decision Tree':
            self.fit_decision_tree(X_train, y_train, X_test, y_test)
            self.model_sp = self.Decision_Tree
        self.validate(X_val, y_val, self.model_sp, self.model)
    def read_in(self):
        csv_file = pd.read_csv('Full_File.csv', index_col=0)
        df = pd.DataFrame(csv_file)
        X = df.drop(['CH', 'C=C', 'C≡C', 'C≡N', 'C-OH','C=O', 'C=OOH', 'N-H', 'C≡C-H'], axis=1)
        y = df[['CH','C=C','C≡C', 'C≡N', 'C-OH','C=O', 'C=OOH', 'N-H', 'C≡C-H']]
        return df, X, y
    def split_data(self):
        mskf = MultilabelStratifiedKFold(n_splits=7, shuffle=True, random_state=42)
        for train_idx, test_idx in mskf.split(self.X, self.y):
            X_temp, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_temp, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            break
        for t_idx, val_idx in mskf.split(X_temp, y_temp):
            X_train, X_val = X_temp.iloc[t_idx], X_temp.iloc[val_idx]
            y_train, y_val = y_temp.iloc[t_idx], y_temp.iloc[val_idx]
            print(X_val)
            break
        return X_train, X_test, y_train, y_test, X_val, y_val
    def fit_KNN(self, X_train, Y_train, X_test, Y_test):
        test_accuracy=[]
        train_accuracy = []
        kl=[]
        for k in range(1,60):
            self.neigh = KNeighborsClassifier(n_neighbors=k)
            self.neigh.fit(X_train, Y_train)
            y_train_pred = self.neigh.predict(X_train)
            y_test_pred = self.neigh.predict(X_test)
            train_accuracy.append(f1_score(Y_train, y_train_pred, average='micro'))
            test_accuracy.append(f1_score(Y_test, y_test_pred, average='micro'))
            kl.append(k)
        plt.plot(kl, test_accuracy, label='Test F1')
        plt.plot(kl, train_accuracy, label='Train F1')
        plt.xlabel('K Neighbors')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()
        best_k = kl[test_accuracy.index(max(test_accuracy))]
        print('Best K :', best_k)
        self.neigh = KNeighborsClassifier(n_neighbors=best_k)
        self.neigh.fit(X_train, Y_train)
        y_train_pred = self.neigh.predict(X_train)
        y_test_pred = self.neigh.predict(X_test)
        self.metrics(y_test_pred, Y_test, 'KNN')  
        

    def fit_LogReg(self, X_train, y_train, X_test, y_test):
        reg = LogisticRegression(solver='liblinear', C=1, class_weight='balanced', max_iter=1000, penalty='l2')
        self.multi_label = OneVsRestClassifier(reg)
        self.multi_label.fit(X_train, y_train)
        y_test_pred = self.multi_label.predict(X_test)
        self.metrics(y_test_pred, y_test, 'Log Reg')  
    def fit_RandomForest(self, X_train, y_train, X_test, y_test):
        n_estimators = [2,10,50,100,200,300,400,500,600,700,800,900,1000]
        scores = []
        for n in n_estimators:
            self.clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=n, class_weight='balanced', random_state=42))
            self.clf.fit(X_train, y_train)
            y_test_pred = self.clf.predict(X_test)
            score = f1_score(y_test, y_test_pred, average='macro')
            scores.append(score)
        plt.plot(n_estimators, scores, label='Test F1')
        plt.xlabel('N Estimators')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()
        best_score = max(scores)
        best_n = n_estimators[scores.index(best_score)]
        self.clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=best_n, class_weight='balanced', random_state=42))
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        self.metrics(y_pred, y_test, 'Random Forest')  
    def fit_SVC(self, X_train, y_train, X_test, y_test, type='RBF'):
        if type == 'rbf':
            c_grid = [0.01,0.1,0.5,1,10,100,1000]
            gamma_grid = [0.01,0.1,0.5,1,10,100,1000]
        if type == 'linear':
            c_grid = [0.01, 0.1, 0.5, 1, 10]
            gamma_grid = [0.01, 0.1, 0.5, 1, 10]
        preds_micro = []
        cl = []
        gl = []
        for c in c_grid:
            pred_c_micro = []
            pred_c_macro = []
            for gamma in gamma_grid:
                cl.append(c)
                gl.append(gamma)
                self.SVC = OneVsRestClassifier(SVC(C=c, gamma=gamma, kernel=type, class_weight='balanced'))
                self.SVC.fit(X_train, y_train)
                y_pred = self.SVC.predict(X_test)
                pred_c_macro.append(f1_score(y_test, y_pred, average='macro'))
                pred_c_micro.append(f1_score(y_test, y_pred, average='micro'))
            preds_micro.append(pred_c_micro)
        plt.scatter(cl, gl, label='Test F1', cmap='plasma', c=preds_micro, linewidths=20,alpha=0.75)
        print('Done')
        plt.xlabel('C')
        plt.ylabel('Gamma')
        if type == 'rbf':
            plt.xscale('log')
            plt.yscale('log')
        plt.legend()
        plt.show()
        preds_micro = np.array(preds_micro)
        best_score = np.max(preds_micro)
        row = np.where(preds_micro==best_score)[0][0]
        col = np.where(preds_micro==best_score)[1][0]
        best_c = c_grid[row]
        best_gamma = gamma_grid[col]
        self.SVC = OneVsRestClassifier(SVC(kernel=type, C=best_c, gamma=best_gamma))
        self.SVC.fit(X_train, y_train)
        y_pred = self.SVC.predict(X_test)
        self.metrics(y_pred, y_test, 'SVC ' + type)  
    def fit_decision_tree(self, X_train, y_train, X_test, y_test):
        scores = []
        dl = []
        for d in range(1,1000, 10):
            self.Decision_Tree = OneVsRestClassifier(DecisionTreeClassifier(max_depth=d))
            self.Decision_Tree.fit(X_train, y_train)
            y_pred = self.Decision_Tree.predict(X_test)
            scores.append(f1_score(y_test, y_pred, average='micro'))
            dl.append(d)
        plt.plot(dl, scores, label='Test F1')
        plt.xlabel('Depth')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()
        best_score = max(scores)
        best_depth = scores.index(best_score)+1
        self.Decision_Tree = OneVsRestClassifier(DecisionTreeClassifier(max_depth=best_depth))
        self.Decision_Tree.fit(X_train, y_train)
        y_pred = self.Decision_Tree.predict(X_test)
        self.metrics(y_pred, y_test, 'Decision Tree')  

    def metrics(self, y_pred, y_test, name):
        print('Macro F1', name, ' :', f1_score(y_test, y_pred, average='macro'))
        print('Micro F1 ', name, ': ', f1_score(y_test, y_pred, average='micro'))
        print('Accuracy Score ', name,':', accuracy_score(y_test, y_pred))       
        print(classification_report(y_test, y_pred, target_names=['CH', 'C=C', 'C≡C', 'C≡N', 'C-OH', 'C=O', 'C=OOH', 'N-H', 'C≡C-H']))

    def predict(self, name, plot=True, model=None):
        if model == None:
            model = self.model
        print('\nPredicting :', name, 'IR Spectra with ', self.model, '\n')
        if model == 'KNN':
            self.predict_KNN(name)
        if model == 'Random Forest':
            self.predict_RF(name, plot)
        if model == 'LogReg':
            self.predict_LogReg(name, plot)
        if 'SVC' in model:
            self.predict_SVC(name, plot, type=self.model)
    def predict_KNN(self, name):
        functional = ['CH','C=C','C≡C', 'C≡N', 'C-OH','C=O', 'C=OOH', 'N-H']
        df_hex = pd.DataFrame(self.X.loc[name]).T
        knn_pred = self.neigh.predict(df_hex)
        knn_l = []
        correct = self.y.loc[name].T
        cr_l = []
        for funct, knn, cr in zip(functional, knn_pred[0], correct):
            if knn == 1:
                knn_l.append(funct)
            if cr == 1:
                cr_l.append(funct)
        print('Correct Fx Group:', cr_l)
        print('KNN Fx Group:', knn_l)
    def predict_SVC(self, name, plot, type):
        wave_numbers = self.X.columns.astype(float)
        funct_d = {'CH':0,'C=C':1,'C≡C':2, 'C≡N':3, 'C-OH':4,'C=O':5, 'C=OOH':6, 'N-H':7, 'C≡C-H':8}
        functionals = ['CH','C=C','C≡C', 'C≡N', 'C-OH','C=O', 'C=OOH', 'N-H', 'C≡C-H']
        df_hex = pd.DataFrame(self.X.loc[name]).T
        svc_l_pred = self.SVC.predict(df_hex)
        svc_l = []
        correct = self.y.loc[name].T
        cr_l = []
        for funct, svc, cr in zip(functionals, svc_l_pred[0], correct):
            if svc == 1:
                svc_l.append(funct)
            if cr == 1:
                cr_l.append(funct)
        print(type+' Fx Group:', svc_l)
        print('Correct Fx Group:', cr_l, '\n')
        if plot ==True and type=='Linear SVC':
            fig, ax = plt.subplots()
            ax.plot(wave_numbers, df_hex.values[0], label='IR Spectrum', alpha=1, marker='.')
            estimators_svc = self.SVC.estimators_
            for k in svc_l:
                exp = estimators_svc[funct_d[k]].coef_[0]
                wave_numbers_shap = list(wave_numbers)
                of_interest = list(exp*df_hex.values[0])
                for i, j in zip(df_hex.values[0], exp):
                    if abs(i*j) < 0.01:
                        del wave_numbers_shap[of_interest.index(i*j)]
                        of_interest.remove(i*j)
                ax.scatter(wave_numbers_shap, of_interest, label=k, alpha=0.5, marker='x', s=100, zorder=2)
                ax.vlines(x=wave_numbers_shap, ymax=of_interest, ymin=0, linestyles='dotted', alpha=0.5, zorder=1)
            plt.legend()
            plt.show()
    def predict_RF(self, name, plot=True):
        wave_numbers = self.X.columns.astype(float)
        funct_d = {'CH':0,'C=C':1,'C≡C':2, 'C≡N':3, 'C-OH':4,'C=O':5, 'C=OOH':6, 'N-H':7, 'C≡C-H':8}
        functionals = ['CH','C=C','C≡C', 'C≡N', 'C-OH','C=O', 'C=OOH', 'N-H', 'C≡C-H']
        df_hex = pd.DataFrame(self.X.loc[name]).T
        random_forest_pred = self.clf.predict(df_hex)
        rf_l = []
        correct = self.y.loc[name].T
        cr_l = []
        for funct, rf, cr in zip(functionals, random_forest_pred[0], correct):
            if rf == 1:
                rf_l.append(funct)
            if cr == 1:
                cr_l.append(funct)
        print('Random Forest Fx Group:', rf_l)
        print('Correct Fx Group:', cr_l, '\n')
        if plot:
            fig, ax = plt.subplots()
            ax.plot(wave_numbers, df_hex.values[0], label='IR Spectrum', alpha=1, marker='.')
            for k in rf_l:
                explainer = shap.TreeExplainer(self.clf.estimators_[funct_d[k]])
                shap_values = explainer.shap_values(df_hex)*5
                wave_numbers_shap = list(wave_numbers)
                of_interest = list(shap_values[0,:,1])
                for i in shap_values[0,:,1]:
                    if abs(i) < 0.05:
                        del wave_numbers_shap[of_interest.index(i)]
                        of_interest.remove(i)
                ax.scatter(wave_numbers_shap, of_interest, label=k, alpha=0.5, marker='x', s=100, zorder=2)
                ax.vlines(x=wave_numbers_shap, ymax=of_interest, ymin=0, linestyles='dotted', alpha=0.5, zorder=1)
            plt.legend()
            plt.show()
    def predict_LogReg(self, name, plot=True):
        wave_numbers = self.X.columns.astype(float)
        funct_d = {'CH':0,'C=C':1,'C≡C':2, 'C≡N':3, 'C-OH':4,'C=O':5, 'C=OOH':6, 'N-H':7, 'C≡C-H':8}
        functionals = ['CH','C=C','C≡C', 'C≡N', 'C-OH','C=O', 'C=OOH', 'N-H', 'C≡C-H']
        df_hex = pd.DataFrame(self.X.loc[name]).T
        log_reg_pred = self.multi_label.predict(df_hex)
        lr_l = []
        correct = self.y.loc[name].T
        cr_l = []
        estimators_ml = self.multi_label.estimators_
        for funct, lr, cr in zip(functionals, log_reg_pred[0], correct):
            if lr == 1:
                lr_l.append(funct)
            if cr == 1:
                cr_l.append(funct)
        print('Log Reg Fx Group:', lr_l)
        print('Correct Fx Group:', cr_l, '\n')
        if plot:
            fig, ax = plt.subplots()
            ax.plot(wave_numbers, df_hex.values[0], label='IR Spectrum', alpha=1, marker='.')
            for k in lr_l:
                exp = estimators_ml[funct_d[k]].coef_[0]
                wave_numbers_shap = list(wave_numbers)
                of_interest = list(exp*df_hex.values[0])
                for i, j in zip(df_hex.values[0], exp):
                    if abs(i*j) < 0.01:
                        del wave_numbers_shap[of_interest.index(i*j)]
                        of_interest.remove(i*j)

                ax.scatter(wave_numbers_shap, of_interest, label=k, alpha=0.5, marker='x', s=100, zorder=2)
                ax.vlines(x=wave_numbers_shap, ymax=of_interest, ymin=0, linestyles='dotted', alpha=0.5, zorder=1)
            plt.legend()
            plt.show()
    
    def validate(self, X_val, y_val, model, name):
        y_pred = model.predict(X_val)
        print('Validation Macro F1', name, ' :', f1_score(y_val, y_pred, average='macro'))
        print('Validation Micro F1 ', name, ': ', f1_score(y_val, y_pred, average='micro'))
        print('Validation Accuracy Score ', name,':', accuracy_score(y_val, y_pred))       
        print(classification_report(y_val, y_pred, target_names=['CH', 'C=C', 'C≡C', 'C≡N', 'C-OH', 'C=O', 'C=OOH', 'N-H', 'C≡C-H']))
k = SpectrumInterpreter()
k.fit('Random Forest')
k.predict('propanamide')