# coding=utf-8
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,\
    classification_report,precision_recall_curve,average_precision_score,roc_auc_score,roc_curve
from metrics.pickle_module import Pickle_module
from metrics.confusion_matrix import plot_confusion_matrix

class MethodValidation:

    def __init__(self):
        self.__indice = 0
        self.__matches_dict = {}

    def confusion_matrix(self,trueClass,predictedClass):
        '''
        :param trueClass:
        :param predictedClass:
        :return:
        '''
        plot_confusion_matrix(trueClass, predictedClass, classes=np.array(["Saida", "Entrada"]),
                              title='Matriz de confusão de pessoas de entrada/saida')
        plt.show()

    def score_alternative_matches(self,trueClass,result):
        result = self.match_to_array(result)
        #print(result)
        indice=0
        length = 0
        for keys,values in result.items():
            if len(values)>1:
                length+=1
                if trueClass.get(values[0]) == trueClass.get(values[1]):
                    indice +=1


        return str("Taxa de acerto na correspondência de pessoas: ") + str(float(indice / length) * 100) + str("%")

    def score_matches(self,trueClass,predictedClass):
        '''

        :param trueClass:
        :param predictedClass:
        :return:
        '''
        predictedClass = self.match_to_array(predictedClass)
        sum = 0
        if len(trueClass)!=0 and len(predictedClass)!=0:
            for tc_keys,tc_value in trueClass.items():
                for keys_predicted,predicted_value in predictedClass.items():

                    if tc_value ==predicted_value:

                        sum +=1
                        break
            return str("Taxa de acerto na correspondência de pessoas: ") + str(float(sum/len(trueClass))*100) + str("%")

    def metricasBin(self,target, result):
        '''

        :param target:
        :param result:
        :return:
        '''
        target,result = self.remove_x(target,result)
        TP = np.sum(target[result == 1] == 1)

        FN = np.sum(target[result == 0] == 1)
        FP = np.sum(target[result == 1] == 0)
        TN = np.sum(target[result == 0] == 0)
        #precision = TP / (TP + FP)
        #recall = TP / (TP + FN)

        print("Precision = (TP/TP+FP)")
        print("Recall = (TP/TP+FN)")
        print("F1-score = 2 * ((precision*recall) / (precision +recall))")
        print("\n")
        print(classification_report(target, result, target_names=['Entrada', 'Saida']))
        #print("TP: " + str(TP), "   FN: " + str(FN), "   FP: " + str(FP), "   TN: " + str(TN))
        self.confusion_matrix(target,result)

    def score_counter(self,trueClass,result):
        x_times = result.count("x")
        print("Percentagem de acerto na contagem: " + str(((len(trueClass)-x_times)/len(trueClass))*100) + str("%"))

    def remove_x(self,trueClass,predictedClass):
        pc_p = []
        tc_c = []
        for i in range(len(predictedClass)):

            if predictedClass[i] != "x":
                pc_p.append(predictedClass[i])
                tc_c.append(trueClass[i])
        return tc_c,pc_p
    def match_to_array(self,match):
        '''

        :param match:
        :return:
        '''
        # lista de objetos Pessoa entrada/saida
        listaEntrada = match[1]
        listaSaida = match[2]

        # verificar se as listas não são nulas
        if len(listaEntrada) != 0 or len(listaSaida) != 0:
            # para cada pessoa dentro de cada lista
            for pessoaEntrada, pessoaSaida in itertools.zip_longest(listaEntrada, listaSaida):
                # verificar se existe matching entre entrada/saida
                if pessoaEntrada is not None and pessoaSaida is not None:
                    self.__matches_dict.update({self.__indice:[pessoaEntrada.id,pessoaSaida.id]})

                else:
                    if pessoaEntrada is None:
                        self.__matches_dict.update({self.__indice: [pessoaSaida.id]})

                    else:
                        self.__matches_dict.update({self.__indice: [pessoaEntrada.id]})
                self.__indice +=1

            return self.__matches_dict

p = Pickle_module("../../../Dataset/Pickles/matching_results.p")
result = p.load()
p = Pickle_module("../../../Dataset/Pickles/TC_19min.p")
trueClass = p.load()
mv = MethodValidation()




