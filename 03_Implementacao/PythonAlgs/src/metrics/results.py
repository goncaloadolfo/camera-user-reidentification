import sys

sys.path.append("..")
from metrics.method_validation import MethodValidation
from metrics.pickle_module import Pickle_module


if __name__ == "__main__":

    trueClass   = Pickle_module("../../../Dataset/Pickles/TC_19min.p")
    trueClass   = trueClass.load()

    pickle_k_10 = Pickle_module("../../../Dataset/Pickles/PC_K_10.p")
    result_k_10 = pickle_k_10.load()

    pickle_k_20 = Pickle_module("../../../Dataset/Pickles/PC_K_20.p")
    result_k_20 = pickle_k_20.load()

    pickle_k_30 = Pickle_module("../../../Dataset/Pickles/PC_K_30.p")
    result_k_30 = pickle_k_30.load()

    pickle_k_40 = Pickle_module("../../../Dataset/Pickles/PC_K_40.p")
    result_k_40 = pickle_k_40.load()

    pickle_t_01_k_10 = Pickle_module("../../../Dataset/Pickles/PC_t_01_k_10.p")
    result_t_01_k_10 = pickle_t_01_k_10.load()

    pickle_t_04_k_10 = Pickle_module("../../../Dataset/Pickles/PC_t_04_k_10.p")
    result_t_04_k_10 = pickle_t_04_k_10.load()

    pickle_iter_20000_k_10 = Pickle_module("../../../Dataset/Pickles/PC_iter_20000_k_10.p")
    result_iter_20000_k_10 = pickle_iter_20000_k_10.load()

    pickle_iter_k_10_t1 = Pickle_module("../../../Dataset/Pickles/PC_iter_k_10_t1.p")
    result_iter_k_10_t1 = pickle_iter_k_10_t1.load()

    pickle_iter_2500_t_05_k_10 = Pickle_module("../../../Dataset/Pickles/PC_iter_2500_t_05_k_10.p")
    result_iter_2500_t_05_k_10 = pickle_iter_2500_t_05_k_10.load()

    pickle_iter_5_k_20 = Pickle_module("../../../Dataset/Pickles/PC_iter_5_k_20.p")
    result_iter_5_k_20 = pickle_iter_5_k_20.load()

    contagem_tc_19min = Pickle_module("../../../Dataset/Pickles/contagem_tc_19.p")
    contagem_tc_19min = contagem_tc_19min.load()

    contagem_pc_19min = Pickle_module("../../../Dataset/Pickles/contagem_pc_19.p")
    contagem_pc_19min = contagem_pc_19min.load()

    print("Teste para k=10: ")
    mv = MethodValidation()
    print(mv.score_alternative_matches(trueClass,result_k_10))
    print(mv.score_matches(trueClass,result_k_10))
    print("")
    print("Teste para k=20: ")
    print(mv.score_alternative_matches(trueClass, result_k_20))
    print(mv.score_matches(trueClass, result_k_20))
    print("")
    print("Teste para k=30: ")
    print(mv.score_alternative_matches(trueClass, result_k_30))
    print(mv.score_matches(trueClass, result_k_30))
    print("")
    print("Teste para k=40: ")
    print(mv.score_alternative_matches(trueClass,result_k_40))
    print(mv.score_matches(trueClass, result_k_40))
    print("")
    print("Teste para t=0.1: ")
    print(mv.score_alternative_matches(trueClass,result_t_01_k_10))
    print(mv.score_matches(trueClass, result_t_01_k_10))
    print("")
    print("Teste para t=0.4: ")
    print(mv.score_alternative_matches(trueClass,result_t_04_k_10))
    print(mv.score_matches(trueClass, result_t_04_k_10))
    print("")
    print("Teste para iter=20000: ")
    print(mv.score_alternative_matches(trueClass,result_iter_20000_k_10))
    print(mv.score_matches(trueClass, result_iter_20000_k_10))
    print("")
    print("Teste para t=1.0 ")
    print(mv.score_alternative_matches(trueClass,result_iter_k_10_t1))
    print(mv.score_matches(trueClass, result_iter_k_10_t1))
    print("")
    print("Teste para t=0.5/iter=2500/k=10 ")
    print(mv.score_alternative_matches(trueClass,result_iter_2500_t_05_k_10))
    print(mv.score_matches(trueClass, result_iter_2500_t_05_k_10))
    print("")
    print("Teste para t=0/iter=5/k=10 ")
    print(mv.score_alternative_matches(trueClass,result_iter_5_k_20))
    print(mv.score_matches(trueClass, result_iter_5_k_20))
    mv.score_counter(contagem_tc_19min,contagem_pc_19min)
    mv.metricasBin(contagem_tc_19min,contagem_pc_19min)

