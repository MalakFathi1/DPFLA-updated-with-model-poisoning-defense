from server import run_exp
from federated_learning.my_dict import replace_1_with_7, replace_dog_with_cat,replace_6_with_9
from federated_learning.my_dict import run_mnist, run_cifar10
from federated_learning.my_dict import fed_avg, simple_median, trimmed_mean, multi_krum, fools_gold, DPFLA

if __name__ == '__main__':
    NUM_WORKERS = 50
    FRAC_WORKERS = 1
    ATTACK_TYPE1 = "gaussian"
    #ATTACK_TYPE = "backdoor"
    ATTACK_TYPE2 ="gaussian"
    GLOBAL_ROUND = 50
    LOCAL_EPOCH = 1
    UNTARGET = False

    REPLACE_METHOD = replace_6_with_9()
    RULE = DPFLA()
    DATASET = run_mnist()

    # REPLACE_METHOD = replace_dog_with_cat()
    # RULE = DPFLA()
    # DATASET = run_cifar10()

    MALICIOUS_RATE = [0, 0.1, 0.2, 0.3, 0.4]  # 0, 0.1, 0.2, 0.3, 0.4, 0.5
    MALICIOUS_BEHAVIOR_RATE = 1

    for rate in MALICIOUS_RATE:
        run_exp(NUM_WORKERS, FRAC_WORKERS, ATTACK_TYPE1 ,ATTACK_TYPE2, RULE, REPLACE_METHOD, DATASET, rate,
                MALICIOUS_BEHAVIOR_RATE, GLOBAL_ROUND, LOCAL_EPOCH, UNTARGET)