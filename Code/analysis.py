import numpy as np
import pickle
import sys

def analysis(file_name):

    file = open(file_name, "rb")
    out = pickle.load(file)
    file.close()

    labels = out["Labels"]
    fp = out["FP"]
    pred_undefended = out["UndefendedPrediction"]
    pred_defeneded_adv = out["AdversarialPredictions"]
    
    acc = 0.0

    detect = np.zeros((10,10))
    unchanged = np.zeros((10,10))
    defended = np.zeros(10)

    targets = np.array([0,1,2,3,4,5,6,7,8,9])

    N = len(fp)
    count = np.zeros((10))
    for i in range(N):

        p = pred_undefended[i]
        
        r = pred_defeneded_adv[i, :]
        
        d = 1 * (r == 10)
        u = 1 * (r == p)

        if not fp[i]:
            detect[p] += d
            unchanged[p] += u
            s = (sum(d) + sum(u) == 9) #The model either detected or didn't change for all attempted attacks
            defended[p] += 1 * s
            if p == labels[i] and s:
                acc += 1.0
            count[p] += 1

    detect /= count[:, None]
    unchanged /= count[:, None]
    defended /= count
    
    sys.stdout = open(file_name + "_analysis.txt", "w")
    print("Adversarial Accuracy")
    print(np.round(acc / N, 4))
    print("False Positive Rate")
    print(np.round(np.sum(fp) / N, 4))
    print("Starting with real images NOT labeled adversarial by the defense")
    #print("Defense Success Rate (either prediction didn't change or labeled as adversarial")
    #print(np.round(unchanged + detect, 4))
    #print("Defense Success Rate: Succeeded for all targets")
    #print(np.round(defended, 4))
    print("Overall Defense Success Rate")
    print(np.round(np.dot(defended, count) / sum(count), 4))
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    analysis(sys.argv[1])
