import numpy as np
import pickle
import torch

def minmax_val(value, minm, maxm):
    return (value - minm) / (maxm - minm)

def norm_app(app):
    R,N,C,G,P,A = app.detach().numpy()

    return [R, minmax_val(N, 0.0111, 4.0316), minmax_val(C, 0.0085, 1.0876), G, P, minmax_val(A, 0.0402, 0.5072)]

def emotion(appraisal):

    R,N,C,G,P,A = appraisal.detach().numpy()

    F = max(0, min(R + N - C, 1))
    A = max(0, min((1 - R) + (1 - G), 1))
    J = max(0, min(R + G, 1))
    S = max(0, min((1 - R) + (1 - G) - P, 1))
    D = max(0, min((1 - R) + (1 - C), 1))
    U = max(0, min(R + A - N, 1))

    return [F,A,J,S,D,U]


def stress_prev(appraisal):
    R,N,C,G,P,A = appraisal.detach().numpy()

    # weighted_values = [R, minmax_val(N, 0.0111, 4.0316), minmax_val(C, 0.0085, 1.0876), G, P, minmax_val(1-A, 0.0402, 0.5072)]

    # Define weights for each appraisal variable
    weights = [0.175, 0.25, 0.125, 0.25, 0.075, 0.125]
      
    # Apply weights to each appraisal variable
    weighted_values = [R,N,C,G,P,1-A]
    weighted_sum = sum(value * weight for value, weight in zip(weighted_values, weights))
    
    # Normalize the weighted sum to obtain the stress level between 0 and 1
    stress_level = (weighted_sum / sum(weights))
    # stress_level = 1/(1+np.exp(stress_level))
    # stress_level = 1/(1+np.exp(stress_level))
    return stress_level

def emo_app(app):
    R,N,C,G,P,A = app.detach().numpy()
    emo_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Guilt', 4: 'Joy', 5: 'Sadness', 6: 'Shame'}
    with open('emotion_estimator.pkl', 'rb') as f:
        emo_clf = pickle.load(f)
    # normapp = [R, minmax_val(N, 0.0111, 4.10), minmax_val(C, 0.000, 1.00), G, P, minmax_val(A, 0.0402, 0.5072)]
    R = (R-0.530971)/0.407309
    N = (N-0.772037)/0.110288
    C = (C-0.106702)/0.133804
    G = (G-0.425943)/0.345998
    P = (P-0.629762)/0.258976
    A = (A-0.938952)/0.127975

    # if(R > 4.956353e-17 and C > -2.379049e-17 and G < 1.467080e-16 and P < -6.740640e-17 and A < -6.938894e-18):
    #     print("Angry")
    # if(N > 4.956353e-17 and C < -2.379049e-17 and P < -6.740640e-17 and R < 4.956353e-17):
    #     print("Disgust")
    # if(N > 4.956353e-17 and C > -2.379049e-17 and P < -6.740640e-17 and G < 1.467080e-16):
    #     print("Fear")
    # if(C < -2.379049e-17 and P < -6.740640e-17 and G > 1.467080e-16 and A > -6.938894e-18):
    #     print("Guilt")
    # if(P > -6.740640e-17 and G > 1.467080e-16 and R > 4.956353e-17):
    #     print("Joy")
    # if(P < -6.740640e-17 and G < 1.467080e-16 and R < 4.956353e-17):
    #     print("Sadness")
    # if(P < -6.740640e-17 and C > -2.379049e-17 and A < -6.938894e-18):
    #     print("Shame")

    normapp = [R,N,C,G,P,A]
    return emo_dict[emo_clf.predict([normapp])[0]]

def norm_app(app):
    norm = []
    for i in range(len(app)):
        R,N,C,G,P,A = app[i].detach().numpy()
        # norm.append([R, minmax_val(N, 0.0111, 4.10), minmax_val(C, 0.000, 1.00), G, P, minmax_val(A, 0.0402, 0.5072)])
        norm.append([R,N,C,G,P,A])
    return torch.tensor(norm, dtype=torch.float32)

def stress(app):
    stress_level = []
    for i in range(len(app)):
        R,N,C,G,P,A = app[i].detach().numpy()
        # Define weights for each appraisal variable (you can adjust these weights based on your specific study or domain knowledge)
        weights = [0.25, 0.05, 0.1, 0.2, 0.35, 0.05] # [0.2, 0.1, 0.1, 0.2, 0.2, 0.2]
        gapp = [1-R,1-N,1-C,1-G,1-P,1-A]
        # Calculate the stress level
        stress_level.append(sum([a*b for a,b in zip(gapp, weights)]))
        # st = (0.8 * (P) - 0.2 * (R + N + A + G + C))
        # st_level = (st + 0.8) / 2.4
        # stress_level.append(st_level)
    return np.average(stress_level)