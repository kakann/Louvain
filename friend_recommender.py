from collections import defaultdict
from email.policy import default
from fileinput import filename
from os import SCHED_OTHER
from re import L
from jinja2 import Undefined
import networkx as nx
import networkx.algorithms.community as nx_comm
import csv
from typing import List, Optional, Tuple, Union
import time
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import sys
import random
import networkx.algorithms.community as nx_comm
import pyfof
from requests import delete
import user

def open_csv(filename : str, threshold):
    file = open(filename)
    csvreader = csv.reader(file, delimiter='\t')
    edge_list : List[Tuple[str, str]]= []
    i = 0
    for being_followed, follower, date in csvreader:
        edge_list.append(tuple((being_followed, follower)))
        if i > threshold:
            break
        i+=1
    return edge_list

def clean_data(edge_list : list(tuple((str, str))), threshold):
    being_followed_dict : dict(str, int) = defaultdict(int)
    follower_dict : dict(str, int) = defaultdict(int)
    for being_followed, follower in edge_list:
        being_followed_dict[being_followed] += 1
        follower_dict[follower] += 1
    
    cleaned_edge_list = []
    for being_followed, follower in edge_list:
        if being_followed_dict[being_followed] + follower_dict[being_followed] >= threshold and follower_dict[follower] + being_followed_dict[being_followed] >= threshold:
            cleaned_edge_list.append(tuple((being_followed, follower)))
    return cleaned_edge_list

def edge_list_to_dict(edge_list):
    follower_being_followed_dict : dict(str, set) = defaultdict(set)
    for being_followed, follower in edge_list:
        follower_being_followed_dict[follower].add(being_followed)
    return follower_being_followed_dict

def training_test_split(ratio : int, follower_being_followed_dict):
    training_edge_list = []
    test_edge_list = []
    for follower in follower_being_followed_dict:
        length = len(follower_being_followed_dict[follower])
        random.shuffle(list(follower_being_followed_dict[follower]))
        elements_to_remove = int(length * ratio)

        for being_followed in list(follower_being_followed_dict[follower])[0:elements_to_remove]:
            training_edge_list.append(tuple((being_followed, follower)))
        for being_followed in list(follower_being_followed_dict[follower])[elements_to_remove:length]:
            test_edge_list.append(tuple((being_followed, follower)))
    return (training_edge_list, test_edge_list)

def count_users(edge_list):
    users : set = set()
    for being_followed, follower in edge_list:
        users.add(being_followed)
        users.add(follower)
    return users

def R(communities, training_edge_dict):
    Mc : list() = []
    x = 0
    for community in communities:
        print(f"Creating R for community {x} of SIze {len(community)}")
        x=x+1
        dim = len(community)
        M = [[0 for x in range(dim)] for y in range(dim)] 
        for i in range(dim):
            for j in range(dim):
                if list(community)[j] in training_edge_dict[list(community)[i]]:
                    M[i][j] = 1
                #elif i == j:
                #    M[i][j] = 1
                else:
                    M[i][j] = 0
        assert len(M) == len(community)
        Mc.append(M)
    return Mc

def matrix_factorizer(Mc, K, communities):
    i = 0
    communities_nR = []
    for R in Mc:
        assert len(R) == len(communities[i])
        i=i+1
        N = len(R)
        M = N

        P = np.random.rand(N, K)
        Q = np.random.rand(M, K)

        nP, nQ = matrix_factorization(R, P, Q, K)
        nR = np.dot(nP, nQ.T)

        communities_nR.append(nR)
    return communities_nR

def matrix_factorization(R, P, Q, K, steps=5, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter
    '''
    Q = Q.T   
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        

        eR = np.dot(P,Q)

        e = 0
        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T


def top_k_followes(row, user_dict, user, community, k):
    indexes = np.argsort(row)[-k:]
    for index in indexes:
        user_dict[user].append(list(community)[index])

def L_intersect_L_p(L, L_p):
    return len(list(L & L_p))
def c_conversion_rate(L, L_p):
    return 1 if  L_intersect_L_p(L, L_p) > 0 else 0
def c_recall(L, L_p):
    return 0 if  len(L_p) == 0 else L_intersect_L_p(L, L_p)/len(L_p)
def c_precision(L, L_p):
    if len(L_p) > 0:
        assert(L_intersect_L_p(L, L_p) <= len(L))
    return 0 if len(L_p) == 0 else L_intersect_L_p(L, L_p)/len(L)

def calculate_metrics(communities, communities_nR, follower_followe_dict, K):
    i = 0
    recall_avg = 0
    precision_avg = 0
    conversion_rate_avg = 0

    for (community, nR) in zip(communities, communities_nR):
        #print(f"Verifying data for community {i}")
        community_size = len(community)
        
        top_k_followes_for_user : dict(str, list) = defaultdict(list)
        for (row, user) in zip(nR, community):
            assert len(row) == community_size
            top_k_followes(row, top_k_followes_for_user, user, community, K)
        
        conversion_rate = 0
        recall = 0
        precision = 0
        #print(f"edge dict len {len(followe_follower_dict)} edge dict values {len(list(followe_follower_dict.values()))}")
        
        for user in top_k_followes_for_user:
            #Predicted followes
            L = set(top_k_followes_for_user[user])
            #actual followes
            L_p = set(follower_followe_dict[user])
            
            conversion_rate += c_conversion_rate(L, L_p)
            recall += c_recall(L, L_p)
            precision += c_precision(L, L_p)

        assert (conversion_rate / community_size) < 1
        conversion_rate_avg= conversion_rate_avg + (conversion_rate/ community_size)
        recall_avg = recall_avg + (recall / community_size)
        precision_avg = precision_avg + (precision / community_size)
        #assert(precision_avg <= 1)
        i = i + 1

    return conversion_rate_avg/len(communities), recall_avg/len(communities), precision_avg/len(communities)



def calculate_avg_metrics_for_community(c_avg, c , r_avg, r, p_avg, p, community_size):
    c_avg= c_avg + (c/ community_size)
    r_avg = r_avg + (r/ community_size)
    p_avg = p_avg + (p/ community_size)
    



def main():

    if len(sys.argv) != 3:
        print("detection.py resolution split ratio")
        quit()
    resolution = int(sys.argv[1])
    test_training_split = float(sys.argv[2])

    edge_list = open_csv("followingAugSept.csv", 200000)
    print(f"len edge_list after reading csv {len(edge_list)}")
    print(f"Users after reading: {len(count_users(edge_list))}")
    edge_list = clean_data(edge_list, 10)
    print(f"len edge_list {len(edge_list)}")
    print(f"Users after cleaning: {len(count_users(edge_list))}")
    edge_dict = edge_list_to_dict(edge_list)
    amount_begin_followed = 0
    for follower in edge_dict:
        amount_begin_followed += len(edge_dict[follower])
        #print(f" {follower} are following {len(edge_dict[follower])} users.")
    print(f"{amount_begin_followed} users are being followed")
    training_edge_list, test_edge_list = training_test_split(test_training_split, edge_dict)
    print(f"len edge_list {len(training_edge_list)}")
    print(f"Users after training/test split: {len(count_users(training_edge_list))}")
    training_edge_dict = edge_list_to_dict(training_edge_list)
    amount_begin_followed = 0
    O_followers = 0
    for follower in training_edge_dict:
        amount_begin_followed += len(training_edge_dict[follower])
        if len(training_edge_dict[follower]) == 0:
            O_followers += 1
        #print(f" {follower} are following {len(training_edge_dict[follower])} users.")
    print(f"followers {len(training_edge_dict)}")
    print(f"followers following 0 users {O_followers}")
    print(f"{amount_begin_followed} users are being followed")

    start = time.perf_counter()
    G = nx.Graph()
    G = nx.from_edgelist(training_edge_list)
    end = time.perf_counter()
    print(f"Construcing the graph took: {end - start} seconds")

    start = time.perf_counter()
    communities = nx_comm.louvain_communities(G, resolution=resolution)
    end = time.perf_counter()
    print(f"Louvain timer: {end - start} seconds")
    print(f"Ammount of communities: {len(communities)}")
    print(f"Modularity of the graph: {nx_comm.modularity(G, communities)}")

    
    Mc = R(communities, training_edge_dict)
    communities_nR =matrix_factorizer(Mc, K=1, communities=communities)
    
    precision_list_k = []
    conversion_list_k = []
    recall_list_k = []

    for K in range(1, 6):
        #print(f"Calculating metrics for K={K}")
        conversion, recall, precision = calculate_metrics(communities, communities_nR, edge_dict, K)
        assert(precision <= 1)
        precision_list_k.append(precision)
        recall_list_k.append(recall)
        conversion_list_k.append(conversion)

    print("K-values:       1, 2, 3, 4, 5")
    print(f"Final recall: {recall_list_k}")
    print(f"Final precision: {precision_list_k}")
    print(f"Final conversion: {conversion_list_k}")

    K_range = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(3)
    fig.suptitle(f"Metrics when Louvain resolution = {resolution} with train/test split {test_training_split}")
    ax[0].plot(K_range, precision_list_k)
    ax[0].set_title("precision")
    ax[1].plot(K_range, recall_list_k, label = "recall")
    ax[1].set_title("recall")
    ax[2].plot(K_range, conversion_list_k, label = "conversion rate")
    ax[2].set_title("conversion")
    plt.savefig(f"metricsR={resolution}_traintest={test_training_split}.png")
    plt.show()






if __name__ == "__main__":
    main()




        
