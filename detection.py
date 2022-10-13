from collections import defaultdict
from email.policy import default
from fileinput import filename
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



def open_csv(filename : str, header : List[str]):
    file = open(filename)
    csvreader = csv.reader(file, delimiter='\t')
    rows : List[Tuple[str, str]]= []
    i = 0
    for row in csvreader:
        rows.append(row[:2])
    return (header, rows)

def open_csv_dic(filename : str, header : List[str]):
    file = open(filename)
    csvreader = csv.reader(file, delimiter='\t')
    rows : dict(str, list) = defaultdict(list)
    i = 0
    for followedId, followedById, date in csvreader:
        rows[followedId].append(followedById)
        i = i + 1
        if i> 200:
            break
        
    return (header, rows)

def clean_data(dictionary,  threshold : int):
    edge_list : list[Tuple[str, str]] = []
    for key in dictionary:
        if (len(dictionary[key]) > threshold):
            for follower in dictionary[key]:
                edge_list.append((key, follower))
    return edge_list

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
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


        
def extract_test_followes(edge_dict, test_traning_ratio : float):
    #Transform dict with key=follwe, value=list(followers) to key= follower value=list(followes)
    follower_dict : dict(str, list) = defaultdict(list)
    for followe in edge_dict:
        for follower in edge_dict[followe]:
            follower_dict[follower].append(followe)
    print(follower_dict)
    
    #Get test and training data
    training_edge_dict : dict(str, list) = defaultdict(list)
    test_edge_dict : dict(str, list) = defaultdict(list)
    tmp_traning_data= {}
    for follower in follower_dict:
        length = len(follower_dict[follower])
        no_elems_to_keep = int(length * test_traning_ratio)
        training_data = set(random.sample(follower_dict[follower], no_elems_to_keep))
        
        test_edge_dict[follower] =  training_data.difference(follower_dict[follower])
        training_edge_dict[follower] = training_data
    #print(training_edge_dict)
    
    #Transform back to original format
    result_training_dict : dict(str, list) = defaultdict(list)
    result_test_dict : dict(str, list) = defaultdict(list)
    for test_follower in test_edge_dict:
        for test_followe in test_edge_dict[test_follower]:
            result_test_dict[test_followe].append(test_follower)

    for training_follower in training_edge_dict:
        for training_followe in test_edge_dict[training_follower]:
            result_test_dict[training_followe].append(training_follower)

    return result_training_dict, result_test_dict 

def main():

    if len(sys.argv) != 2:
        print("detection.py Threshold")
        quit()
    threshold = int(sys.argv[1])

    print("Reading csv")
    edge_list = []
    edge_dict = {}
    header : List[str]= ["FollowedID", "FollowedByID"]

    start = time.perf_counter()
    (header, edge_dict) = open_csv_dic("followingAugSept.csv", header)
    end = time.perf_counter()
    print(f"Reading the csv took: {end - start} seconds")


    training_edge_dict = {}
    test_edge_dict = {}
    training_edge_dict, test_edge_dict = extract_test_followes(edge_dict, 0.9)

    print(len(training_edge_dict))
    print(test_edge_dict)


    start = time.perf_counter()
    edge_list = clean_data(edge_dict, threshold=threshold)
    end = time.perf_counter()
    print(f"Lenght of edge list: {len(edge_list)}")
    print(f"Cleaning the data took: {end - start} seconds")





    start = time.perf_counter()
    G = nx.Graph()
    G = nx.from_edgelist(edge_list)
    end = time.perf_counter()
    print(f"Construcing the graph took: {end - start} seconds")

    start = time.perf_counter()
    communities = nx_comm.louvain_communities(G)
    end = time.perf_counter()
    print(f"Louvain timer: {end - start} seconds")
    print(f"Ammount of communities: {len(communities)}")


    #print(communities[0])
    #print(edge_dict)
    Mc : list() = []
    i = 0
    for community in communities:
        print(f"Creating R for community {i}")
        i=i+1
        dim = len(community)
        M = [[0 for x in range(dim)] for y in range(dim)] 
        for i in range(len(community)):
            for j in range(len(community)):
                if list(community)[j] in edge_dict[list(community)[i]]:
                    M[i][j] = 1
                else:
                    M[i][j] = 0
        Mc.append(M)
    i = 0
    for R in Mc:
        print(f"Matrix factorization for community: {i}")
        i=i+1
        N = len(R)
        M = N
        K = 1
        P = np.random.rand(N, K)
        Q = np.random.rand(M, K)

        nP, nQ = matrix_factorization(R, P, Q, K)

        nR = np.dot(nP, nQ.T)
        print(f"Community Size: {N}")
        print(nR)

    print("done")
    


    

    









    
    #Assign Colors
    node_colors = []
    colored_nodes = []
    start = time.perf_counter()
    for edge in edge_list:
        for node in edge:
            if node not in colored_nodes:
                colored_nodes.append(node)
                i = 0
                for community in communities:
                    if node in community:
                        node_colors.append(i)
                        break
                    else:
                        i = i + 1
    end = time.perf_counter()
    print(f"Coloring took: {end - start} seconds")

    start = time.perf_counter()
    
    # drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.spring_layout(G)
    ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                            node_size=50, cmap=plt.cm.jet)
    plt.axis('off')
    end = time.perf_counter()
    print(f"Drawing the graph took: {end - start} seconds")
    #plt.savefig("communities1.png")
    plt.show()
    



    #print (colors)
    #colors : list = ["r", "b", "g"]
    #pos = nx.spring_layout(G, seed=225)
    #nx.draw(G, pos, node_color= range(203), cmap=plt.cm.jet)
    #nx.draw_networkx_nodes();
    #plt.show()
    

if __name__ == "__main__":
    main()








    

    
