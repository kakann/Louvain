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

users = {}

def get_user(user_id):
    if user_id not in users:
        users[user_id] = user.User(user_id)
    return users[user_id]

def open_csv(filename : str, header : List[str]):
    file = open(filename)
    csvreader = csv.reader(file, delimiter='\t')
    rows : List[Tuple[str, str]]= []
    i = 0
    for row in csvreader:
        rows.append(row[:2])
    print(f"len of original edgelist {len(csvreader)}")
    return (header, rows)

def open_csv_dic(filename : str, header : List[str]):
    file = open(filename)
    csvreader = csv.reader(file, delimiter='\t')
    follower_followe : dict(str, list) = defaultdict(list)
    followe_followers : dict(str, list) = defaultdict(list)
    i = 0
    

    edge_list = []
    for followedId, followedById, date in csvreader:
        edge_list.append(tuple((followedId, followedById)))
        i +=1
        if i > 200000:
            break
        #get_user(followedId).add_followers(get_user(followedById))
        #get_user(followedById).add_follows(get_user(followedId))

        #i+=1
        #if i > 200000:
        #    break
    #users1 = {}
    #print(f"lenght of user dict {len(users)}")
    #for id, user in list(users.items()):
    #    if len(user.followers) <= 10 or len(user.follows) <= 10:
    #        #print(f"User removed: {id} with {len(user.follows)} followers and {len(user.followers)} followers")
    #        user.delete_user()
    #    else:
    #        users1[id] = user
    print(f"lenght of user dict {len(users)}")
    #remove all users with less than 10 followers/follwes
    
    followe_dict : dict(str, int) = defaultdict(int)
    follower_dict : dict(str, int) = defaultdict(int)
    for followe, follower in edge_list:
        followe_dict[followe] += 1
        follower_dict[follower] += 1

    print(len(followe_dict))
    print(len(follower_dict))

    original_len = 0
    after_removal_n = 0
    for followe, follower in edge_list:
        original_len += 1
        if followe_dict[followe] + follower_dict[followe] >= 10 and follower_dict[follower] + followe_dict[follower] >= 10:
            after_removal_n +=1
            followe_followers[followe].append(follower)
            follower_followe[follower].append(followe)

    
        
    print(f"len of original edgelist: {original_len}")
    print(f"len when 10 followers/followes removed: {after_removal_n}")
    
    for followe, follower in edge_list:
        followe_followers[followe].append(follower)
    return (header, followe_followers)

def convert_dict_to_edgelist(dictionary):
    edge_list : list[Tuple[str, str]] = []
    for key in dictionary:
        #assert(len(dictionary[key]) < 10)
        for follower in dictionary[key]:
                edge_list.append((key, follower))
    return edge_list

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


        
def extract_test_followes(followe_follower_dict, test_training_ratio : float):
    #Transform dict with key=follwe, value=list(followers) to key= follower value=list(followes)
    
    follower_dict : dict(str, list) = defaultdict(list)
    for followe in followe_follower_dict:
        for follower in followe_follower_dict[followe]:
            follower_dict[follower].append(followe)

    #Get test and training data
    training_edge= []
    test_edge = []
    for follower in follower_dict:
        length = len(follower_dict[follower])
        random.shuffle(follower_dict[follower])
        elements_to_remove = int(length * test_training_ratio)

        for followe in follower_dict[follower][0:elements_to_remove]:
            training_edge.append(tuple((followe, follower)))
        for followe in follower_dict[follower][elements_to_remove:length]:
            test_edge.append(tuple((followe, follower)))


    #Transform back to original format
    result_training_dict : dict(str, list) = defaultdict(list)
    result_test_dict : dict(str, list) = defaultdict(list)
    for followe, follower in training_edge:
        result_training_dict[followe].append(follower)
    for followe, follower in test_edge:
        result_test_dict[followe].append(follower)

    return result_training_dict, result_test_dict 

def R(communities, trainin_followe_follower_dict):
    Mc : list() = []
    x = 0
    for community in communities:
        #print(f"Creating R for community {x} of SIze {len(community)}")
        x=x+1
        dim = len(community)
        M = [[0 for x in range(dim)] for y in range(dim)] 
        for i in range(dim):
            for j in range(dim):
                if list(community)[j] in trainin_followe_follower_dict[list(community)[i]]:
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

def calculate_metrics(communities, communities_nR, followe_follower_dict, K):
    i = 0
    recall_avg = 0
    precision_avg = 0
    conversion_rate_avg = 0

    follower_followe_dict : dict(str, list) = defaultdict(list)
    for followe in followe_follower_dict:
        for follower in followe_follower_dict[followe]:
            follower_followe_dict[follower].append(followe)

    for (community, nR) in zip(communities, communities_nR):
        #print(f"Verifying data for community {i}")
        community_size = len(community)
        

        user_dict : dict(str, list) = defaultdict(list)
        for (row, user) in zip(nR, community):
            assert len(row) == community_size
            top_k_followes(row, user_dict, user, community, K)
        
        conversion_rate = 0
        recall = 0
        precision = 0
        #print(f"edge dict len {len(followe_follower_dict)} edge dict values {len(list(followe_follower_dict.values()))}")
        
        for user in user_dict:
            #Predicted followes
            L = set(user_dict[user])
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
    
def calculate_metrics_for_user(user_dict, followe_follower_dict):
    conversion_rate = 0
    recall = 0
    precision = 0
    for user in user_dict:
        #Predicted followes
        L = set(user_dict[user])
        #actual followes
        L_p = set(followe_follower_dict[user])
        conversion_rate += c_conversion_rate(L, L_p)
        recall += c_recall(L, L_p)
        precision += c_precision(L, L_p)
    return conversion_rate, recall, precision
    

def main():

    if len(sys.argv) != 3:
        print("detection.py resolution split ratio")
        quit()
    resolution = int(sys.argv[1])
    test_training_split = float(sys.argv[2])
    users: dict(str, user.User) = defaultdict(user.User)

    print("Reading csv")
    edge_list = []
    followe_follower_dict = {}
    header : List[str]= ["FollowedID", "FollowedByID"]

    start = time.perf_counter()
    (header, followe_follower_dict) = open_csv_dic("followingAugSept.csv", header)
    end = time.perf_counter()
    print(f"Reading the csv took: {end - start} seconds")


    trainin_followe_follower_dict = {}
    test_followe_follower_dict = {}
    trainin_followe_follower_dict, test_followe_follower_dict = extract_test_followes(followe_follower_dict, test_training_split)

    print(f"Training data len{len(list(trainin_followe_follower_dict.values()))}")
    print(f"Test data len {len(list(test_followe_follower_dict.values()))}")


    start = time.perf_counter()
    edge_list = convert_dict_to_edgelist(followe_follower_dict)
    end = time.perf_counter()
    print(f"Lenght of NEW edge list: {len(edge_list)}")
    print(f"Converting to edge_list took {end - start} seconds")

    start = time.perf_counter()
    G = nx.Graph()
    G = nx.from_edgelist(edge_list)
    end = time.perf_counter()
    #print(f"Construcing the graph took: {end - start} seconds")
    #print(f" Nodes len before clean {len(G.nodes())}")
    #remove = [node for node, degree in G.degree() if degree < 5]
    #G.remove_nodes_from(remove)
    #print(f" Nodes len before clean {len(G.nodes())}")


    start = time.perf_counter()
    communities = nx_comm.louvain_communities(G, resolution=resolution)
    end = time.perf_counter()
    print(f"Louvain timer: {end - start} seconds")
    print(f"Ammount of communities: {len(communities)}")
    print(f"Modularity of the graph: {nx_comm.modularity(G, communities)}")

    
    Mc = R(communities, trainin_followe_follower_dict)
    communities_nR =matrix_factorizer(Mc, K=1, communities=communities)
    
    precision_list_k = []
    conversion_list_k = []
    recall_list_k = []

    for K in range(1, 6):
        #print(f"Calculating metrics for K={K}")
        conversion, recall, precision = calculate_metrics(communities, communities_nR, followe_follower_dict, K)
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


    
    





        
    print("done")
    '''
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
    '''
    

if __name__ == "__main__":
    main()








    

    
