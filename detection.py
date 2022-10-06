from collections import defaultdict
from email.policy import default
from fileinput import filename
import networkx as nx
import networkx.algorithms.community as nx_comm
import csv
from typing import List, Optional, Tuple, Union
import time
import matplotlib.pyplot as plt

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
        #i = i + 1
        #if i > 5000000:
        #    break
    return (header, rows)

def clean_data(dictionary,  threshold : int):
    edge_list : list[Tuple[str, str]] = []
    for key in dictionary:
        if (len(dictionary[key]) > threshold):
            for follower in dictionary[key]:
                edge_list.append((key, follower))
    return edge_list
        

def main():

    print("Reading csv")
    edge_list = []
    edge_dict = {}
    header : List[str]= ["FollowedID", "FollowedByID"]

    start = time.perf_counter()
    (header, edge_dict) = open_csv_dic("followingAugSept.csv", header)
    end = time.perf_counter()
    print(f"Reading the csv took: {end - start} seconds")


    start = time.perf_counter()
    edge_list = clean_data(edge_dict, threshold=1000)
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
    print(f"Lenght of edge list after graph is constructed: {len(edge_list)}")
    print(f"Louvain timer: {end - start} seconds")
    print(f"Ammount of communities: {len(communities)}")

    #pos = nx.spring_layout(G, seed=225)
    #nx.draw(G, pos)
    #plt.show()
    

if __name__ == "__main__":
    main()








    

    