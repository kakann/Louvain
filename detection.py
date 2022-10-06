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
        if i > 200:
            break
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
    edge_list = clean_data(edge_dict, threshold=1)
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
    #print(f"Lenght of edge list after graph is constructed: {len(edge_list)}")
    print(f"Louvain timer: {end - start} seconds")
    print(f"Ammount of communities: {len(communities)}")

    #Transform data to be plottable
    node_groups : dict(str) = {}
    nodes_with_edge_data : list[Tuple[str, dict]] = []
    i = 0
    for community in communities:
        node_groups[i] = community
        i= i + 1

    #print(f"Node groups {node_groups}")

    for community in node_groups:
        for node in node_groups[community]:
            group : dict = {}
            group['g'] = community
            nodes_with_edge_data.append((node, group))
    #print(f" Nodes with edge data: {nodes_with_edge_data}")

    print(G.nodes())
    node_colors = []

    color_list = np.linspace(0, 1, len(edge_dict))

    for node in nodes_with_edge_data:
        node_colors.append(node[1]['g'])

    

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.spring_layout(G)
    ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(G, pos,  node_color=node_colors, 
                            node_size=100, cmap=plt.cm.jet)
    plt.colorbar(nc)
    plt.axis('off')
    plt.show()



    #print (colors)
    #colors : list = ["r", "b", "g"]
    #pos = nx.spring_layout(G, seed=225)
    #nx.draw(G, pos, node_color= range(203), cmap=plt.cm.jet)
    #nx.draw_networkx_nodes();
    #plt.show()
    

if __name__ == "__main__":
    main()








    

    