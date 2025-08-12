#goal for today (do not have to reach it), is experimental driver
#this is a): random graph generation with maximum O(root(n)) degree per node and
#b) subidivision mapping, linear time planarity testing
#gonna use the networkx library (comes with kurotawksi subraph O(n) planarity checking I really don't wannat redo from scratch cuz that would be a whole project in and of itself). This afternoon, watch a tutorial


#today, thursday: set this up in a trialable manner

import networkx as nx
import matplotlib.pyplot as plt
import random
import copy
import sys
import time

def generateGraph(numNodes): #creates a graph with n nodes and with each node having an average degree of root(n)
    #maxDegree = averageDegree #misnomer
    G = nx.Graph()
    #lets start with 64 nodes with degree up to 8
    #lets adjust the weights with a "priority matrix" we multiply with the remaining set of vertices for priority

    #numNodes = 64 #please use square numbers

    myNodes = [i for i in range(0,numNodes)]

    #randomly generated graph mapping

    maxDegree = int(numNodes**(1/2)) #misnomer, I actually meant "average degree" lol oops


    edgesMap = {i:dict() for i in myNodes}

    #unit test step by step!
    nodesList = myNodes[:-1] #utility thing for the next step dw about it
    for node in edgesMap:
        numConnections = random.randint(1,maxDegree-1) #do not do self connections
        connections = random.sample(nodesList,numConnections)
        for connection in connections:
            if connection >= node:#avoids self connections
                connection += 1
            edgesMap[node][connection] = True
    #with this procedure, the average degree is root(n) + odds of each other node making connection to this one
    #so is root(n) + n * (chance that another node targets this one)
    # which is root(n) + n * (root n / n) = 2 * root n
    #that does make sense though, since there are n root n total edges, and so the sum of the degrees should be 2 n root n 
    #which makes the average degree 2 root n 
    G.add_nodes_from(connections)
    
    for node in edgesMap: #generate the graph
        for connection in edgesMap[node]:
            edge = (node,connection)
            G.add_edge(*edge)
    #nx.draw(G)
    #plt.show()
    
    return G










#steps 1 and 2: break into planar subgraphs
def getPlanarSubgraphs(myGraph,verbose = True):
    start_time = time.time()
    graphCopy = copy.deepcopy(myGraph) #check that it actually deepcopies
    
    myMap = [] #map of graphs. note that in networkx nodes can be anything, including pointers to graphs.
    
    while True: #outer loop
        isPlanar,certificate =  nx.check_planarity(graphCopy)
        if isPlanar:
            myMap += [list(graphCopy)] #remaining nodes are a "survivor" set
            break
        
        nextGraphCopy = copy.deepcopy(graphCopy)
        myNodes = set(list(nextGraphCopy))
        #print("cut1")
        while True: #inner loop 
            #first, pick a victim node
            #print("cut2")
            victim = random.choice(list(myNodes)) #list call is a pythonic compromise, does not increase amortization order as check planarity is in this loop anyways
            #print("victim:",victim)
            myNodes.remove(victim) #???
            #print(myNodes)
            nextGraphCopy.remove_node(victim)
            
            isPlanar,certificate =  nx.check_planarity(nextGraphCopy)
            if isPlanar:

                survivors = list(nextGraphCopy)
                if verbose:
                    print("survivors:",survivors)
                myMap += [survivors]
                graphCopy.remove_nodes_from(survivors)
                break
    
    #print("subgraphs:",myMap) #next step: map connections between subgraphs, draw them, etc
    total_time = time.time() - start_time
    print("planar subgraph divisions took:",total_time)
    return myMap
    
    
#test with fully connected graph?
#G = nx.complete_graph(32) 
#64:16, 256:64, 16:4, 32:8 
#With complete graphs, the number of subgraphs is n/4... 
#Can we say that to put together max-ind sets, we can just trim "border vertices"?

#G = generateGraph(64)

#sgs = getPlanarSubgraphs(G)
#print("subgraphs:",sgs)
#print("number of subgraphs:",len(sgs))


#lets find the relationship between number of planar subgraphs and n given the constraint that average degree ~= root(n)
testSizes = [36,64,100,144,196,256,324,400,676,900,1024] #order 0.6, sublinear?
#with O(n^0.6) subgraphs, we get O(n^1.2) possible "border edges" between subgraphs, so for max ind set we get at least optimal - n^1.2 ?
#this is not P = NP FPTAS because it requires the condition that average degree <= C*root(n), C >= 1 

#maybe the planarity check in this library is not O(N) and I have to write one myself :(
#next step: address this if necessary
#and then also find or implement baker's method
def getBorderEdges(subgraphs,G):
    border_edges = set()
    for edge in G.edges():
        node1, node2 = edge
        is_border_edge = True
        for subgraph in subgraphs:
            if node1 in subgraph and node2 in subgraph:
                is_border_edge = False
                break
        if is_border_edge:
            border_edges.add(edge)
    return border_edges

#test baker's algo here
 #problem: we have to do this for each fully connected compoment, of which there may be several!

def getIndSet(G,sgs,k = 10):#okay lets see if we can do better
    start_time = time.time()
    solutions = set() #set instead of list for O(1) "in" query
    for subgraphSet in sgs:
        subgraph = G.subgraph(subgraphSet)
        subgraphSolution = baker(subgraph,k)
        solutions= solutions.union(set(subgraphSolution))
    print("initial solution set size:",len(solutions))

    num_collisions = 0
    collidingVertices = set()

    

    solutionsSubgraph = G.subgraph(solutions)
    #build a dict
    #have lower numbered vertice in edge be key, val be set of 
    #destinations of edges
    #remove as we go through priority map until survivors are empty
    

    def setOutEdges(edgesObject):
        myList = []
        for edge in edgesObject:
            a,b = edge
            myList += [b]
        return(set(myList))
    priorityMap = {vertex:set([item[1] for item in solutionsSubgraph.edges(vertex)]) for vertex in solutions}
    #removal: O(V) * O(V) (check and remove on each vertex is O(1))
    priorityMap = dict(sorted(priorityMap.items(), key=lambda item: len(item[1]), reverse = True))
    #print("priority map sizes:",{key:len(val) for key,val in priorityMap.items()})
    
    itemsList = [item for item in priorityMap.items()]
    
    i = 0 #current index of item with most connections.
    while (itemsList):
        vertice,connections = itemsList[i]#(O(V))
        for connection in connections: #(O(root(V)))
            if vertice in priorityMap[connection]:#O(1)
                priorityMap[connection].remove(vertice) #O(1)
        del priorityMap[vertice]
        del itemsList[i]
        
        isDone = True
        for vertice in priorityMap: #O(V). 
            if not len(priorityMap[vertice]) == 0: #O(1), set tracks its own size
                isDone = False #if any border edges remain, keep trimming.
                break
        if isDone:
            answer = set(list(priorityMap.keys()))
            print("number of survivors: ",len(answer))
            total_time = time.time() - start_time
            print("ind set pruning took:",total_time)
            return answer
        i = itemsList.index(max(itemsList,key=lambda item: len(item[1])))
#change code to allow unlimited k in bakers?

    




#testSubgraph = testSubgraph.subgraph(max(nx.connected_components(testSubgraph),key=len)) #biggest chunk of subgraph... gonna have to do this for every subgraph! We can say for chunks of 2 or less we pick at random otherwise do baker on each chunk. We should probably modify our baker to handle chunks

#nx.draw(testSubgraph)
#plt.show()

def baker(planarGraph,k,root=False): #k = 1/epsilon. returns list
    vertices = list(planarGraph)
    if root:
        #print("num vertices:",len(vertices))
        pass
    if len(vertices) == 1:#single vertice
        return [vertices[0]]#return it
    
    myChunks = nx.connected_components(planarGraph)
    
    #make it recursive
    #base case: single chunk
    #if single vertice, return single vertice instead of running algo
    #test if works on double vertice (should)
    listedChunks = []
    for item in myChunks:
        listedChunks += [item] #for some reason list constructor no work right?
    if len(listedChunks) > 1: #multiple disjoint chunks
        if root:
            #print("disjoint chunks:",len(listedChunks))
            pass
        combined_solutions = []
        for chunk in listedChunks:
            chunkGraph = planarGraph.subgraph(chunk)
            combined_solutions += baker(chunkGraph,k)
        if root:
            #print("length of solution:",len(combined_solutions))
            #average_degree = planarGraph.number_of_edges()/len(vertices)
            #expected_size = len(vertices)/average_degree
            #built_in_estimation = nx.approximation.maximum_independent_set(planarGraph)
            #print("built in guess size:",len(built_in_estimation))
            #print("expected solution size:",expected_size)
            pass
        return combined_solutions
        
    """
    print("my chunks:",[[item] for item in myChunks])
    print("num chunks:",len([[item] for item in myChunks]))
    input("pause")
    """

    levels = {} #distances to root
    root = random.choice(vertices)
    levels[root] = 0
    
    bfsQueue = [root]
    
    while bfsQueue:
        current_vertex = bfsQueue.pop(0)
        for neighbor in planarGraph[current_vertex]:
            if neighbor not in levels:
                levels[neighbor] = (levels[current_vertex] + 1)
                bfsQueue.append(neighbor)
    maxLevel = max(levels.items(), key=lambda x: x[1])[1]
    maxLevel = min(k,maxLevel)
    possibleSolutions = []
    for l in range(0,maxLevel):
        #print("l:",l)
        #print("levels:",levels)
        #print("vertices:",vertices)
        GiVertices = [vertice for vertice in vertices if levels[vertice]%k == l] #O(log(V))*O(V) 
        Gi = planarGraph.subgraph(set(vertices) - set(GiVertices)) #no we delete these
        
        connectedComponents = nx.connected_components(Gi)
        solutionForGi = []
        for connectedComponent in connectedComponents:
            component = Gi.subgraph(connectedComponent)
            maximalSet = nx.maximal_independent_set(component)
            solutionForGi += maximalSet
        possibleSolutions += [solutionForGi]
    bestSolution = max(possibleSolutions,key= lambda x:len(x))
    #print("maxLevel:",maxLevel)

    #now get connected components on every even numbered level
    
    return bestSolution
#print("baker output:",baker(testSubgraph, 10, root=True))


#__________________________
testSize = 512
G = generateGraph(testSize)
expectedSize = testSize/(testSize**(1/2))
print("expectedSize:",expectedSize)
sgs = getPlanarSubgraphs(G,verbose = False)
#testSubgraph = G.subgraph(sgs[0])
getIndSet(G,sgs)
#__________________________


def runExperiment(testSize,numTrials,k):
    #expectedSize = testSize/(testSize**(1/2))
    mySum = 0
    total_expected_sizes = 0
    for i in range(0,numTrials):
        G = generateGraph(testSize)
        vertices = list(G)
        averageDegree = sum([G.degree[vertice] for vertice in vertices])/len(vertices)
        expectedSize = testSize/averageDegree
        #print("expected Size:",expectedSize)
        sgs = getPlanarSubgraphs(G,verbose = False)

        resultSet = getIndSet(G,sgs,k)
        mySum += len(resultSet)
        total_expected_sizes += expectedSize
        #testSubgraph = G.subgraph(sgs[0])
        if i == 0:

            #ts1 = G.subgraph(sgs[0])
            #nx.draw(ts1)
            #plt.show()

            #testSubgraph = G.subgraph(resultSet)
            #nx.draw(G)
            #plt.show()
            #nx.draw(testSubgraph) #woah it actually works I'm on to something!!!
            #plt.show()
            pass
    print("estimated average score:",mySum/(total_expected_sizes)) #assumes V/(averate degree) is expected size
    #source: https://www.math.cmu.edu/~af1p/Texfiles/indgnp.pdf
    return mySum/numTrials, mySum/(total_expected_sizes)
#runExperiment(512,25,20)
testList = [64,128,256,512,676,841]
writer = open("scores.csv","w")
writer2 = open("sizes.csv","w")
from math import log
for testSize in testList:
    #to make it fair, scale k with log
    base = testSize**(1/2)
    k = int(testSize/3) #arbitrarily scale accuracy bounds?
    print("k val for size:",testSize," is: ",k)
    average_size,average_score = runExperiment(testSize,25,k) #arbitrarily, k = 20
    writer.write(str(testSize))
    writer.write(",")
    writer.write(str(average_score))
    writer.write(",")
    writer.write(str(k))
    writer.write("\n")
    
    writer2.write(str(testSize))
    writer2.write(",")
    writer2.write(str(average_size))
    writer2.write(",")
    writer2.write(str(k))
    writer2.write("\n")
    
writer.close()
#experiment procedure below
"""
writer = open("output_data.csv","w")
edgesWriter = open("num_border_edges.csv","w")
import time
numSubgraphs = []
times = []
for size in testSizes:
    print("doing size: ",size)
    sublist = []
    #edgelist = []
    st = time.time()
    for i in range(0,10): #6 trials
        G = generateGraph(size)
        sgs = getPlanarSubgraphs(G,verbose = False)
        num_edges = len(getBorderEdges(sgs,G))
        #edgelist += [num_edges]
        count = len(sgs)
        sublist += [count]
        writer.write(str(count))
        writer.write(",")
        
        edgesWriter.write(str(num_edges))
        edgesWriter.write(",")
    writer.write("\n")
    edgesWriter.write("\n")
    
    averageNumSubgraphs = sum(sublist)/len(sublist) #get the average number
    numSubgraphs += [averageNumSubgraphs]
    print("done size: ",size)
    print("result for size: ",size," is ",averageNumSubgraphs)
    elapsedTime = time.time() - st
    print("took: ",elapsedTime," second")
    times += [elapsedTime]
writer.close()
edgesWriter.close()
    
print("test sizes:",testSizes)
print("num Subgraphs:",numSubgraphs)
print("elapsed times:",times)
"""
#OKAY the number of subgraphs appears to increase sublinearly with number of nodes
#but is this still true when we apply same procedure to remaining nodes? Does this require O(root(n)) average degree?
#We know that 

#K33 and K5 have sizes 6 and 5 respectively, so in theory any original graph producing less than that number of nodes works fine...
#what happens if we do start with a fully connected graph (as most resultant graphs will probably be) though?















#actually seems to be pretty good odds of a small-ish number of subgraphs depending on the conditions? needs more 
#experimentation. Also, number of islands seems to increase sublinearly with number of nodes? This is strange

#First, render. Then, get resultant graph


#next step: render subgraphs and connections then rewrite code to run multiple trials

#rendering and resultant graphs and also space reduction


#ON THE REPORT: FIRST, WE EXPERIMENT TO FIND INTUITION TO HELP OUR ANALYSIS
#ALSO: FIND CONDITIONS ON THE SUBGRAPHS

#chatgpt rendering code:_____________________________
def drawResults(subgraphs,G):

    num_subgraphs = len(subgraphs)
    colors = plt.cm.rainbow  # Choose a colormap
    node_color_map = {}
    for i, subgraph in enumerate(subgraphs):
        color = colors(i / num_subgraphs)  # Use colormap to generate colors
        for node in subgraph:
            node_color_map[node] = color

    # Create a color map for the edges
    default_edge_color = 'gray'  # Set a default color for edges not belonging to any subgraph
    edge_color_map = {}
    for edge in G.edges():
        node1, node2 = edge
        color = default_edge_color  # Default color
        for subgraph in subgraphs:
            if node1 in subgraph and node2 in subgraph:
                color = node_color_map[node1]
                break
        edge_color_map[edge] = color

    # Plot the original graph with different colors for each subgraph
    pos = nx.spring_layout(G)  # You can choose any layout you prefer
    nx.draw(G, pos, node_color=[node_color_map[node] for node in G.nodes()],
            edge_color=[edge_color_map[edge] for edge in G.edges()], with_labels=True, font_weight='bold')
    plt.show()


    #____________________________________________
    G_new = nx.Graph()

    # Add vertices to the new graph representing each subgraph
    for i in range(0,num_subgraphs):
        G_new.add_node(i, color=node_color_map[subgraphs[i][0]])  # Assign color as the nodes in the subgraph

    # Check for adjacency between subgraphs and add edges accordingly
    for i in range(0,num_subgraphs):
        for j in range(i + 1, num_subgraphs): #O(num_subgraphs**2)
            for node1 in subgraphs[i]: #O(n/num_subgraphs)
                for node2 in subgraphs[j]:#O(n/num_subgraphs)
                    if G.has_edge(node1,node2):#O(1) assuming networx implements this adjacency table with hashing
                        edge = (i,j)
                        G_new.add_edge(*edge)
    #O(num_subgraphs**2*n**2/(num_subgraphs**2)) = O(n**2) < O(n**3) for basic proccess = don't care

    # Plot the new graph
    node_colors = [G_new.nodes[i]['color'] for i in G_new.nodes]
    nx.draw(G_new, pos, node_color=node_colors, with_labels=True, font_weight='bold')
    plt.show()
    #_____________________________

#drawResults(sgs,G)


#mention karger as inspiration
#discuss success boosting as a possible method
#theory the reason it works so well is as the number of border edges shrinks, the more likely that the independent subsets interact each other, leaving us with a huge chunk of a lot of them
#found set size/expected set size appears to be a monotonically increasing function of input size so that should mean its a ptas. If I did it right.