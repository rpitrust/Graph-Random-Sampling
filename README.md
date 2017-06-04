# RPI-Masters-Project Edited by Rongrong Miao

This branch is a modified version of Nicholas Fay's master project.
Changes in graphics:
- Provide a curve fit and scatter diagram and residual value for in/out degree distribtuion for original graph,
sample graph, and test data, with sample being 85% of a sample drawn using DURW and test data
being the rest 15%. 

- Requirement and usage: The program runs with python3
1. Need a local directory called testGraphs with a txt file containing all edges. E.g. wikiTalk.txt as provided
2. Need a local directory called FittedGraph to store the output graphs 
3. Need a local directory called stats to store the NMSE graph

Sample command line args
bash $ python3 main.py -d ./testGraphs -it 8000 -deg True

Explaination: 
-d Name of directory where input resides
-it Number of node to sample. Default = 20000
-deg Whether to use default polynomial degree to fit (False) or run the try fit algorithm
to find the best fitting degree based on number of nodes to sampled and other conditions (True).
Default is False

For more args, use -h to check out.

