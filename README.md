<b>Difference between supervised and unsupervised:</b>

* **supervised**:
	- **evaluation:** accuracy, precision, recall
* **unsupervised**:
	- **evaluation:** how? - clusters are in the eye of beholder :-)

**Cluster Analysis:** Finding groups of objects such that the objects in a group will be similar to one another and different from the objects in other groups.

**Measures of cluster validity:(numerical) --> <span style="color:red">The validation of clustering structures is the most difficult and frustrating part of cluster analysis </span>**

* **external index:**
	- entropy
* **internal index:**
	- Sum of Squared Error(SSE)
* <b>relative index:</b> used to compare two clusterings
	- SSE or entropy
	- cohesion and separation

**Evaluation Metrics for clustering:(<a href="">reference</a>)**

 * **Conductance: <a href="">reference</a>** 
 * **Coverage: <a href="">reference</a> **
 * **Modularity: <a href="">reference</a> **
 * **Performance: <a href="">reference</a> **
 * **Silhouette index : <a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html">reference</a> **

<b>Validation</b>


<b>Algorithms</b>

<a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#example-cluster-plot-cluster-comparison-py">comparing different algorithms</a>

<a href="http://papers.nips.cc/paper/2388-learning-spectral-clustering.pdf">Good roadmap</a> <br><br>
cluster validation: <a href="http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf">this</a>  --> idea: <u>using binary search</u> <br>
cluster validation(statistical approach) : <a href="http://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf">this</a> <br>
R package: <a href="https://cran.r-project.org/web/packages/clValid/vignettes/clValid.pdf">this</a> <br>

dbscan visualization : <a href="http://www.naftaliharris.com/blog/visualizing-dbscan-clustering/">this</a> <br>
good review: <a href="http://www.cs.kent.edu/~jin/DM08/cluster.pdf">this</a> <br>
validation methods: <a href="http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf">this</a><br>

internal measures: cohesion and separation <br>
external measures: entropy and purity <br>


stanford dataset: <a href="http://snap.stanford.edu/data/">this</a><br>


