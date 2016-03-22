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

**Evaluation Metrics for clustering:(<a href="#">reference</a>)**

 * **Conductance: <a href="#">reference</a>** 
 * **Coverage: <a href="#">reference</a> **
 * **Modularity: <a href="#">reference</a> **
 * **Performance: <a href="#">reference</a> **
 * **Silhouette index : <a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html">reference</a> **

<b>Validation</b>

 * internal measures: cohesion and separation <br>
 * external measures: entropy and purity <br>

<b>Algorithms:</b>

 * <a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#example-cluster-plot-cluster-comparison-py">comparing different algorithms</a>
 * <a href="http://papers.nips.cc/paper/2388-learning-spectral-clustering.pdf">Good roadmap</a> <br><br>

** some references: **

 * cluster validation: <a href="http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf">reference</a>  --> idea: <u>using binary search</u> <br>
 * cluster validation(statistical approach) : <a href="http://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf">reference</a> <br>


** good Tutorials: **

 * good review: <a href="http://www.cs.kent.edu/~jin/DM08/cluster.pdf">this</a> <br>
 * validation methods: <a href="http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf">this</a><br>
 * ...



** some implementation: **
 
 * dbscan visualization : <a href="http://www.naftaliharris.com/blog/visualizing-dbscan-clustering/">reference</a> <br>
 * dbscan code: <a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html">code</a>
 * MDS(MultiDimensional Scaling): <a href="http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html">code</a>


** some packages: **

 * R package: <a href="https://cran.r-project.org/web/packages/clValid/vignettes/clValid.pdf">this</a> <br>

** Dataset: **

 * stanford dataset: <a href="http://snap.stanford.edu/data/">this</a><br>
 * list of repositories: <a href="http://www.datasciencecentral.com/profiles/blogs/top-20-open-data-sources">reference </a>


** for documentation: **

 * introduction to cluster analysis: <a href="http://www-users.cs.umn.edu/~han/dmclass/cluster_survey_10_02_00.pdf">reference</a><br>