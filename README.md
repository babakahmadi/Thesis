Difference between supervised and unsupervised:
----

* **supervised**:
	- **evaluation:** accuracy, precision, recall
* **unsupervised**:
	- **evaluation:** how? - clusters are in the eye of beholder :-)

**Cluster Analysis:** Finding groups of objects such that the objects in a group will be similar to one another and different from the objects in other groups.

**Measures of cluster validity:(numerical)** --> The validation of clustering structures is the most difficult and frustrating part of cluster analysis

* **external index:**
	- entropy
* **internal index:**
	- Sum of Squared Error(SSE)
* **relative index:** used to compare two clusterings
	- SSE or entropy
	- cohesion and separation

**Evaluation Metrics for clustering:( [reference](#) )**

 * **Conductance: [reference](#)** 
 * **Coverage: [reference](#) **
 * **Modularity: [reference](#) **
 * **Performance: [reference](#) **
 * **Silhouette index : [reference](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)**

**Validation:**

 * internal measures: cohesion and separation <br>
 * external measures: entropy and purity <br>

**Algorithms:**

 * [comparing different algorithms](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#example-cluster-plot-cluster-comparison-py)
 * [Good roadmap](http://papers.nips.cc/paper/2388-learning-spectral-clustering.pdf)


**some references:**

 * cluster validation: [reference](http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf)  --> idea: *using binary search* <br>
 * cluster validation(statistical approach) : [reference](http://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf) <br>


**good Tutorials:**

 * good review: [this](http://www.cs.kent.edu/~jin/DM08/cluster.pdf)
 * validation methods: [this](http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf)
 * ...



**some implementation:**
 
 * dbscan visualization : [reference](http://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
 * dbscan code: [code](http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html)
 * MDS(MultiDimensional Scaling): [code](http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html)


**some packages:**

 * R package: [this](https://cran.r-project.org/web/packages/clValid/vignettes/clValid.pdf)

**Dataset:**

 * stanford dataset: [this](http://snap.stanford.edu/data/)
 * list of repositories: [reference](http://www.datasciencecentral.com/profiles/blogs/top-20-open-data-sources)


**for documentation:**

 * introduction to cluster analysis: [reference](http://www-users.cs.umn.edu/~han/dmclass/cluster_survey_10_02_00.pdf)
 * 