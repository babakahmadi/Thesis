####Difference between supervised and unsupervised:

* **supervised**:numerical
	- **evaluation:** accuracy, precision, recall
* **unsupervised**:
	- **evaluation:** how? - clusters are in the eye of beholder :-)

**Cluster Analysis:** Finding groups of objects such that the objects in a group will be similar to one another and different from the objects in other groups.

**Measures of cluster validity:()**

* **external index:**
	- entrop
    - y
* **internal index:**
	- Sum of Squared Error(SSE)
* **relative index:** used to compare two clusterings
	- SSE or entropy
	- cohesion and separation

> __The validation of clustering structures is the most difficult and frustrating part of cluster analysis__


#### **Evaluation Metrics for clustering:( [reference](#) )**

 * **Conductance: [reference][1]** 
 * **Coverage: [reference][2]**
 * **Modularity: [reference][3]**
 * **Performance: [reference][4]**
 * **Silhouette index : [reference](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)**
[1]: #
[2]: #
[3]: #
[4]: #

#### **Validation:**

 * internal measures: cohesion and separation <br>
 * external measures: entropy and purity <br>

#### **Algorithms:**

 * [comparing different algorithms](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#example-cluster-plot-cluster-comparison-py)
 * [Good roadmap](http://www.eecs.berkeley.edu/Pubs/TechRpts/2003/CSD-03-1249.pdf), [Completed Paper](http://www.di.ens.fr/~fbach/jmlrfinal_bach06b.pdf)
 * SOM (Self-organizing Map)


#### **some references:**

 * cluster validation: [reference](http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf)  --> idea: *using binary search* 
 * cluster validation(statistical approach) : [reference](http://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf)
 * for proofs: [reference](http://www.math.iastate.edu/thesisarchive/PhD/LiuSijiaPhDSS11.pdf)  --> chapter 3 
 * solve two weaknesses of spectral clustering: [reference](https://papers.nips.cc/paper/2766-fixing-two-weaknesses-of-the-spectral-method.pdf) --> method instead of k-means
 * diffusion maps, spectral clustering and eigenfunctions of fokker-planck operators: [reference](http://papers.nips.cc/paper/2942-diffusion-maps-spectral-clustering-and-eigenfunctions-of-fokker-planck-operators.pdf) --> euclidean distance in new representation has a meaningful description
 * consistency of spectral clustering: [annals of statistics 2008](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/pdfs/pdf3199.pdf)
 * limits of spectral clustering: [reference](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/pdfs/pdf2775.pdf)
 * random walk survey: [reference](http://www.cs.elte.hu/~lovasz/erdos.pdf)
 * kdd bipartite spectral: [reference](http://www.cs.utexas.edu/users/inderjit/public_papers/kdd_bipartite.pdf)
 * ........
 * co-training spectral: [ICML2011](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Kumar_272.pdf)
 * mining clustering dimensions: [ICML2010](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_DasguptaN10.pdf)
 * Large-Scale Multi-View Spectral Clustering via Bipartite Graph : [AAAI2015](http://www.contrib.andrew.cmu.edu/~uqxchan1/papers/Yeqing_Li_AAAI2015.pdf)
 * Incremental Spectral Clustering with the Normalised Laplacian: [NIPS2011](https://hal.inria.fr/hal-00745666/document)



#### **good Tutorials:**

 * good review: [this](http://www.cs.kent.edu/~jin/DM08/cluster.pdf)
 * validation methods: [this](http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf)
 * kernel and spectral: [this](http://people.bu.edu/bkulis/pubs/spectral_techreport.pdf)
 * Spectral Clustering with Purturbed Data: [nips 2009](http://papers.nips.cc/paper/3480-spectral-clustering-with-perturbed-data.pdf)
 * Spectral graph theory: [chung 97](http://www.meduniwien.ac.at/user/yll.haxhimusa/prip/praktikas/spectral_pyramid_praktikum/spectral_graph_chapter.pdf)
 * ...



#### **some implementation:**
 
 * dbscan visualization : [reference](http://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
 * dbscan code: [code](http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html)
 * MDS(MultiDimensional Scaling): [code](http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html)
 * numpy: [tutorial](http://www.scipy-lectures.org/intro/numpy/numpy.html)


#### __some packages:__

 * R package: [this](https://cran.r-project.org/web/packages/clValid/vignettes/clValid.pdf)

#### __Dataset:__

 * stanford dataset: [this](http://snap.stanford.edu/data/)
 * list of repositories: [reference](http://www.datasciencecentral.com/profiles/blogs/top-20-open-data-sources)
 * wine: [link](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)
 * Dermatology: [link](http://archive.ics.uci.edu/ml/datasets/Dermatology?ref=datanews.io)
 * Letter Recognition: [link](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
 * handWritten digits: [link](http://yann.lecun.com/exdb/mnist/)
 * UCI:
     - fisher iris
     - wine
     - breast cancer winsconsin
     - heart
     - handwritten digit
     - 

#### __bipartite graph__:
 * Solving Cluster Ensemble Problems by Bipartite Graph Partitioning: [link](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2004_FernB04.pdf)

#### __Constructing Similarity Graph:__
 * local clustering on multiple manifold: [link](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tnn11smmc.pdf)
     -  Train M d-dimensional local linear manifolds by using MPPCA to approximate the underlying manifolds
     -  Determine the local tangent space of each point ( using EM )
     -  Compute pairwise affinity between two local tangent spaces using
     -  p_ij = (max dot(u_1, v_1) )x ... x(max dot(u_k,v_k) ) --> u_i in tangent space
     -  q_ij = if in KNN 1, 0 o.w
     -  w_ij = p_ij q_ij
 * based on random walk: [link](http://ftp.cs.rochester.edu/u/qyou/papers/qzy_FCSC.pdf)
 * based on neighbor propagation: [link](http://ac.els-cdn.com/S0925231212005000/1-s2.0-S0925231212005000-main.pdf?_tid=34cf8d5c-007a-11e6-9438-00000aacb362&acdnat=1460443735_c50910ef570b53dee8a5de41d6da18c6)
     - construct similarity graph and neighborhood graph, propagate it
 * based on newton equations: [link](http://cs.uoi.gr/~lagaris/papers/Newton2.pdf)
     - sparsify the affinity matrix using second newton equation
 * Local density adaptive: [link](http://www.sciencedirect.com/science/article/pii/S0167865510003181?np=y)
 * density sensitive: [link](http://www.sciencedirect.com/science/article/pii/S0950705111000219)

#### **for feature selection:**
 * Laplacian Score: [nips 2005](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2005_149.pdf)
 * Spectral Feature Selection:[2007](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_ZhaoL07.pdf)
 * Efficient Spectral Feature Selection with minimum dependency: [2010](http://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-123.pdf)
 * Semi-supervised Feature Selection via Spectral Analysis: [2007](http://www.public.asu.edu/~huanliu/papers/ssfs.pdf)


#### **for documentation:**

 * introduction to cluster analysis: [reference](http://www-users.cs.umn.edu/~han/dmclass/cluster_survey_10_02_00.pdf)
 * Laplacian properties: [reference](http://www.sfu.ca/~mohar/Papers/Spec.pdf), - , [reference](http://www.fmf.uni-lj.si/~mohar/Papers/Montreal.pdf)


#### **base papers:**
 * [SCORE](http://www.stat.cmu.edu/tr/SCORE.pdf)
 * [On Spectral Clustering](http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf)
 * [Learnng Spectral Clustering](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2003_AA39.pdf)
 * [Multiple non-redundant spectral clustering views](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NiuDJ10.pdf)

---

#### **to read:**
 * **for Kernel:** [  Bernhard Scholkopf](http://dip.sun.ac.za/~hanno/tw796/lesings/mlss06au_scholkopf_lk.pdf)
 * __good papers:__ [University of Washington](http://www.stat.washington.edu/spectral/)
 * __regularized spectral clustering:__[jmlr](http://www.stat.washington.edu/mmp/Papers/jmlr-reg-super-learning-revised.pdf)
 * __model based clustering(select number of clusters):__ [University of Mishigan](http://www.dtic.mil/cgi-bin/GetTRDoc?AD=ADA458798)
 * __statistical view of marginal spectral clustering:__ [Jordan](http://www.cs.berkeley.edu/~jordan/papers/zhang-jordan-statsci08.pdf)
 * __spectral dimension reduction:__ [Jordan 2011](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_NiuDJ11.pdf)



