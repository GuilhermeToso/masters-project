# Master's Project
--------------------
This repository contains the files of my master's research project in computational intelligence. 

The project is a research on the synchronization analysis of biological neuron models. The ideia is to analyze the possibility of trajectories segmentation by synchronizing a group of neurons with strong coupling force between each one and the dessynchronization of those by a weak coupling force. Finally, a proof of concept is created as a machine learning model of semi-supervised learning, thus representing samples with the same classification by a cluster of synchronized neurons, and the samples with different labels are dessynchronized.

<h2> Project Tree </h2>
<pre>
<code>
    masters-project 
    |
    |---- data
    |---- imgs
    |---- nsc
    |---- tests
</code>
</pre>
---------------------------
<h2>Example of Iris Dataset Classification</h2>

<p>The first figure represents the decision limits generated by the adjusted model, while the second image shows the action potential of the neurons of the 150 samples, and we can see the clustering phenomenon by segmenting the trajectories in time in the third image, which represents the times when neural firing occurs.</p>
<img src="imgs/contour.png" width=70% height=70%>
<img src="imgs/trajectories.png" width=70% height=70%>
<img src="imgs/activity.png" width=70% height=70%>