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

<p>The <b>data</b> folder contain the raw and cleaned datasets used to test the semi-supervised learning model, as much as the scores obtained by testing the model with different parameters combinations. <br> 
The <b>imgs</b> folder contain the three images used in the <i>README.md</i> file. The <b>nsc</b> package, which is an abbreviation for Neuron Synchronization Competition, contain the python files for the model of semi-supervised learning (<i>neurongraph.py</i>), while it contain others files that calculates the phases of the trajectories to detect the synchronization (<i>unwrap.py</i>), the file to plot some specific kinds of graphics (<i>ngplot.py</i>), different models of self-training (<i>standard_self_training.py</i>) and tri-training (<i>tri_training.py</i>), etc. <br>
Also there is the neuron subdirectory that contains the biological neuron models defined below:
</p>
<ul>
    <li>Hodgkin-Huxley</li>
    <li>Hindmarsh-Rose</li>
    <li>Integrate-and-Fire</li>
    <li>Rulkov</li>
    <li>Izhikevich</li>
    <li>Courbage-Nekorkin-Vdovin</li>
</ul>

Finally, the <b>tests</b> folder contain all the steps made in the project and the figures obtained as the results. Some subdirectories, specifically the last one, must be updated, which will be in a recent future.

At this last folder (<b>11 - NSC Data Tests</b>), from the <b>tests</b> directory, there is a folder called <b>Notebooks</b> which contains the jupyter notebooks of the exploratory data analysis and cleaning processes of the datasets used for machine learning models comparison. 

<hr>
<h2>Model implementation</h2>

The implementation of the NSC model is quite simple. Down below there is an example of the implementation, where you import the <b>NSC</b> class, then you instantiate it with some values as the <b>data</b> to be inserted (must be a Pandas DataFrame), the <b>target</b> name, <b>similarity</b> used (like euclidean distance), <b>model</b> to indicate the neuron model, <b>alpha</b>, <b>beta</b> and <b>gamma</b> are parameters used to adapt the coupling force. The <b>w_step</b> means the maximum/minimum value of the coupling force adaptation, <b>neighbors</b> are the maximum amountof connections a neuron can make, <b>search_expand</b> represents the speed at which the searchs' hypersphere grows. Finally, <b>print_info</b> and <b>print_steps</b>, prints the classified data and all the steps of the agorithm, respectively. 

<pre>
    <code style="color:pink;">
    from nsc import NSC
    import pandas as pd

    iris = datasets.load_iris(as_frame=True).frame
    model = 'CNV' # CNV stands for Courbage-Nekorkin-Vdovin 

    nc = NeuronGraph(data=data, target='target' ,similarity='Euclidean', model=model, alpha = 0.1, w_step = 0.2, time_step=1, print_info=False, print_steps=False, beta=2.0, gamma=1.5)
    nc.neighbors = 15
    nc.search_expand = 100
    nc.preprocess_data(shuffle=False,not_null=10,standarlize=False)
    </code>
</pre>


<h2>Example of Iris Dataset Classification</h2>

<p>The first figure represents the decision limits generated by the adjusted model, while the second image shows the action potential of the neurons of the 150 samples, and we can see the clustering phenomenon by segmenting the trajectories in time in the third image, which represents the times when neural firing occurs.</p>
<img src="imgs/contour.png" width=70% height=70%>
<img src="imgs/trajectories.png" width=70% height=70%>
<img src="imgs/activity.png" width=70% height=70%>

<h2>Bibliography</h2>

<cite>Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of physiology, 117(4), 500-544.</cite>
<br>

<cite>Hindmarsh, J. L. and Rose, R. (1984). A model of neuronal bursting using three coupled
first order differential equations. Proceedings of the Royal society of London. Series B.
Biological sciences, 221(1222):87–102.</cite>
<br>

<cite>Lapicque, L. (1907). Recherches quantitatives sur l’excitation electrique des nerfs
traitee comme une polarization. Journal de Physiologie et de Pathologie Generalej,
9:620–635.</cite>
<br>

<cite>Rulkov, N. F. (2002). Modeling of spiking-bursting neural behavior using twodimensional
map. Physical Review E, 65(4):041922.</cite>
<br>

<cite>Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on
neural networks, 14(6):1569–1572.</cite>
<br>

<cite>Courbage, M., Nekorkin, V., and Vdovin, L. (2007). Chaotic oscillations in a map-based
model of neural activity. Chaos: An Interdisciplinary Journal of Nonlinear Science,
17(4):043109.</cite>