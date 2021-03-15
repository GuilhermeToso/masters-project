# -*- coding: utf-8 -*-
"""
===========================================
Access to Biological Neuron Models
===========================================

================================================================================
Author: Guilherme M. Toso
Title: __init__.py
Project: Semi-Supervised Learning Using Competition for Neurons' Synchronization
================================================================================

=============================================================

This modules access the classes of the biological neuron models.
In this project, only the first eight models were analysed.

=============================================================

==========================================
Models:

    (1) Hodgkin-Huxley
    (2) Hindmarsh-Rose
    (3) Integrate-And-Fire
    (4) Spike-Response-Model
    (5) Aihara
    (6) Rulkov
    (7) Izhikevic
    (8) Courbage-Nekorkin-Vdovin
    (9) Fitzhugh-Nagumo
==========================================

==========================================
Bibliography:

(1) Hodgkin, A. L. and Huxley, A. F. (1952). A quantitative description of membrane current
    and its application to conduction and excitation in nerve. The Journal of physiology,
    117(4):500–544.

(2) Hindmarsh, J. L. and Rose, R. (1984). A model of neuronal bursting using three coupled
    first order differential equations. Proceedings of the Royal society of London. Series B.
    Biological sciences, 221(1222):87–102.

(3) Lapicque, L. (1907). Recherches quantitatives sur l’excitation electrique des nerfs
    traitee comme une polarization. Journal de Physiologie et de Pathologie Generalej,
    9:620–635.

(4) Gerstner, W. (2001). A framework for spiking neuron models: The spike response model.
    In Handbook of Biological Physics, volume 4, pages 469–516. Elsevier.

(5) Aihara, K., Takabe, T., and Toyoda, M. (1990). Chaotic neural networks. Physics letters
    A, 144(6-7):333–340.

(6) Rulkov, N. F. (2002). Modeling of spiking-bursting neural behavior using twodimensional
    map. Physical Review E, 65(4):041922.

(7) Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on
    neural networks, 14(6):1569–1572.

(8) Courbage, M., Nekorkin, V., and Vdovin, L. (2007). Chaotic oscillations in a map-based
    model of neural activity. Chaos: An Interdisciplinary Journal of Nonlinear Science,
    17(4):043109.

(9) FitzHugh R. (1955) "Mathematical models of threshold phenomena in the nerve membrane". Bull. Math. Biophysics, 17:257—278
    FitzHugh R. (1961) "Impulses and physiological states in theoretical models of nerve membrane". Biophysical J. 1:445–466
    FitzHugh R. (1969) "Mathematical models of excitation and propagation in nerve". Chapter 1 (pp. 1–85 in H. P. Schwan, ed. Biological Engineering, McGraw–Hill Book Co., N.Y.)
    Nagumo J., Arimoto S., and Yoshizawa S. (1962) "An active pulse transmission line simulating nerve axon". Proc. IRE. 50:2061–2070.

"""

from .hindmarshrose import HindmarshRose
from .integrateandfire import IntegrateAndFire
from .aihara import Aihara
from .rulkov import Rulkov
from .izhikevic import Izhikevic
from .cnv import CNV
from .hodgkinhuxley import HodgkinHuxley, Chemical, SDE
from .core import *
from .glm import Kernel, GLM


__all__ = ['HindmarshRose','IntegrateAndFire','Aihara','Rulkov','Izhikevic','CNV',
            'HodgkinHuxley','Chemical','SDE','negative_refractory','iaf_refractory',
            'threshold_refractory','mean_lifetime','delay_response','normal_delay_response', 'gamma_response',\
            'moto_response', 'alpha_response','linear','senoidal','exponential','rate_constant',
            'fire_probability','survivor','interval_dist','Kernel','GLM']