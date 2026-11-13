# Bibliography {#bibliography}

The sources the library implements. Each entry has an anchor named by first author and year, and
a comment cites it as `\ref Runnalls2007 "Runnalls (2007)"`. The library has nothing of its own
to cite: every algorithm is one published here, and the comment that cites an entry says what
the code takes from it, such as the equation numbers or the section, and where it departs.

An algorithm has two citations, often the same entry. The source that introduced it gets the
credit. The source the code was implemented from is one that was read, in full, and every
equation, section, or algorithm number in a comment is taken from that source. When the original
was not read, the comment cites it for credit and names the source that was:
`\ref Rauch1965 "Rauch, Tung, and Striebel (1965)", as given in \ref Sarkka2013 "Särkkä (2013)", §8.2`.
The source read may be a free description, such as a Wikipedia section, linked inline. The
entry for a source not read says so in a final sentence, **Not read.**, and names the sources
that describe its method; when the original is read, check the code against it and delete the
sentence. Add an entry when a comment first cites it. Foundational concepts link to Wikipedia
inline, not through this page.

Entries are in alphabetical order by first author.

\anchor Alspach1972
D. L. Alspach and H. W. Sorenson. Nonlinear Bayesian estimation using %Gaussian sum
approximations. *IEEE Transactions on Automatic Control*, 17(4):439–448, 1972.
**Not read.** \ref Bergman1999 "Bergman (1999)", §3.5.3, sketches the method; no free source read
describes it fully.

\anchor Arasaratnam2010
I. Arasaratnam, S. Haykin, and T. R. Hurd. Cubature Kalman filtering for continuous-discrete
systems: theory and simulations. *IEEE Transactions on Signal Processing*, 58(10):4977–4993,
2010. §IV-A states the third-degree spherical-radial cubature rule of the cubature Kalman filter
(Arasaratnam and Haykin, *IEEE Transactions on Automatic Control*, 54(6):1254–1269, 2009).
Free: [PDF, author's copy](https://ms.mcmaster.ca/tom/ArasHayKHurd2010.pdf).

\anchor Arthur2007
D. Arthur and S. Vassilvitskii. k-means++: the advantages of careful seeding. In *Proceedings
of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, pages 1027–1035,
2007. Free: [PDF](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf).

\anchor Arulampalam2002
M. S. Arulampalam, S. Maskell, N. Gordon, and T. Clapp. A tutorial on particle filters for online
nonlinear/non-%Gaussian Bayesian tracking. *IEEE Transactions on Signal Processing*,
50(2):174–188, 2002. Free: [PDF, course copy](https://www.csd.uoc.gr/~hy475/resources/particleFilterTutorial.pdf).

\anchor Athans1977
M. Athans, D. Castañon, K.-P. Dunn, C. S. Greene, W. H. Lee, N. R. Sandell, Jr., and
A. S. Willsky. The stochastic control of the F-8C aircraft using a multiple model adaptive control
(MMAC) method—Part I: Equilibrium flight. *IEEE Transactions on Automatic Control*,
22(5):768–780, 1977. Free: [PDF, scan](https://willsky.lids.mit.edu/publ_pdfs/19_pub_IEEE.pdf).

\anchor Barber2006
D. Barber. Expectation correction for smoothed inference in switching linear dynamical systems.
*Journal of Machine Learning Research*, 7:2515–2540, 2006. Section 2.2 is the backward pass; (12)
and (13) give the smoothed switch given the next one, and the mean approximation evaluates the
average in (12) at the mean of the smoothed %Gaussian. Free: [JMLR](https://jmlr.org/papers/v7/barber06a.html).

\anchor BarShalom1975
Y. Bar-Shalom and E. Tse. Tracking in a cluttered environment with probabilistic data
association. *Automatica*, 11(5):451–460, 1975.

\anchor BarShalom2001
Y. Bar-Shalom, X. R. Li, and T. Kirubarajan. *Estimation with Applications to Tracking and
Navigation: Theory, Algorithms and Software*. Wiley, 2001.
**Not read.** The comments that cite it take their numbers from \ref Chen2023 "Chen et al. (2023)"
(NIS and NEES), \ref Hashimzade2024 "Hashimzade et al. (2024)" (IMM),
\ref BarShalom1975 "Bar-Shalom and Tse (1975)" (gating), and
[Discretization](https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models).

\anchor Bell1993
B. M. Bell and F. W. Cathey. The iterated Kalman filter update as a Gauss-Newton method. *IEEE
Transactions on Automatic Control*, 38(2):294–297, 1993.
**Not read.** Described in J. Huai and X. Gao, A quick guide for the iterated extended Kalman filter
on manifolds, [arXiv:2307.09237](https://arxiv.org/abs/2307.09237), §II and Algorithm 1.

\anchor Bergman1999
N. Bergman. *Recursive Bayesian Estimation: Navigation and Tracking Applications*. PhD thesis,
Linköping University, Linköping Studies in Science and Technology, Dissertation No. 579, 1999. Free: [PDF](https://rt.isy.liu.se/research/reports/Ph.D.Thesis/PhD579.pdf).

\anchor Bhattacharyya1943
A. Bhattacharyya. On a measure of divergence between two statistical populations defined by their
probability distributions. *Bulletin of the Calcutta Mathematical Society*, 35:99–109, 1943. Free: [scan](https://archive.org/details/dli.calcutta.11603/page/98/mode/2up).

\anchor Bierman1977
G. J. Bierman. *Factorization Methods for Discrete Sequential Estimation*. Academic Press, 1977
(Mathematics in Science and Engineering, volume 128; Dover reprint, 2006).
**Not read.** The measurement update and the elimination of variables are as given in
\ref Ke2024 "Ke et al. (2024)", §II and §III-D; no free source read gives the time update's data
array.

\anchor Blom1988
H. A. P. Blom and Y. Bar-Shalom. The interacting multiple model algorithm for systems with
Markovian switching coefficients. *IEEE Transactions on Automatic Control*, 33(8):780–783, 1988.
**Not read.** Described in \ref Hashimzade2024 "Hashimzade et al. (2024)", §2.2.3.

\anchor Bucy1971
R. S. Bucy and K. D. Senne. Digital synthesis of non-linear filters. *Automatica*, 7(3):287–298,
1971.
**Not read.** Described in \ref Bergman1999 "Bergman (1999)", §2.3 and §5.2.

\anchor Carpenter1999
J. Carpenter, P. Clifford, and P. Fearnhead. Improved particle filter for nonlinear problems.
*IEE Proceedings – Radar, Sonar and Navigation*, 146(1):2–7, 1999.
**Not read.** Described in \ref Douc2005 "Douc et al. (2005)", §2.4, who credit Carpenter et al.;
\ref Arulampalam2002 "Arulampalam et al. (2002)", §V-A, credit \ref Kitagawa1996 "Kitagawa (1996)".

\anchor Chahbazian2024
C. Chahbazian, K. Dahia, N. Merlinge, B. Winter-Bonnet, A. Blanc, and C. Musso. Recursive
posterior Cramér-Rao lower bound on Lie groups. *Automatica*, 160:111422, 2024. Proposition 2
extends the recursion of Tichavský, Muravchik, and Nehorai (1998) to unimodular matrix Lie
groups, and §5, eqs. (68) to (72), gives its matrices for %Gaussian noise entering by the group
exponential. Free: [PDF, HAL](https://hal.science/hal-04612274/document).

\anchor Chen2023
Z. Chen, H. Biggie, N. Ahmed, S. Julier, and C. Heckman. Kalman filter auto-tuning through
enforcing chi-squared normalized error distributions with Bayesian optimization. arXiv:2306.07225,
2023. Free: [arXiv](https://arxiv.org/abs/2306.07225).

\anchor Dempster1977
A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum likelihood from incomplete data via the EM
algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1):1–38, 1977. Free: [PDF, course copy](https://www.ms.uky.edu/~mai/sta705/DLR.pdf).

\anchor Ding2016
P. Ding. On the conditional distribution of the multivariate t distribution. *The American
Statistician*, 70(3):293–295, 2016. Free: [arXiv](https://arxiv.org/abs/1604.00561).

\anchor Douc2005
R. Douc, O. Cappé, and E. Moulines. Comparison of resampling schemes for particle filtering.
arXiv:cs/0507025, 2005. Free: [arXiv](https://arxiv.org/abs/cs/0507025).

\anchor Dowson1982
D. C. Dowson and B. V. Landau. The Fréchet distance between multivariate normal distributions.
*Journal of Multivariate Analysis*, 12(3):450–455, 1982.
**Not read.** Described in
[Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance#As_a_distance_between_probability_distributions_(the_FID_score))
and [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions).

\anchor Doucet2000
A. Doucet, N. de Freitas, K. Murphy, and S. Russell. Rao-Blackwellised particle filtering for
dynamic Bayesian networks. In *Proceedings of the Sixteenth Conference on Uncertainty in
Artificial Intelligence (UAI)*, pages 176–183, 2000. Free: [PDF](https://www.cs.ubc.ca/~murphyk/Papers/rbpf_uai00.pdf).

\anchor Dunik2019
J. Duník, M. Soták, M. Veselý, O. Straka, and W. Hawkinson. Design of Rao–Blackwellized
point-mass filter with application in terrain aided navigation. *IEEE Transactions on Aerospace
and Electronic Systems*, 55(1):251–272, 2019.
**Not read.** Described in \ref Pocoma2023 "Pocoma et al. (2023)", §II-D and §III, which cite it
for the correction and prediction.

\anchor Fischer1993
W. Fischer and K. Meier-Hellstern. The Markov-modulated Poisson process (MMPP) cookbook.
*Performance Evaluation*, 18(2):149–171, 1993.
**Not read.** The likelihood is as given in S. Mews, J.-O. Koslik, and R. Langrock, How to build
your latent Markov model, [arXiv:2406.19157](https://arxiv.org/abs/2406.19157), §3.1.

\anchor GarciaFernandez2015
Á. F. García-Fernández, L. Svensson, M. R. Morelande, and S. Särkkä. Posterior linearization
filter: principles and implementation using sigma points. *IEEE Transactions on Signal
Processing*, 63(20):5561–5573, 2015. Algorithm 1, the iterated posterior linearization filter;
§IV-C on earlier iterated sigma-point filters. Free:
[author's version](https://publications.lib.chalmers.se/records/fulltext/224518/local_224518.pdf).

\anchor Gelb1974
A. Gelb, editor. *Applied Optimal Estimation*. MIT Press, 1974.
**Not read.** The Gauss-Markov process is described in
[Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process); no
free source read describes the linearized Kalman filter.

\anchor Godsill2004
S. J. Godsill, A. Doucet, and M. West. Monte Carlo smoothing for nonlinear time series.
*Journal of the American Statistical Association*, 99(465):156–168, 2004. Backward simulation
for particles, which decisions.md [smoothing] leaves out until a process model exposes its
transition density.

\anchor Golub2013
G. H. Golub and C. F. Van Loan. *Matrix Computations*. Fourth edition, Johns Hopkins University
Press, 2013.
**Not read.** The comments link
[QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections),
[Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition), and
[Square root of a matrix](https://en.wikipedia.org/wiki/Square_root_of_a_matrix).

\anchor Gordon1993
N. J. Gordon, D. J. Salmond, and A. F. M. Smith. Novel approach to nonlinear/non-%Gaussian
Bayesian state estimation. *IEE Proceedings F – Radar and Signal Processing*, 140(2):107–113,
1993. Free: [PDF, course copy](https://www.irisa.fr/aspi/legland/ensta/ref/gordon93a.pdf).

\anchor Grassmann1985
W. K. Grassmann, M. I. Taksar, and D. P. Heyman. Regenerative analysis and steady state
distributions for Markov chains. *Operations Research*, 33(5):1107–1116, 1985.
**Not read.** The GTH algorithm, as the code implements it (%Gaussian elimination of the states
from the last, the diagonal never read), is described in Q. Bu and Y. Q. Zhao, GTH algorithm,
censored Markov chains, and RG-factorization in block-form,
[arXiv:2604.14347](https://arxiv.org/abs/2604.14347), §2.

\anchor Hashimzade2024
N. Hashimzade, O. Kirsanov, T. Kirsanova, and J. Maih. On Bayesian filtering for Markov regime
switching models. arXiv:2402.08051, 2024. Free: [arXiv](https://arxiv.org/abs/2402.08051).

\anchor Helmick1995
R. E. Helmick, W. D. Blair, and S. A. Hoffman. Fixed-interval smoothing for Markovian switching
systems. *IEEE Transactions on Information Theory*, 41(6):1845–1855, 1995. The two-filter IMM
smoother, which decisions.md [smoothing] rejects, since it needs backward-time dynamics.

\anchor Hertzberg2013
C. Hertzberg, R. Wagner, U. Frese, and L. Schröder. Integrating generic sensor fusion algorithms
with sound state representations through encapsulation of manifolds. *Information Fusion*,
14(1):57–77, 2013. Free: [arXiv](https://arxiv.org/abs/1107.1119).

\anchor Higham2005
N. J. Higham. The scaling and squaring method for the matrix exponential revisited. *SIAM
Journal on Matrix Analysis and Applications*, 26(4):1179–1193, 2005. The method of Eigen's
matrix exponential, which ContinuousTimeMarkovChain::transition() calls.

\anchor Hol2006
J. D. Hol, T. B. Schön, and F. Gustafsson. On resampling algorithms for particle filters. In
*IEEE Nonlinear Statistical Signal Processing Workshop*, pages 79–82, 2006. Free: [PDF](https://people.isy.liu.se/en/rt/schon/Publications/HolSG2006.pdf).

\anchor Huber2008
M. F. Huber, T. Bailey, H. Durrant-Whyte, and U. D. Hanebeck. On entropy approximation for
%Gaussian mixture random vectors. In *IEEE International Conference on Multisensor Fusion and
Integration for Intelligent Systems (MFI)*, pages 181–188, 2008. The upper bound on a mixture's
entropy, the components' weighted entropies plus the entropy of the weights, which bounds the
lower end of kullbackLeiblerCollapseBounds().

\anchor Isserlis1918
L. Isserlis. On a formula for the product-moment coefficient of any order of a normal frequency
distribution in any number of variables. *Biometrika*, 12(1–2):134–139, 1918. Free: [PDF](https://zenodo.org/records/1431593).

\anchor Ito2000
K. Ito and K. Xiong. %Gaussian filters for nonlinear filtering problems. *IEEE Transactions on
Automatic Control*, 45(5):910–927, 2000. The Gauss-Hermite filter, which GaussHermite
implements as \ref Sarkka2013 "Särkkä (2013)" states it.

\anchor Julier2003
S. J. Julier. The spherical simplex unscented transformation. In *Proceedings of the American
Control Conference*, volume 3, pages 2430–2434, 2003. The spherical simplex set, which
SphericalSimplex builds by the construction of \ref Julier2004 "Julier and Uhlmann (2004)".

\anchor Julier2004
S. J. Julier and J. K. Uhlmann. Unscented filtering and nonlinear estimation. *Proceedings of the
IEEE*, 92(3):401–422, 2004.
**Not read.** The transform, its weights, and the Kalman filter built on it are as given in
\ref Sarkka2013 "Särkkä (2013)", §5.5, §5.6, and §6.5; no free source read describes the spherical
simplex construction.

\anchor Kaminski1971
P. G. Kaminski, A. E. Bryson, and S. F. Schmidt. Discrete square root filtering: a survey of
current techniques. *IEEE Transactions on Automatic Control*, 16(6):727–736, 1971. The
square-root covariance filter as a triangularization of stacked factors, in prediction and in the
update.
**Not read.** Described in K. S. Tracy, A square-root Kalman filter using only QR decompositions,
[arXiv:2208.06452](https://arxiv.org/abs/2208.06452), §3, which forms the innovation factor and the
posterior separately rather than as one array.

\anchor Kalman1960
R. E. Kalman. A new approach to linear filtering and prediction problems. *Journal of Basic
Engineering*, 82(1):35–45, 1960. Free: [PDF, course copy](https://www.cs.yale.edu/homes/yry/readings/general/Kalman1960.pdf).

\anchor Ke2024
T. Ke, P. Agrawal, Y. Zhang, W. Zhen, C. X. Guo, T. Sharp, and R. C. Dutoit. PC-SRIF:
Preconditioned Cholesky-based square root information filter for vision-aided inertial
navigation. arXiv:2409.11372, 2024. Free: [arXiv](https://arxiv.org/abs/2409.11372).

\anchor Kim1994
C.-J. Kim. Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60(1–2):1–22,
1994. The smoothed regime probabilities, from the filtered ones and the next step's smoothed ones
under the approximation that the regime at \f$ t \f$ given the regime at \f$ t + 1 \f$ depends on
the data up to \f$ t \f$ alone, which Barber (2006) corrects.
**Not read.** Described in \ref Hashimzade2024 "Hashimzade et al. (2024)", §2.3.1, and
\ref Sarkka2013 "Särkkä (2013)", §11.4.

\anchor Kitagawa1996
G. Kitagawa. Monte Carlo filter and smoother for non-%Gaussian nonlinear state space models.
*Journal of Computational and Graphical Statistics*, 5(1):1–25, 1996.
**Not read.** Stratified resampling is described in \ref Douc2005 "Douc et al. (2005)", §2.3, and
the smoother in \ref Sarkka2013 "Särkkä (2013)", §11.1.

\anchor Kong1994
A. Kong, J. S. Liu, and W. H. Wong. Sequential imputations and Bayesian missing data problems.
*Journal of the American Statistical Association*, 89(425):278–288, 1994.
**Not read.** Described in \ref Bergman1999 "Bergman (1999)", §6.4.1, equation (6.28), and
\ref Arulampalam2002 "Arulampalam et al. (2002)", §V-A.

\anchor Kullback1951
S. Kullback and R. A. Leibler. On information and sufficiency. *Annals of Mathematical
Statistics*, 22(1):79–86, 1951. Free: [open access](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full).

\anchor Lefebvre2002
T. Lefebvre, H. Bruyninckx, and J. De Schutter. Comment on "A new method for the nonlinear
transformation of means and covariances in filters and estimators". *IEEE Transactions on
Automatic Control*, 47(8):1406–1409, 2002. Free: [PDF, course copy](https://www.ene.unb.br/gaborges/disciplinas/pe/papers/julier2000comment.pdf).

\anchor Lentz1976
W. J. Lentz. Generating Bessel functions in Mie scattering calculations using continued
fractions. *Applied Optics*, 15(3):668–671, 1976.
**Not read.** Described in
[Lentz's algorithm](https://en.wikipedia.org/wiki/Lentz%27s_algorithm#Algorithm).

\anchor Lin1991
J. Lin. Divergence measures based on the Shannon entropy. *IEEE Transactions on Information
Theory*, 37(1):145–151, 1991.
**Not read.** Described in
[Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence#Definition).

\anchor Liu1998
J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic systems. *Journal of the
American Statistical Association*, 93(443):1032–1044, 1998.
**Not read.** Residual resampling is described in \ref Douc2005 "Douc et al. (2005)", §2.2, and
resampling on the effective sample size in \ref Arulampalam2002 "Arulampalam et al. (2002)", §V-A.

\anchor Markley2003
F. L. Markley. Attitude error representations for Kalman filtering. *Journal of Guidance,
Control, and Dynamics*, 26(2):311–317, 2003. The text checked is the NASA preprint of 2002
(NTRS 20020060647), whose reset section says the reset leaves the covariance unchanged. Free: [NASA preprint](https://ntrs.nasa.gov/citations/20020060647).

\anchor Mueller2017
M. W. Mueller, M. Hehn, and R. D'Andrea. Covariance correction step for Kalman filtering with an
attitude. *Journal of Guidance, Control, and Dynamics*, 40(9):2301–2306, 2017. The text checked
is the authors' preprint. Free: [preprint](https://hiperlab.berkeley.edu/wp-content/uploads/2018/05/2016_CovarianceCorrectionStepForKalmanFilteringWithAnAttitude.pdf).

\anchor Murray2016
L. M. Murray, A. Lee, and P. E. Jacob. Parallel resampling in the particle filter. *Journal of
Computational and Graphical Statistics*, 25(3):789–805, 2016. Free: [arXiv](https://arxiv.org/abs/1301.4019).

\anchor Musso2001
C. Musso, N. Oudjane, and F. Le Gland. Improving regularised particle filters. In A. Doucet,
N. de Freitas, and N. Gordon, editors, *Sequential Monte Carlo Methods in Practice*, pages
247–271. Springer, 2001.
**Not read.** Described in \ref Arulampalam2002 "Arulampalam et al. (2002)", §V-B3, with an
Epanechnikov kernel where the library uses a %Gaussian one.

\anchor Mutambara1998
A. G. O. Mutambara. *Decentralized Estimation and Control for Multisensor Systems*. CRC Press,
1998. The information filter, the Kalman filter in terms of the precision and information vector,
and its extended form, which linearizes the measurement at the estimate and adds
\f$ H^T R^{-1} H \f$ and the matching information vector.
**Not read.** The linear information filter is described in
[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter#Information_filter); no free source read
covers the extended form.

\anchor Nadarajah2009
N. Nadarajah. *Retrodiction for Multitarget Tracking*. PhD thesis, McMaster University, 2009.
Chapter 8 derives and states the backward IMM smoother. Free: [PDF](https://macsphere.mcmaster.ca/bitstream/11375/17364/1/Nadarajah_N_2009Jul_PhD.pdf).

\anchor Nadarajah2012
N. Nadarajah, R. Tharmarasa, M. McDonald, and T. Kirubarajan. IMM forward filtering and backward
smoothing for maneuvering target tracking. *IEEE Transactions on Aerospace and Electronic
Systems*, 48(3):2673–2678, 2012. The journal version of Nadarajah (2009), chapter 8; the code
follows the thesis, which is the text checked.
**Not read.** The comment that cites it follows \ref Nadarajah2009 "Nadarajah (2009)", which this
pass did not recheck.

\anchor Pasha2009
S. A. Pasha, B.-N. Vo, H. D. Tuan, and W.-K. Ma. A %Gaussian mixture PHD filter for jump Markov
system models. *IEEE Transactions on Aerospace and Electronic Systems*, 45(3):919–936, 2009. Free: [repository](https://curate.curtin.edu.au/articles/A_Gaussian_Mixture_PHD_Filter_for_Jump_Markov_System_Models/31758640).

\anchor Pennec2006
X. Pennec. Intrinsic statistics on Riemannian manifolds: basic tools for geometric measurements.
*Journal of Mathematical Imaging and Vision*, 25(1):127–154, 2006.
**Not read.** The iteration is as given in \ref Hertzberg2013 "Hertzberg et al. (2013)", §3.4. Free
copy: [CORE](https://core.ac.uk/download/49969478.pdf), not yet read.

\anchor Pocoma2023
E. Pocoma, L. Storrer, F. Quitin, L. Vandendorpe, P. De Doncker, and F. Horlin. Direct tracking of
a wireless transmitter based on Rao-Blackwellized point mass filter. *IEEE Transactions on
Vehicular Technology*, 2023. §II-C gives the factorization into the block and the rest, §II-D
and Algorithm 1 the filter, equations (17) and (18) the correction for a measurement of the block
alone, and §III-D the prediction between grids. Free:
[PDF, author's copy](https://dipot.ulb.ac.be/dspace/bitstream/2013/369562/3/difusion_RBPMF_TVT_2023.pdf).

\anchor Press2007
W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery. *Numerical Recipes: The Art
of Scientific Computing*, third edition. Cambridge University Press, 2007.
**Not read.** The series and continued fraction are described in
[Incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function#Evaluation_formulae);
no free source read gives the switch between them or rtsafe's safeguard.

\anchor Rabiner1989
L. R. Rabiner. A tutorial on hidden Markov models and selected applications in speech
recognition. *Proceedings of the IEEE*, 77(2):257–286, 1989. Free: [PDF, course copy](https://courses.physics.illinois.edu/ece417/fa2017/rabiner89.pdf).

\anchor Rahmathullah2014
A. S. Rahmathullah, L. Svensson, and D. Svensson. Merging-based forward-backward smoothing on
%Gaussian mixtures. In *Proceedings of the 17th International Conference on Information Fusion*,
2014. Section IV and Algorithm 2 give the backward pass through merged components that the
mixture smoother follows; its equation (16) as printed drops the division by the merged weight
that its (15) has. Free: [PDF](https://publications.lib.chalmers.se/records/fulltext/198613/local_198613.pdf).

\anchor Rauch1965
H. E. Rauch, F. Tung, and C. T. Striebel. Maximum likelihood estimates of linear dynamic systems.
*AIAA Journal*, 3(8):1445–1450, 1965.
**Not read.** Described in \ref Sarkka2013 "Särkkä (2013)", §8.2 and §10.1.

\anchor Runnalls2007
A. R. Runnalls. Kullback-Leibler approach to %Gaussian mixture reduction. *IEEE Transactions on
Aerospace and Electronic Systems*, 43(3):989–999, 2007. Free: [preprint](https://www.cs.kent.ac.uk/pubs/2007/2797/content.pdf).

\anchor Sarkka2012
S. Särkkä, P. Bunch, and S. J. Godsill. A backward-simulation based Rao-Blackwellized particle
smoother for conditionally linear %Gaussian models. In *16th IFAC Symposium on System
Identification (SYSID)*, 2012. The Rao-Blackwellized smoother that decisions.md [smoothing]
defers. Free: [author's PDF](https://users.aalto.fi/~ssarkka/pub/rb-smoother-sysid.pdf).

\anchor Sarkka2013
S. Särkkä. *Bayesian Filtering and Smoothing*. Cambridge University Press, 2013. Free: [author's PDF](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).

\anchor Schmidt1966
S. F. Schmidt. Application of state-space methods to navigation problems. In C. T. Leondes,
editor, *Advances in Control Systems*, volume 3, pages 293–340. Academic Press, 1966. The
consider, or Schmidt-Kalman, update.

\anchor Schon2005
T. Schön, F. Gustafsson, and P.-J. Nordlund. Marginalized particle filters for mixed
linear/nonlinear state-space models. *IEEE Transactions on Signal Processing*,
53(7):2279–2289, 2005. Free: [preprint](https://people.isy.liu.se/rt/schon/Publications/SchonGN2004.pdf).

\anchor Silverman1986
B. W. Silverman. *Density Estimation for Statistics and Data Analysis*. Chapman and Hall, 1986.
**Not read.** Described in
[Multivariate kernel density estimation](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb),
with a diagonal bandwidth where the library takes the data's covariance.

\anchor Smidl2013
V. Šmídl and M. Gašperin. Rao-Blackwellized point mass filter for reliable state estimation. In
*Proceedings of the 16th International Conference on Information Fusion*, pages 312–318, 2013.
**Not read.** Described in \ref Pocoma2023 "Pocoma et al. (2023)", §II-D.

\anchor Sola2017
J. Solà. Quaternion kinematics for the error-state Kalman filter. Technical report,
arXiv:1711.02508, 2017. §6.3, the ESKF reset, eqs. (284) to (288) and (294). Free: [arXiv](https://arxiv.org/abs/1711.02508).

\anchor Sola2018
J. Solà, J. Deray, and D. Atchuthan. A micro Lie theory for state estimation in robotics.
Technical report IRI-TR-18-01, arXiv:1812.01537, 2018 (version 9, 2021). Appendix B, the right
Jacobian of SO(3), eq. (143). Free: [arXiv](https://arxiv.org/abs/1812.01537).

\anchor VanDerMerwe2001
R. van der Merwe and E. A. Wan. The square-root unscented Kalman filter for state and
parameter-estimation. In *Proceedings of the IEEE International Conference on Acoustics, Speech,
and Signal Processing (ICASSP)*, volume 6, pages 3461–3464, 2001. Sigma points spread from the
covariance's factor, and the predicted factor by triangularizing their weighted deviations with
the noise's factor; the code departs in the update, which triangularizes the joint array rather
than applying their rank-one downdates, and takes the measurement's statistical linearization.
**Not read.** No free source read describes the square-root unscented Kalman filter.

\anchor Vasconcelos1999
N. Vasconcelos and A. Lippman. Learning mixture hierarchies. In *Advances in Neural Information
Processing Systems 11 (NIPS 1998)*, pages 606–612. MIT Press, 1999. Free: [PDF](https://proceedings.neurips.cc/paper_files/paper/1998/file/819c9fbfb075d62a16393b9fe4fcbaa5-Paper.pdf).

\anchor Viterbi1967
A. J. Viterbi. Error bounds for convolutional codes and an asymptotically optimum decoding
algorithm. *IEEE Transactions on Information Theory*, 13(2):260–269, 1967.
**Not read.** Described in \ref Rabiner1989 "Rabiner (1989)", §III-B.

\anchor Welford1962
B. P. Welford. Note on a method for calculating corrected sums of squares and products.
*Technometrics*, 4(3):419–420, 1962.
**Not read.** Described in
[Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).

\anchor Thompson1986
I. J. Thompson and A. R. Barnett. Coulomb and Bessel functions of complex arguments and order.
*Journal of Computational Physics*, 64(2):490–509, 1986.
**Not read.** Described in
[Lentz's algorithm](https://en.wikipedia.org/wiki/Lentz%27s_algorithm#Thompson_and_Barnett_modification).

\anchor Tichavsky1998
P. Tichavský, C. H. Muravchik, and A. Nehorai. Posterior Cramér-Rao bounds for discrete-time
nonlinear filtering. *IEEE Transactions on Signal Processing*, 46(5):1386–1396, 1998.
Proposition 1, eq. (21), is the recursion on the information matrix, and eqs. (34) to (36) its
matrices for additive %Gaussian noise; Section IV treats a singular process noise. Free:
[PDF, SEDICI](https://sedici.unlp.edu.ar/handle/10915/122993).
