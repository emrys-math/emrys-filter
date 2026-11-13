#pragma once

/** \file
 * Divergences between distributions (decisions.md [divergences]): Kullback-Leibler, Bhattacharyya,
 * Hellinger, Jensen-Shannon, and 2-Wasserstein, in closed form between two Gaussians or two
 * Categoricals and by MonteCarlo estimate otherwise; kullbackLeiblerCollapseBounds(), how far a
 * %Gaussian mixture or a hybrid state is from its moment-matched %Gaussian; GroundMetric for the
 * Wasserstein distance's units; and expectedSquaredError() about a known truth. Read
 * Conditionals.h next.
 */

#include <concepts>
#include <expected>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <EmrysFilter/Core/Execution.h>
#include <EmrysFilter/Core/Macros.h>
#include <EmrysFilter/Core/Math.h>
#include <EmrysFilter/Core/Multiply.h>
#include <EmrysFilter/Core/Solve.h>
#include <EmrysFilter/Core/Triangularization.h>
#include <EmrysFilter/Core/Weights.h>
#include <EmrysFilter/Distributions/Categorical.h>
#include <EmrysFilter/Distributions/Gaussian.h>
#include <EmrysFilter/Distributions/GaussianForm.h>
#include <EmrysFilter/Distributions/Hybrid.h>
#include <EmrysFilter/Distributions/Mixture.h>
#include <EmrysFilter/Distributions/Moments.h>
#include <EmrysFilter/Distributions/Product.h>
#include <EmrysFilter/Distributions/Sampling.h>
#include <EmrysFilter/Distributions/SqrtGaussian.h>
#include <EmrysFilter/Spaces/Space.h>
#include <EmrysFilter/Spaces/TagPath.h>

namespace Emrys
{

/** \defgroup Divergences Divergences
 * \ingroup Distributions
 *
 * How far apart two distributions are. Each function takes the two distributions in the order
 * the quantity is written, \f$ D(p \parallel q) \f$ as <tt>f(p, q)</tt>:
 *
 * | Function | Quantity | Range | Symmetric |
 * |---|---|---|---|
 * | kullbackLeibler() | \f$ \mathrm{KL}(p \parallel q) = E_p[\log p - \log q] \f$ | \f$ [0, \infty] \f$ | no |
 * | bhattacharyya() | \f$ D_B = -\log \int \sqrt{p q} \f$ | \f$ [0, \infty] \f$ | yes |
 * | hellinger() | \f$ H = \sqrt{1 - e^{-D_B}} \f$, a metric | \f$ [0, 1] \f$ | yes |
 * | jensenShannon() | \f$ \tfrac12 [\mathrm{KL}(p \| m) + \mathrm{KL}(q \| m)] \f$ | \f$ [0, \log 2] \f$ | yes |
 * | wasserstein() | \f$ W_2 \f$, the cost of moving \f$ p \f$ onto \f$ q \f$, a metric | \f$ [0, \infty) \f$ | yes |
 *
 * In jensenShannon(), \f$ m = \tfrac12 (p + q) \f$, the equal mixture of the two. Distances are
 * returned as distances, not squared. Logarithms are natural, so the divergences are in nats.
 *
 * kullbackLeiblerCollapseBounds() takes one distribution, a %Gaussian mixture, a HybridGaussian, or
 * a mixture of them, and brackets its kullbackLeibler() divergence from its moment-matched
 * %Gaussian in closed form: how much a report of its mean and covariance leaves out.
 *
 * Closed forms exist between two %Gaussian densities of one space (all but Jensen-Shannon, whose
 * mixture has no closed-form entropy) and between two Categorical distributions (all but
 * Wasserstein). The %Gaussian densities are what toDensity() returns in any of the four forms,
 * Gaussian, SqrtGaussian, InfoGaussian, and SqrtInfoGaussian, in any pair, so a square-root
 * filter's state is compared with a covariance-form reference as it is. They take densities
 * because they read a triangular root of each covariance, the cached Cholesky factor, the stored
 * root, or the inverse of the precision's root, and the log-determinants; a GaussianMarginal is
 * parameters, so copy() it first. Each
 * result is clamped at zero, so rounding never gives a negative divergence; a distribution
 * against itself gives exactly zero, except that wasserstein() gives about \f$ 10^{-8} \f$, the
 * square root of a rounding residue in \f$ W_2^2 \f$. Between categoricals the Bhattacharyya
 * coefficient is summed in log space, so one that underflows gives \f$ +\infty \f$.
 *
 * Any other pair is estimated by passing a MonteCarlo method first, as transform() takes its
 * method. A call without one for a pair that has no closed form fails to compile, with one error,
 * a message that names the estimator. A ParticleSet has no density, so it is compared through its
 * toDensity(), and the result is that kernel density's.
 *
 * On a manifold, the closed forms compare covariances that live in tangent spaces at
 * different means, without transporting one to the other: the approximation that the
 * tangent-space mean of a mixture makes. It is accurate while the means are close, as two
 * headings on either side of the seam are.
 *
 * The divergences are unchanged by any invertible map applied to both distributions, so a
 * weighting of the coordinates means nothing to them. Wasserstein, which measures in the
 * space's units, takes one: see GroundMetric and expectedSquaredError().
 *
 * The quantities are \ref Kullback1951 "Kullback and Leibler (1951)",
 * \ref Bhattacharyya1943 "Bhattacharyya (1943)", the
 * [Hellinger distance](https://en.wikipedia.org/wiki/Hellinger_distance), the Jensen-Shannon
 * divergence of \ref Lin1991 "Lin (1991)", and the
 * [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric); each function's comment
 * gives the closed form it uses and where that comes from.
 */

/** \ingroup Divergences
 * \brief Method that estimates a divergence from \c count draws.
 *
 * Under execution::sequential an estimate takes its draws one at a time and stores none, so on
 * a bounded space it allocates nothing [allocation]; under another policy it stores the \c count
 * draws, which the policy consents to.
 */
struct MonteCarlo
{
    Eigen::Index count;  ///< draws from each distribution sampled, at least two
};

/** \ingroup Divergences
 * \brief An estimate from MonteCarlo draws, with its standard error.
 */
template <typename Scalar = double>
struct MonteCarloEstimate
{
    Scalar value;          ///< the estimate
    Scalar standardError;  ///< its standard error, by the delta method for a function of a mean
    Eigen::Index count;    ///< draws taken from each distribution sampled
};

namespace internal
{

/** The number of indices in \a indices: an index or an Eigen arithmetic sequence, as
 * diffIdx() returns.
 */
template <typename Indices>
[[nodiscard]] Eigen::Index
indexCount(Indices const & indices)
{
    if constexpr (std::is_integral_v<Indices>)
    {
        return 1;
    }
    else
    {
        return Eigen::Index(indices.size());
    }
}

/** The \a k-th index in \a indices. */
template <typename Indices>
[[nodiscard]] Eigen::Index
indexAt(Indices const & indices, Eigen::Index k)
{
    if constexpr (std::is_integral_v<Indices>)
    {
        return Eigen::Index(indices);
    }
    else
    {
        return Eigen::Index(indices[k]);
    }
}

}  // namespace internal

/** \ingroup Divergences
 * \brief A weighting of a space's differences, built by tag path: the ground metric of
 * wasserstein().
 *
 * A matrix \f$ W \f$ with a row and a column per difference coordinate measures a difference
 * \f$ \delta \f$ as \f$ \|W \delta\| \f$. It starts as the identity, and each call applies a
 * map to the measured coordinates, so the calls compose in the order they are made:
 * scale() multiplies a block's rows, and combine() adds a multiple of one block's rows to
 * another's.
 *
 * Weighting expresses what an error costs, in units chosen by the caller, where a covariance
 * says how uncertain it is. To value every state by the error in position it causes over a
 * look-ahead time \f$ T \f$:
 *
 * \code
 * auto metric = GroundMetric{State{}};
 * metric.combine<Position, Velocity>(T);              // position T seconds ahead
 * metric.combine<Position, AccelerometerBias>(T * T / 2);
 * metric.scale<Velocity>(0);                          // measured only through position
 * metric.scale<AccelerometerBias>(0);
 * double const distance = wasserstein(p.toDensity().value(), q.toDensity().value(), metric);
 * \endcode
 *
 * Any matrix with a column per difference coordinate serves as well; this builds the square
 * ones by name.
 *
 * \tparam Space_   the space whose differences are measured
 * \tparam Scalar_  scalar type, \c double by default
 */
template <Space Space_, RealScalar Scalar_ = double>
class GroundMetric
{
public:

    using SpaceType = Space_;                                          ///< the space whose differences are measured
    using Scalar = Scalar_;                                            ///< the scalar type
    using Index = Eigen::Index;                                        ///< Eigen's index type
    using MatrixType = typename Moments<Space_, Scalar_>::MatrixType;  ///< \f$ W \f$'s type, square over differences

    /** The identity on \a space: each difference measured by its Euclidean norm. */
    explicit GroundMetric(Space_ space)
        : m_matrix{MatrixType::Identity(Emrys::differenceDimension(space), Emrys::differenceDimension(space))}
        , m_space{std::move(space)}
    {}

    /** Multiplies the measured coordinates of the block named by \a Path (a tag or a TagPath)
     * by \a factor. A factor of zero leaves the block measured only through what has been
     * combined into other blocks.
     */
    template <typename Path>
    void
    scale(Scalar factor)
    {
        auto const rows = diffIdx<Path>(m_space);
        for (Index k = 0; k < internal::indexCount(rows); ++k)
        {
            m_matrix.row(internal::indexAt(rows, k)) *= factor;
        }
    }

    /** Adds \a factor times the measured coordinates of the block named by \a From to those
     * of \a Into, coordinate by coordinate: \a Into then measures
     * \f$ \delta_{\mathrm{Into}} + \mathrm{factor} \, \delta_{\mathrm{From}} \f$. The two
     * blocks must have the same difference dimension.
     */
    template <typename Into, typename From>
    void
    combine(Scalar factor)
    {
        auto const into = diffIdx<Into>(m_space);
        auto const from = diffIdx<From>(m_space);
        EMRYS_FILTER_ASSERT(
            internal::indexCount(into) == internal::indexCount(from)
            && "combined blocks must have the same difference dimension"
        );
        MatrixType const before = m_matrix;
        for (Index k = 0; k < internal::indexCount(into); ++k)
        {
            m_matrix.row(internal::indexAt(into, k)) += factor * before.row(internal::indexAt(from, k));
        }
    }

    /** \returns \f$ W \f$ */
    [[nodiscard]] MatrixType const &
    matrix() const
    {
        return m_matrix;
    }

    /** \returns the space whose differences are measured */
    [[nodiscard]] Space_ const &
    space() const
    {
        return m_space;
    }

private:

    MatrixType m_matrix;                            ///< \f$ W \f$
    EMRYS_FILTER_NO_UNIQUE_ADDRESS Space_ m_space;  ///< the space, for resolving tag paths
};

namespace internal
{

/** False for every type: makes a static_assert in a template depend on its parameter. */
template <typename>
inline constexpr bool alwaysFalse = false;

/** What a divergence with no closed form returns. It converts to any type, so that a call that
 * keeps the result fails only at the static_assert that names the estimator, and it names no
 * scalar, so that a \c float build holds no \c double [embedded-builds]. The conversion is
 * declared and never defined, since no call that returns this type compiles.
 */
struct NoClosedForm
{
    /** Converts to the type the caller keeps the result as. */
    template <typename T>
    operator T() const;
};

/** A programmer error unless \a p and \a q have the same value and difference dimensions. */
inline void
assertSameDimensions(auto const & p, auto const & q)
{
    EMRYS_FILTER_ASSERT(
        p.valueDimension() == q.valueDimension() && p.differenceDimension() == q.differenceDimension()
        && "the distributions must have the same dimensions"
    );
}

/** Running mean and variance of a stream of terms, by the update of
 * \ref Welford1962 "Welford (1962)", which is stable where accumulating sums of squares is not. A
 * term of \f$ +\infty \f$ makes the mean and its error \f$ +\infty \f$ for good, where the update
 * would make them NaN.
 */
template <typename Scalar>
class RunningMean
{
public:

    /** Adds the term \a x. */
    void
    add(Scalar x)
    {
        if (x == Eigen::NumTraits<Scalar>::infinity())
        {
            m_infinite = true;
        }
        if (m_infinite)
        {
            return;
        }
        ++m_count;
        Scalar const delta = x - m_mean;
        m_mean += delta / Scalar(m_count);
        m_sumOfSquares += delta * (x - m_mean);
    }

    /** \returns whether a term was \f$ +\infty \f$, after which no term changes the mean */
    [[nodiscard]] bool
    infinite() const
    {
        return m_infinite;
    }

    /** \returns the mean of the terms so far */
    [[nodiscard]] Scalar
    mean() const
    {
        return m_infinite ? Eigen::NumTraits<Scalar>::infinity() : m_mean;
    }

    /** \returns the standard error of mean(), from the sample variance */
    [[nodiscard]] Scalar
    standardError() const
    {
        return m_infinite ? Eigen::NumTraits<Scalar>::infinity()
                          : Eigen::numext::sqrt(m_sumOfSquares / Scalar(m_count - 1) / Scalar(m_count));
    }

private:

    Eigen::Index m_count = 0;   ///< terms so far
    Scalar m_mean = 0;          ///< their mean
    Scalar m_sumOfSquares = 0;  ///< the sum of their squared deviations from the running mean
    bool m_infinite = false;    ///< whether a term was +infinity
};

/** \returns \a Scalar's positive infinity */
template <typename Scalar>
[[nodiscard]] Scalar
infinity()
{
    return Eigen::NumTraits<Scalar>::infinity();
}

/** \f$ \log(\tfrac12 (e^a + e^b)) \f$, the log-density of the even mixture of two densities. */
template <typename Scalar>
[[nodiscard]] Scalar
logEvenMixture(Scalar a, Scalar b)
{
    auto sum = LogSumExp<Scalar>{};
    sum.add(a);
    sum.add(b);
    return sum.value() - Eigen::numext::log(Scalar(2));
}

/** The Hellinger distance \f$ \sqrt{1 - e^{-D_B}} \f$ from the Bhattacharyya distance. */
template <typename Scalar>
[[nodiscard]] Scalar
hellingerFromBhattacharyya(Scalar distance)
{
    return Eigen::numext::sqrt(Eigen::numext::maxi(-Eigen::numext::expm1(-distance), Scalar(0)));
}

/** \f$ W_2 \f$ between two Gaussians, given the measured difference of their means and roots
 * \a first and \a second of their measured covariances: the closed form of
 * \ref Dowson1982 "Dowson and Landau (1982)", with the cross term
 * \f$ \mathrm{tr}\big((\Sigma_p \Sigma_q)^{1/2}\big) \f$ the nuclear norm of
 * \f$ A^T B \f$, since the squared singular values of \f$ A^T B \f$ are the eigenvalues of
 * \f$ A A^T B B^T \f$.
 */
template <typename Scalar, typename DifferenceDerived, typename FirstDerived, typename SecondDerived>
[[nodiscard]] Scalar
buresWasserstein(
    Eigen::MatrixBase<DifferenceDerived> const & difference,
    Eigen::MatrixBase<FirstDerived> const & first,
    Eigen::MatrixBase<SecondDerived> const & second
)
{
    using Cross = MatrixOf<
        Scalar,
        FirstDerived::ColsAtCompileTime,
        SecondDerived::ColsAtCompileTime,
        FirstDerived::MaxColsAtCompileTime,
        SecondDerived::MaxColsAtCompileTime>;
    Cross const cross = multiply(first.transpose(), second);
    Scalar const nuclearNorm = Eigen::JacobiSVD<Cross>{cross}.singularValues().sum();
    Scalar const squared =
        difference.squaredNorm() + first.squaredNorm() + second.squaredNorm() - Scalar(2) * nuclearNorm;
    return Eigen::numext::sqrt(Eigen::numext::maxi(squared, Scalar(0)));
}

/** Whether \a D is a %Gaussian density: a Gaussian, or the FormGaussian of a square-root or an
 * information form, each what its parameters' toDensity() returns.
 */
template <typename D>
inline constexpr bool isGaussianDensity = false;

/** A Gaussian is a %Gaussian density. */
template <typename Space, typename Scalar>
inline constexpr bool isGaussianDensity<Gaussian<Space, Scalar>> = true;

/** The density of a square-root or information form is a %Gaussian density. */
template <typename Params>
inline constexpr bool isGaussianDensity<FormGaussian<Params>> = true;

/** Two %Gaussian densities, in any of the four forms, of one space and scalar: the pairs that
 * the closed forms take.
 */
template <typename P, typename Q>
concept GaussianDensitiesOfOneSpace =
    isGaussianDensity<P> && isGaussianDensity<Q> && std::same_as<typename P::SpaceType, typename Q::SpaceType>
    && std::same_as<typename P::Scalar, typename Q::Scalar>;

/** Whether the %Gaussian density \a D is the covariance form's Gaussian, with a cached Cholesky
 * factorization.
 */
template <typename D>
inline constexpr bool isCovarianceDensity = std::same_as<D, Gaussian<typename D::SpaceType, typename D::Scalar>>;

/** The triangle that covarianceRootOf() fills for the %Gaussian density \a D: lower, but upper
 * in the information forms, whose covariance root is the inverse of the precision's upper root.
 */
template <typename D>
inline constexpr int covarianceRootTriangle = D::ParamsType::IsInfoForm ? Eigen::Upper : Eigen::Lower;

/** \returns a triangular root \f$ A \f$ of the covariance of the %Gaussian density \a d,
 * \f$ A A^T = \Sigma \f$, in the triangle covarianceRootTriangle names, with zeros in the other:
 * the Cholesky factor of a Gaussian, the stored root of a SqrtGaussian, and in the information
 * forms the inverse of the precision's root, by one triangular solve. Every divergence in closed
 * form reads each %Gaussian through this root, so it takes the four forms in any pair.
 */
template <typename D>
[[nodiscard]] typename D::MatrixType
covarianceRootOf(D const & d)
{
    using MatrixType = typename D::MatrixType;
    if constexpr (isCovarianceDensity<D>)
    {
        return MatrixType{d.llt().matrixL()};
    }
    else if constexpr (D::ParamsType::IsInfoForm)
    {
        Eigen::Index const n = d.differenceDimension();
        return d.colorNormals(MatrixType::Identity(n, n));
    }
    else
    {
        return d.root();
    }
}

/** Solves \f$ A X = B \f$ in place for the covariance root \f$ A \f$ of the %Gaussian density
 * \a d, whitening the columns of \a matrix: by the cached factorization of a Gaussian and the
 * stored root of a SqrtGaussian, and by covarianceRootOf() in the information forms.
 */
template <typename D, typename Derived>
void
whitenInPlace(D const & d, Eigen::MatrixBase<Derived> & matrix)
{
    if constexpr (isCovarianceDensity<D>)
    {
        solveInPlace(d.llt().matrixL(), matrix);
    }
    else if constexpr (D::ParamsType::IsInfoForm)
    {
        auto const root = covarianceRootOf(d);
        solveInPlace(root.template triangularView<Eigen::Upper>(), matrix);
    }
    else
    {
        solveInPlace(d.root().template triangularView<Eigen::Lower>(), matrix);
    }
}

/** \returns the lower-triangular root of \f$ \tfrac12 (A_p A_p^T + A_q A_q^T) \f$, the average
 * covariance of two %Gaussian densities, by triangularization of \f$ [\,A_p \;\; A_q\,] / \sqrt 2 \f$
 * with their covarianceRootOf()
 */
template <typename P, typename Q>
[[nodiscard]] typename P::MatrixType
averageFactor(P const & p, Q const & q)
{
    using Scalar = typename P::Scalar;
    using MatrixType = typename P::MatrixType;
    constexpr int Rows = MatrixType::RowsAtCompileTime;
    constexpr int MaxRows = MatrixType::MaxRowsAtCompileTime;
    using Stack = MatrixOf<
        Scalar,
        Rows,
        Rows == Eigen::Dynamic ? Eigen::Dynamic : 2 * Rows,
        MaxRows,
        MaxRows == Eigen::Dynamic ? Eigen::Dynamic : 2 * MaxRows>;
    Eigen::Index const n = p.differenceDimension();
    auto stack = Stack{n, 2 * n};
    Scalar const half = Eigen::numext::sqrt(Scalar(0.5));
    stack.leftCols(n) = half * covarianceRootOf(p);
    stack.rightCols(n) = half * covarianceRootOf(q);
    return lowerTriangularRoot(stack);
}

/** A distribution that drawSampleWithLogDensity() can sample. */
template <typename P, typename G>
concept SampleWithLogDensityFrom = requires(P const & p, G & rng) { drawSampleWithLogDensity(p, rng); };

/** Two distributions a MonteCarlo estimate can compare: \a P is sampled, and \a Q evaluated. */
template <typename P, typename Q, typename G>
concept MonteCarloComparable =
    SampleWithLogDensityFrom<P, G> && HasLogDensity<Q> && std::same_as<typename P::SpaceType, typename Q::SpaceType>
    && std::same_as<typename P::Scalar, typename Q::Scalar>;

/** \returns the RunningMean of \a termOf(sample, logP) over \a count draws from \a p, each with
 * \a p's log-density \a logP at it, added in the order drawn.
 *
 * Under Sequential each draw is taken by drawSampleWithLogDensity(), evaluated, and added in turn,
 * so nothing is stored and a bounded space allocates nothing [allocation]; a term of
 * \f$ +\infty \f$ ends the stream, since it settles the mean. Under another policy the draws are
 * taken the same way first, into storage the policy consents to, and their terms are a for-each
 * over them (see \ref Execution), so the mean and its error are Sequential's.
 */
template <typename Scalar, typename Policy, typename P, typename G, typename TermOf>
[[nodiscard]] RunningMean<Scalar>
meanOverDraws(Policy && policy, P const & p, Eigen::Index count, G & rng, TermOf const & termOf)
{
    EMRYS_FILTER_ASSERT(count >= 2 && "a Monte Carlo estimate needs at least two draws");
    auto terms = RunningMean<Scalar>{};
    if constexpr (isSequential<Policy>)
    {
        for (Eigen::Index j = 0; j < count && !terms.infinite(); ++j)
        {
            auto const draw = drawSampleWithLogDensity(p, rng);
            terms.add(termOf(draw.sample, draw.logDensity));
        }
    }
    else
    {
        auto samples = SampleSet<typename P::SpaceType, Scalar>{p.valueDimension(), count};
        auto logP = Eigen::Array<Scalar, Eigen::Dynamic, 1>{count};
        for (Eigen::Index j = 0; j < count; ++j)
        {
            auto const draw = drawSampleWithLogDensity(p, rng);
            samples.col(j) = draw.sample;
            logP(j) = draw.logDensity;
        }
        auto values = Eigen::Array<Scalar, Eigen::Dynamic, 1>{count};
        forEachIndex(std::forward<Policy>(policy), count, [&](Eigen::Index j) {
            values(j) = termOf(samples.col(j), logP(j));
        });
        for (Eigen::Index j = 0; j < count && !terms.infinite(); ++j)
        {
            terms.add(values(j));
        }
    }
    return terms;
}

/** The terms \f$ \sqrt{q(x) / p(x)} \f$ at draws from \a p, whose mean is the Bhattacharyya
 * coefficient \f$ \int \sqrt{p q} \f$, evaluated under \a policy.
 */
template <typename Scalar, typename Policy, typename P, typename Q, typename G>
[[nodiscard]] RunningMean<Scalar>
bhattacharyyaCoefficientTerms(Policy && policy, MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    return meanOverDraws<Scalar>(
        std::forward<Policy>(policy),
        p,
        method.count,
        rng,
        [&](auto const & sample, Scalar logP) {
            return Eigen::numext::exp(Scalar(0.5) * (q.logDensity(sample) - logP));
        }
    );
}

/** The terms \f$ \log p(x) - \log m(x) \f$ at draws from \a p, with \f$ m \f$ the even
 * mixture of \a p and \a q, whose mean is \f$ \mathrm{KL}(p \parallel m) \f$, evaluated under
 * \a policy.
 */
template <typename Scalar, typename Policy, typename P, typename Q, typename G>
[[nodiscard]] RunningMean<Scalar>
mixtureDivergenceTerms(Policy && policy, MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    return meanOverDraws<Scalar>(
        std::forward<Policy>(policy),
        p,
        method.count,
        rng,
        [&](auto const & sample, Scalar logP) {
            return Scalar(logP - logEvenMixture(logP, Scalar(q.logDensity(sample))));
        }
    );
}

/** \f$ E\|x \boxminus x^*\|^2 \f$ from \a moments. */
template <typename Space, typename MomentsType, typename TruthDerived>
[[nodiscard]] typename MomentsType::Scalar
expectedSquaredErrorOf(Space const & space, MomentsType const & moments, Eigen::MatrixBase<TruthDerived> const & truth)
{
    EMRYS_FILTER_ASSERT(truth.size() == moments.valueDimension() && "the truth must be a value of the space");
    typename MomentsType::DifferenceVectorType const error = Emrys::minus(space, moments.mean(), truth);
    return error.squaredNorm() + moments.covarianceLower().trace();
}

/** \f$ E\|W (x \boxminus x^*)\|^2 \f$ from \a moments, under the weighting \a weights. */
template <typename Space, typename MomentsType, typename TruthDerived, typename WeightsDerived>
[[nodiscard]] typename MomentsType::Scalar
expectedSquaredErrorOf(
    Space const & space,
    MomentsType const & moments,
    Eigen::MatrixBase<TruthDerived> const & truth,
    Eigen::MatrixBase<WeightsDerived> const & weights
)
{
    EMRYS_FILTER_ASSERT(truth.size() == moments.valueDimension() && "the truth must be a value of the space");
    EMRYS_FILTER_ASSERT(
        weights.cols() == moments.differenceDimension() && "the weights must have a column per difference coordinate"
    );
    typename MomentsType::DifferenceVectorType const error = Emrys::minus(space, moments.mean(), truth);
    // tr(W Σ W^T) is the sum of the entries of (W Σ) ∘ W.
    return multiply(weights, error).squaredNorm()
         + multiply(weights, moments.covarianceLower()).cwiseProduct(weights).sum();
}

/** \a compute applied to the moments of \a distribution, and wrapped in a std::expected when
 * its moments() returns one.
 */
template <typename Distribution, typename Compute>
[[nodiscard]] auto
fromMoments(Distribution const & distribution, Compute compute)
{
    using Result = std::remove_cvref_t<decltype(distribution.moments())>;
    if constexpr (isExpected<Result>)
    {
        using Value = decltype(compute(std::declval<typename Result::value_type const &>()));
        using Expected = std::expected<Value, typename Result::error_type>;
        Result const moments = distribution.moments();
        if (!moments)
        {
            return Expected{std::unexpect, moments.error()};
        }
        return Expected{compute(*moments)};
    }
    else
    {
        return compute(distribution.moments());
    }
}

}  // namespace internal

/** \ingroup Divergences
 * \returns \f$ \mathrm{KL}(p \parallel q) = \tfrac12 \left[ \mathrm{tr}(\Sigma_q^{-1} \Sigma_p)
 *     + (\mu_p \boxminus \mu_q)^T \Sigma_q^{-1} (\mu_p \boxminus \mu_q) - d
 *     + \log|\Sigma_q| - \log|\Sigma_p| \right] \f$
 * between two Gaussians, with \f$ d \f$ the difference dimension: the
 * [closed form for two multivariate
 * normals](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence). The
 * trace is \f$ \|A_q^{-1} A_p\|_F^2 \f$, by a triangular solve with roots \f$ A \f$ of the
 * covariances, so no covariance is formed. \a p and \a q are densities of one space in any of the
 * four forms, Gaussian, SqrtGaussian, InfoGaussian, or SqrtInfoGaussian, in any pair: the root is
 * the cached Cholesky factor of a Gaussian, the stored root of a SqrtGaussian, and the inverse of
 * the precision's root in the information forms.
 */
template <typename P, typename Q>
    requires internal::GaussianDensitiesOfOneSpace<P, Q>
[[nodiscard]] typename P::Scalar
kullbackLeibler(P const & p, Q const & q)
{
    using Scalar = typename P::Scalar;
    internal::assertSameDimensions(p, q);
    typename P::MatrixType whitened = internal::covarianceRootOf(p);
    internal::whitenInPlace(q, whitened);
    Scalar const divergence = Scalar(0.5)
                            * (whitened.squaredNorm() + q.squaredMahalanobisDistance(p.mean())
                               - Scalar(p.differenceDimension()) + q.logDeterminant() - p.logDeterminant());
    return Eigen::numext::maxi(divergence, Scalar(0));
}

/** \ingroup Divergences
 * \returns \f$ \mathrm{KL}(p \parallel q) = \sum_i p_i (\log p_i - \log q_i) \f$ between two
 * categorical distributions: \f$ +\infty \f$ where \f$ q \f$ gives zero probability to a value
 * that \f$ p \f$ doesn't.
 */
template <DiscreteSpace Space, typename Scalar>
[[nodiscard]] Scalar
kullbackLeibler(Categorical<Space, Scalar> const & p, Categorical<Space, Scalar> const & q)
{
    EMRYS_FILTER_ASSERT(p.cardinality() == q.cardinality() && "the distributions must have the same values");
    auto const logP = p.logProbabilities();
    auto const logQ = q.logProbabilities();
    Scalar divergence = 0;
    for (Eigen::Index i = 0; i < p.cardinality(); ++i)
    {
        if (logP(i) == -internal::infinity<Scalar>())
        {
            continue;
        }
        if (logQ(i) == -internal::infinity<Scalar>())
        {
            return internal::infinity<Scalar>();
        }
        divergence += Eigen::numext::exp(logP(i)) * (logP(i) - logQ(i));
    }
    return Eigen::numext::maxi(divergence, Scalar(0));
}

/** \ingroup Divergences
 * \returns a MonteCarlo estimate of \f$ \mathrm{KL}(p \parallel q) \f$: the mean of
 * \f$ \log p(x) - \log q(x) \f$ over draws from \a p. \a p is anything
 * drawSampleWithLogDensity() takes, such as a GaussianMixture or a KernelDensity, and
 * \a q any density (HasLogDensity) of the same space. The estimate is \f$ +\infty \f$ if \a q
 * is zero at a draw. The draws are taken one at a time and never stored, so a bounded space
 * allocates nothing [allocation].
 */
template <typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
kullbackLeibler(MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    return kullbackLeibler(execution::sequential, method, p, q, rng);
}

/** \ingroup Divergences
 * The MonteCarlo kullbackLeibler() under \a policy: the draws and \f$ \log p \f$ at each are
 * taken as Sequential takes them and stored, and \f$ \log q \f$ at each is a for-each over them,
 * so the estimate is Sequential's (see \ref Execution). The same form serves bhattacharyya(), hellinger(), and
 * jensenShannon().
 */
template <ExecutionPolicy Policy, typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
kullbackLeibler(Policy && policy, MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    using Scalar = typename P::Scalar;
    auto const terms = internal::meanOverDraws<Scalar>(
        std::forward<Policy>(policy),
        p,
        method.count,
        rng,
        [&](auto const & sample, Scalar logP) {
            Scalar const logQ = q.logDensity(sample);
            return logQ == -internal::infinity<Scalar>() ? internal::infinity<Scalar>() : Scalar(logP - logQ);
        }
    );
    return {terms.mean(), terms.standardError(), method.count};
}

/** \ingroup Divergences
 * kullbackLeibler() for a pair with no closed form: a compile-time error naming the estimator.
 */
template <typename P, typename Q>
[[nodiscard]] internal::NoClosedForm
kullbackLeibler(P const &, Q const &)
{
    static_assert(
        internal::alwaysFalse<P>,
        "kullbackLeibler(p, q) has a closed form only between two Gaussian densities of one space, in any of the four "
        "forms (call toDensity() on the parameters, and copy() a GaussianMarginal first), or two Categorical "
        "views of one space; "
        "estimate it otherwise with kullbackLeibler(MonteCarlo{count}, p, q, rng)"
    );
    return {};
}

/** \ingroup Divergences
 * \brief The bracket \f$ \max(B - H(\hat w), 0) \le \mathrm{KL}(p \parallel q) \le B \f$ on the
 * divergence of a distribution from its collapse, which kullbackLeiblerCollapseBounds() returns.
 */
template <typename Scalar = double>
struct CollapseBounds
{
    Scalar lower;  ///< \f$ \max(B - H(\hat w), 0) \f$: the divergence is at least this
    Scalar upper;  ///< \f$ B \f$: the divergence is at most this
};

namespace internal
{

/** \returns the CollapseBounds of a mixture of Gaussians whose moment-matched covariance is
 * \a covarianceLower (only its lower triangle is read). \a forEach calls the function it is given
 * once per %Gaussian, with the %Gaussian's normalized log-weight and its GaussianParams, whose
 * log-determinant is cached. It factorizes only the covariance, and allocates nothing on a bounded
 * space.
 */
template <typename Scalar, typename CovarianceDerived, typename ForEach>
[[nodiscard]] CollapseBounds<Scalar>
collapseBounds(Eigen::MatrixBase<CovarianceDerived> const & covarianceLower, ForEach forEach)
{
    using MatrixType = typename CovarianceDerived::PlainObject;
    auto const llt = Eigen::LLT<MatrixType>{covarianceLower};
    EMRYS_FILTER_ASSERT(llt.info() == Eigen::Success && "the mixture's covariance must be positive definite");
    Scalar componentLogDeterminants = 0;
    Scalar weightEntropy = 0;
    forEach([&](Scalar const logWeight, auto const & gaussian) {
        if (logWeight > -Eigen::NumTraits<Scalar>::infinity())
        {
            Scalar const weight = Eigen::numext::exp(logWeight);
            componentLogDeterminants += weight * FactorizedGaussian::logDeterminant(gaussian);
            weightEntropy -= weight * logWeight;
        }
    });
    Scalar const logDeterminant = Scalar(2) * llt.matrixLLT().diagonal().array().log().sum();
    Scalar const upper = Eigen::numext::maxi(Scalar(0.5) * (logDeterminant - componentLogDeterminants), Scalar(0));
    return CollapseBounds<Scalar>{Eigen::numext::maxi(upper - weightEntropy, Scalar(0)), upper};
}

}  // namespace internal

/** \ingroup Divergences
 * \returns the bracket on \f$ \mathrm{KL}(p \parallel q) \f$ from the %Gaussian mixture \a p to
 * its collapse \f$ q \f$, the %Gaussian with \a p's moments() that Mixture::collapsed() makes:
 * \f[ \max\big(B - H(\hat w), 0\big) \le \mathrm{KL}(p \parallel q) \le B, \qquad
 *     B = \tfrac12 \Big[ \log|\Sigma| - \sum_i \hat w_i \log|\Sigma_i| \Big], \f]
 * with \f$ \Sigma \f$ the mixture's covariance, \f$ \Sigma_i \f$ the components',
 * \f$ \hat w_i \f$ the normalized weights, and \f$ H(\hat w) = -\sum_i \hat w_i \log \hat w_i \f$
 * their entropy, at most \f$ \log n \f$ for \f$ n \f$ components.
 *
 * It says whether the mixture's mean and covariance, the ellipse a report draws, describe it. The
 * upper end is zero when every component is the same %Gaussian and grows as the components spread
 * apart; by [Pinsker's inequality](https://en.wikipedia.org/wiki/Pinsker%27s_inequality), no
 * probability read off \f$ q \f$, such as an ellipse's coverage, differs from the mixture's by more
 * than \f$ \sqrt{B / 2} \f$. The lower end says how far from %Gaussian the mixture is at least, so a
 * monitor can tell a mixture far from its collapse from one that the upper end only fails to vouch
 * for. The two are at most \f$ H(\hat w) \f$ apart, and the divergence approaches the lower end as
 * the components separate. The upper end depends only on the distribution, but the lower end
 * depends on how it is written: splitting a component into copies of itself raises
 * \f$ H(\hat w) \f$ and so lowers the lower end, so prune or merge copies before reading it. The upper end is loose
 * while the components overlap: for two equal unit-variance components at \f$ \pm 1 \f$, \f$ B = 0.35 \f$ where the
 * divergence is 0.009.
 *
 * A merge of two components leaves the mixture's covariance unchanged and replaces two terms of
 * \f$ \sum_i \hat w_i \log|\Sigma_i| \f$ by one, so it lowers \f$ B \f$ by exactly its Runnalls
 * cost: the upper end before a reduction less the upper end after it is what the reduction
 * threw away, and \f$ B \f$ is the total cost of merging down to one component, in any order.
 *
 * Since \f$ \log q \f$ is quadratic and \f$ q \f$ has \f$ p \f$'s moments,
 * \f$ \mathrm{KL}(p \parallel q) = H(q) - H(p) \f$, a difference of
 * [entropies](https://en.wikipedia.org/wiki/Differential_entropy). Conditioning on which
 * component drew a value cannot raise its entropy, so \f$ H(p) \ge \sum_i \hat w_i H(g_i) \f$,
 * which gives \f$ B \f$. Nor can the value's entropy exceed that of the pair, so
 * \f$ H(p) \le \sum_i \hat w_i H(g_i) + H(\hat w) \f$, the upper bound of
 * \ref Huber2008 "Huber et al. (2008)", which gives the lower end. \f$ B \f$ is the cost
 * \ref Runnalls2007 "Runnalls (2007)" charges for merging two components, as
 * MixtureParams::merged() says, taken over all of them at once.
 *
 * It takes the view for the components' cached log-determinants, factorizes only
 * \f$ \Sigma \f$, and allocates nothing on a bounded space. On a manifold, \f$ \Sigma \f$ is a
 * tangent-space covariance, as moments() computes it, so the bracket holds to the same
 * approximation as the closed forms.
 *
 * \sa kullbackLeibler(MonteCarlo const &, P const &, Q const &, G &), which estimates the
 * divergence itself.
 */
template <typename Space, typename Scalar, typename Allocator>
[[nodiscard]] CollapseBounds<Scalar>
kullbackLeiblerCollapseBounds(GaussianMixture<Space, Scalar, Allocator> const & p)
{
    auto const moments = p.moments();
    return internal::collapseBounds<Scalar>(moments.covarianceLower(), [&](auto const visit) {
        for (auto const & entry : p)
        {
            visit(entry.logWeight - p.logTotalWeight(), entry.component);
        }
    });
}

/** \ingroup Divergences
 * \returns the bracket on \f$ \mathrm{KL}(p \parallel q) \f$ from the hybrid state \a p, read
 * over its continuous state, to the %Gaussian \f$ q \f$ that Hybrid::collapsed() makes: the
 * bracket of the mixture of its conditionals weighted by the mode probabilities (see
 * kullbackLeiblerCollapseBounds(GaussianMixture const &)). It checks whether an ImmFilter's
 * report of its state as one %Gaussian describes the state.
 */
template <typename Space, typename ModeSpace, typename Scalar, typename Allocator>
[[nodiscard]] CollapseBounds<Scalar>
kullbackLeiblerCollapseBounds(HybridGaussian<Space, ModeSpace, Scalar, Allocator> const & p)
{
    auto const moments = p.moments();
    return internal::collapseBounds<Scalar>(moments.covarianceLower(), [&](auto const visit) {
        auto const logProbabilities = internal::logProbabilitiesOf(p.modes());
        for (Eigen::Index m = 0; m < p.size(); ++m)
        {
            visit(Scalar(logProbabilities(m)), p.conditional(m));
        }
    });
}

/** \ingroup Divergences
 * \returns the bracket on \f$ \mathrm{KL}(p \parallel q) \f$ from the mixture of hybrid states
 * \a p, read over its continuous state, to the %Gaussian \f$ q \f$ with its moments, which
 * <tt>p.collapsed().collapsed()</tt> makes: the bracket of the mixture of every component's
 * conditionals, each weighted by its component's weight times its mode's probability (see
 * kullbackLeiblerCollapseBounds(GaussianMixture const &)). It checks whether a mixture of IMM
 * states is described by one %Gaussian.
 */
template <typename Space, typename ModeSpace, typename Scalar, typename ComponentAllocator, typename Allocator>
[[nodiscard]] CollapseBounds<Scalar>
kullbackLeiblerCollapseBounds(
    Mixture<HybridGaussian<Space, ModeSpace, Scalar, ComponentAllocator>, Allocator> const & p
)
{
    auto const moments = p.moments();
    return internal::collapseBounds<Scalar>(moments.covarianceLower(), [&](auto const visit) {
        for (auto const & entry : p)
        {
            Scalar const logWeight = entry.logWeight - p.logTotalWeight();
            auto const logProbabilities = internal::logProbabilitiesOf(entry.component.modes());
            for (Eigen::Index m = 0; m < entry.component.size(); ++m)
            {
                visit(Scalar(logWeight + logProbabilities(m)), entry.component.conditional(m));
            }
        }
    });
}

/** \ingroup Divergences
 * \returns the Bhattacharyya distance between two Gaussians,
 * \f[ D_B = \tfrac18 (\mu_p \boxminus \mu_q)^T \Sigma^{-1} (\mu_p \boxminus \mu_q)
 *     + \tfrac12 \log \frac{|\Sigma|}{\sqrt{|\Sigma_p| |\Sigma_q|}},
 *     \qquad \Sigma = \tfrac12 (\Sigma_p + \Sigma_q). \f]
 * The first term measures how far apart the means are, the second how different the
 * covariances are: the
 * [closed form for two Gaussians](https://en.wikipedia.org/wiki/Bhattacharyya_distance#Gaussian_case).
 *
 * \a p and \a q are densities of one space in any of the four forms, in any pair, as in
 * kullbackLeibler(). Between two Gaussians the average covariance is factorized; otherwise its
 * root is one triangularization of \f$ [\,A_p \;\; A_q\,] / \sqrt 2 \f$, with \f$ A \f$ the roots
 * that kullbackLeibler() reads, so no covariance is formed.
 */
template <typename P, typename Q>
    requires internal::GaussianDensitiesOfOneSpace<P, Q>
[[nodiscard]] typename P::Scalar
bhattacharyya(P const & p, Q const & q)
{
    using Scalar = typename P::Scalar;
    using MatrixType = typename P::MatrixType;
    internal::assertSameDimensions(p, q);
    typename P::DifferenceVectorType difference = Emrys::minus(p.space(), p.mean(), q.mean());
    Scalar logDeterminant{};
    if constexpr (internal::isCovarianceDensity<P> && internal::isCovarianceDensity<Q>)
    {
        MatrixType const average = Scalar(0.5) * (p.covarianceLower() + q.covarianceLower());
        auto const llt = Eigen::LLT<MatrixType>{average};
        EMRYS_FILTER_ASSERT(llt.info() == Eigen::Success && "the average covariance must be positive definite");
        llt.matrixL().solveInPlace(difference);
        logDeterminant = Scalar(2) * llt.matrixLLT().diagonal().array().log().sum();
    }
    else
    {
        MatrixType const average = internal::averageFactor(p, q);
        EMRYS_FILTER_ASSERT(
            (average.diagonal().array() != Scalar(0)).all() && "the average covariance must be positive definite"
        );
        average.template triangularView<Eigen::Lower>().solveInPlace(difference);
        logDeterminant = Scalar(2) * average.diagonal().array().abs().log().sum();
    }
    Scalar const distance = Scalar(0.125) * difference.squaredNorm()
                          + Scalar(0.5) * (logDeterminant - Scalar(0.5) * (p.logDeterminant() + q.logDeterminant()));
    return Eigen::numext::maxi(distance, Scalar(0));
}

/** \ingroup Divergences
 * \returns the Bhattacharyya distance \f$ -\log \sum_i \sqrt{p_i q_i} \f$ between two
 * categorical distributions: \f$ +\infty \f$ if they share no value.
 */
template <DiscreteSpace Space, typename Scalar>
[[nodiscard]] Scalar
bhattacharyya(Categorical<Space, Scalar> const & p, Categorical<Space, Scalar> const & q)
{
    EMRYS_FILTER_ASSERT(p.cardinality() == q.cardinality() && "the distributions must have the same values");
    auto logCoefficient = LogSumExp<Scalar>{};
    for (Eigen::Index i = 0; i < p.cardinality(); ++i)
    {
        logCoefficient.add(Scalar(0.5) * (p.logProbability(i) + q.logProbability(i)));
    }
    return Eigen::numext::maxi(-logCoefficient.value(), Scalar(0));
}

/** \ingroup Divergences
 * \returns a MonteCarlo estimate of the Bhattacharyya distance: minus the log of the mean of
 * \f$ \sqrt{q(x) / p(x)} \f$ over draws from \a p, which estimates \f$ \int \sqrt{p q} \f$. See
 * kullbackLeibler(MonteCarlo const &, P const &, Q const &, G &) for what \a p and \a q may be.
 */
template <typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
bhattacharyya(MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    return bhattacharyya(execution::sequential, method, p, q, rng);
}

/** \ingroup Divergences
 * The MonteCarlo bhattacharyya() under \a policy, as the MonteCarlo kullbackLeibler() runs under
 * one.
 */
template <ExecutionPolicy Policy, typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
bhattacharyya(Policy && policy, MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    using Scalar = typename P::Scalar;
    auto const terms = internal::bhattacharyyaCoefficientTerms<Scalar>(std::forward<Policy>(policy), method, p, q, rng);
    return {-Eigen::numext::log(terms.mean()), terms.standardError() / terms.mean(), method.count};
}

/** \ingroup Divergences
 * bhattacharyya() for a pair with no closed form: a compile-time error naming the estimator.
 */
template <typename P, typename Q>
[[nodiscard]] internal::NoClosedForm
bhattacharyya(P const &, Q const &)
{
    static_assert(
        internal::alwaysFalse<P>,
        "bhattacharyya(p, q) has a closed form only between two Gaussian densities of one space, in any of the four "
        "forms (call toDensity() on the parameters, and copy() a GaussianMarginal first), or two Categorical "
        "views of one space; "
        "estimate it otherwise with bhattacharyya(MonteCarlo{count}, p, q, rng)"
    );
    return {};
}

/** \ingroup Divergences
 * \returns the Hellinger distance \f$ \sqrt{1 - e^{-D_B}} \f$ between two Gaussians, with
 * \f$ D_B \f$ their bhattacharyya() distance: a metric, bounded by 1. \a p and \a q are
 * densities of one space in any of the four forms, in any pair, as in kullbackLeibler().
 */
template <typename P, typename Q>
    requires internal::GaussianDensitiesOfOneSpace<P, Q>
[[nodiscard]] typename P::Scalar
hellinger(P const & p, Q const & q)
{
    return internal::hellingerFromBhattacharyya(bhattacharyya(p, q));
}

/** \ingroup Divergences
 * \returns the Hellinger distance \f$ \sqrt{1 - \sum_i \sqrt{p_i q_i}} \f$ between two
 * categorical distributions: 1 if they share no value.
 */
template <DiscreteSpace Space, typename Scalar>
[[nodiscard]] Scalar
hellinger(Categorical<Space, Scalar> const & p, Categorical<Space, Scalar> const & q)
{
    return internal::hellingerFromBhattacharyya(bhattacharyya(p, q));
}

/** \ingroup Divergences
 * \returns a MonteCarlo estimate of the Hellinger distance, from the same draws as
 * bhattacharyya(MonteCarlo const &, P const &, Q const &, G &).
 */
template <typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
hellinger(MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    return hellinger(execution::sequential, method, p, q, rng);
}

/** \ingroup Divergences
 * The MonteCarlo hellinger() under \a policy, as the MonteCarlo kullbackLeibler() runs under one.
 */
template <ExecutionPolicy Policy, typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
hellinger(Policy && policy, MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    using Scalar = typename P::Scalar;
    auto const terms = internal::bhattacharyyaCoefficientTerms<Scalar>(std::forward<Policy>(policy), method, p, q, rng);
    Scalar const distance = Eigen::numext::sqrt(Eigen::numext::maxi(Scalar(1) - terms.mean(), Scalar(0)));
    // dH/dc = -1 / (2H); at H = 0 the first-order error vanishes, and H is of the order of the
    // square root of the coefficient's error.
    Scalar const standardError = distance > Scalar(0) ? terms.standardError() / (Scalar(2) * distance)
                                                      : Eigen::numext::sqrt(terms.standardError());
    return {distance, standardError, method.count};
}

/** \ingroup Divergences
 * hellinger() for a pair with no closed form: a compile-time error naming the estimator.
 */
template <typename P, typename Q>
[[nodiscard]] internal::NoClosedForm
hellinger(P const &, Q const &)
{
    static_assert(
        internal::alwaysFalse<P>,
        "hellinger(p, q) has a closed form only between two Gaussian densities of one space, in any of the four "
        "forms (call toDensity() on the parameters, and copy() a GaussianMarginal first), or two Categorical "
        "views of one space; "
        "estimate it otherwise with hellinger(MonteCarlo{count}, p, q, rng)"
    );
    return {};
}

/** \ingroup Divergences
 * \returns the Jensen-Shannon divergence between two categorical distributions,
 * \f$ \tfrac12 \mathrm{KL}(p \parallel m) + \tfrac12 \mathrm{KL}(q \parallel m) \f$ with
 * \f$ m = \tfrac12 (p + q) \f$: \f$ \log 2 \f$ if they share no value. Its square root is a
 * metric.
 */
template <DiscreteSpace Space, typename Scalar>
[[nodiscard]] Scalar
jensenShannon(Categorical<Space, Scalar> const & p, Categorical<Space, Scalar> const & q)
{
    EMRYS_FILTER_ASSERT(p.cardinality() == q.cardinality() && "the distributions must have the same values");
    Scalar divergence = 0;
    for (Eigen::Index i = 0; i < p.cardinality(); ++i)
    {
        Scalar const logP = p.logProbability(i);
        Scalar const logQ = q.logProbability(i);
        Scalar const logMixture = internal::logEvenMixture(logP, logQ);
        if (logP != -internal::infinity<Scalar>())
        {
            divergence += Eigen::numext::exp(logP) * (logP - logMixture);
        }
        if (logQ != -internal::infinity<Scalar>())
        {
            divergence += Eigen::numext::exp(logQ) * (logQ - logMixture);
        }
    }
    Scalar const bound = Eigen::numext::log(Scalar(2));
    return Eigen::numext::mini(Eigen::numext::maxi(Scalar(0.5) * divergence, Scalar(0)), bound);
}

/** \ingroup Divergences
 * \returns a MonteCarlo estimate of the Jensen-Shannon divergence: \c count draws from each of
 * \a p and \a q, each evaluated under both. Both must be anything drawSampleWithLogDensity()
 * takes, and both must be a density (HasLogDensity). This is the only form for two Gaussians,
 * whose even mixture has no closed-form entropy.
 */
template <typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G> && internal::MonteCarloComparable<Q, P, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
jensenShannon(MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    return jensenShannon(execution::sequential, method, p, q, rng);
}

/** \ingroup Divergences
 * The MonteCarlo jensenShannon() under \a policy, as the MonteCarlo kullbackLeibler() runs under
 * one.
 */
template <ExecutionPolicy Policy, typename P, typename Q, std::uniform_random_bit_generator G>
    requires internal::MonteCarloComparable<P, Q, G> && internal::MonteCarloComparable<Q, P, G>
[[nodiscard]] MonteCarloEstimate<typename P::Scalar>
jensenShannon(Policy && policy, MonteCarlo const & method, P const & p, Q const & q, G & rng)
{
    using Scalar = typename P::Scalar;
    auto const fromP = internal::mixtureDivergenceTerms<Scalar>(policy, method, p, q, rng);
    auto const fromQ = internal::mixtureDivergenceTerms<Scalar>(policy, method, q, p, rng);
    Scalar const standardError =
        Scalar(0.5)
        * Eigen::numext::sqrt(
            fromP.standardError() * fromP.standardError() + fromQ.standardError() * fromQ.standardError()
        );
    return {Scalar(0.5) * (fromP.mean() + fromQ.mean()), standardError, method.count};
}

/** \ingroup Divergences
 * jensenShannon() for a pair with no closed form: a compile-time error naming the estimator.
 */
template <typename P, typename Q>
[[nodiscard]] internal::NoClosedForm
jensenShannon(P const &, Q const &)
{
    static_assert(
        internal::alwaysFalse<P>,
        "jensenShannon(p, q) has a closed form only between two Categorical views of one space; between Gaussians "
        "or anything else, estimate it with jensenShannon(MonteCarlo{count}, p, q, rng)"
    );
    return {};
}

/** \ingroup Divergences
 * \returns the 2-Wasserstein distance between two Gaussians,
 * \f[ W_2^2 = \|\mu_p \boxminus \mu_q\|^2 + \mathrm{tr}\,\Sigma_p + \mathrm{tr}\,\Sigma_q
 *     - 2 \, \mathrm{tr}\big( (\Sigma_q^{1/2} \Sigma_p \Sigma_q^{1/2})^{1/2} \big), \f]
 * measured in the space's own units. The last trace is the nuclear norm of
 * \f$ A_p^T A_q \f$, with \f$ A \f$ the roots of the covariances that kullbackLeibler() reads,
 * so it takes one singular-value decomposition and no matrix square root. \a p and \a q are
 * densities of one space in any of the four forms, in any pair, as in kullbackLeibler().
 *
 * \sa GroundMetric, under which the overload that takes one measures
 */
template <typename P, typename Q>
    requires internal::GaussianDensitiesOfOneSpace<P, Q>
[[nodiscard]] typename P::Scalar
wasserstein(P const & p, Q const & q)
{
    using MatrixType = typename P::MatrixType;
    internal::assertSameDimensions(p, q);
    typename P::DifferenceVectorType const difference = Emrys::minus(p.space(), p.mean(), q.mean());
    MatrixType const first = internal::covarianceRootOf(p);
    MatrixType const second = internal::covarianceRootOf(q);
    return internal::buresWasserstein<typename P::Scalar>(difference, first, second);
}

/** \ingroup Divergences
 * \returns the 2-Wasserstein distance between two Gaussians under the ground metric
 * \f$ \|W \delta\| \f$: their distance after the map \f$ W \f$, with a column per difference
 * coordinate and any number of rows. A \f$ W \f$ of fewer rows, or with rows of zeros, measures
 * only the combinations it names, so two different distributions can be at distance zero: under a
 * singular \f$ W \f$ this is a pseudometric, and a threshold at zero does not tell them apart.
 */
template <typename P, typename Q, typename WeightsDerived>
    requires internal::GaussianDensitiesOfOneSpace<P, Q>
[[nodiscard]] typename P::Scalar
wasserstein(P const & p, Q const & q, Eigen::MatrixBase<WeightsDerived> const & weights)
{
    using Scalar = typename P::Scalar;
    using DensityType = P;
    using Measured =
        internal::MatrixOf<Scalar, WeightsDerived::RowsAtCompileTime, 1, WeightsDerived::MaxRowsAtCompileTime, 1>;
    using Factor = internal::MatrixOf<
        Scalar,
        WeightsDerived::RowsAtCompileTime,
        DensityType::DifferenceDimensionAtCompileTime,
        WeightsDerived::MaxRowsAtCompileTime,
        DensityType::MaxDifferenceDimensionAtCompileTime>;
    internal::assertSameDimensions(p, q);
    EMRYS_FILTER_ASSERT(
        weights.cols() == p.differenceDimension() && "the weights must have a column per difference coordinate"
    );
    typename DensityType::DifferenceVectorType const difference = Emrys::minus(p.space(), p.mean(), q.mean());
    Measured const measured = internal::multiply(weights, difference);
    Factor const first = internal::multiply(
        weights,
        internal::asTriangular<internal::covarianceRootTriangle<P>>(internal::covarianceRootOf(p))
    );
    Factor const second = internal::multiply(
        weights,
        internal::asTriangular<internal::covarianceRootTriangle<Q>>(internal::covarianceRootOf(q))
    );
    return internal::buresWasserstein<Scalar>(measured, first, second);
}

/** \ingroup Divergences
 * wasserstein() under the ground metric \a metric (see GroundMetric).
 */
template <typename P, typename Q>
    requires internal::GaussianDensitiesOfOneSpace<P, Q>
[[nodiscard]] typename P::Scalar
wasserstein(P const & p, Q const & q, GroundMetric<typename P::SpaceType, typename P::Scalar> const & metric)
{
    return wasserstein(p, q, metric.matrix());
}

/** \ingroup Divergences
 * wasserstein() for a pair with no closed form: a compile-time error.
 */
template <typename P, typename Q>
[[nodiscard]] internal::NoClosedForm
wasserstein(P const &, Q const &)
{
    static_assert(
        internal::alwaysFalse<P>,
        "wasserstein(p, q) has a closed form only between two Gaussian densities of one space, in any of the "
        "four forms (call toDensity() on the parameters, and copy() a GaussianMarginal first); between "
        "mixtures or particle sets it needs an optimal-transport solver, "
        "which the library does not have. expectedSquaredError(p, truth) gives the distance to a point"
    );
    return {};
}

/** \ingroup Divergences
 * \returns the expected squared error of \a distribution about \a truth, a value of its space:
 * \f$ E_p \|x \boxminus x^*\|^2 = \|\mu \boxminus x^*\|^2 + \mathrm{tr}\,\Sigma \f$, which is
 * \f$ W_2^2 \f$ between \f$ p \f$ and a point mass at \f$ x^* \f$. It needs only
 * \c moments(), so it is exact for any distribution that has them: a %Gaussian, a mixture, a
 * kernel density, a particle set, a hybrid state. Where \c moments() returns a
 * \c std::expected, as a ParticleSet's does, so does this, with the same error.
 *
 * Moments in a space that is not linear are tangent-space moments, so on a manifold this is
 * the expected squared error to first order.
 */
template <internal::HasMoments Distribution, typename TruthDerived>
[[nodiscard]] auto
expectedSquaredError(Distribution const & distribution, Eigen::MatrixBase<TruthDerived> const & truth)
{
    return internal::fromMoments(distribution, [&](auto const & moments) {
        return internal::expectedSquaredErrorOf(distribution.space(), moments, truth);
    });
}

/** \ingroup Divergences
 * expectedSquaredError() under the ground metric \f$ \|W \delta\| \f$:
 * \f$ \|W (\mu \boxminus x^*)\|^2 + \mathrm{tr}(W \Sigma W^T) \f$, with \a weights a matrix
 * with a column per difference coordinate.
 */
template <internal::HasMoments Distribution, typename TruthDerived, typename WeightsDerived>
[[nodiscard]] auto
expectedSquaredError(
    Distribution const & distribution,
    Eigen::MatrixBase<TruthDerived> const & truth,
    Eigen::MatrixBase<WeightsDerived> const & weights
)
{
    return internal::fromMoments(distribution, [&](auto const & moments) {
        return internal::expectedSquaredErrorOf(distribution.space(), moments, truth, weights);
    });
}

/** \ingroup Divergences
 * expectedSquaredError() under the ground metric \a metric (see GroundMetric).
 */
template <internal::HasMoments Distribution, typename TruthDerived>
[[nodiscard]] auto
expectedSquaredError(
    Distribution const & distribution,
    Eigen::MatrixBase<TruthDerived> const & truth,
    GroundMetric<typename Distribution::SpaceType, typename Distribution::Scalar> const & metric
)
{
    return expectedSquaredError(distribution, truth, metric.matrix());
}

}  // namespace Emrys
